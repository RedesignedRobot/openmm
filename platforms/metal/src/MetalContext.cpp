/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2026 Stanford University and the Authors.      *
 * Portions copyright (c) 2020-2023 Advanced Micro Devices, Inc.              *
 * Authors: Peter Eastman, Nicholas Curtis                                    *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "MetalContext.h"
#include "MetalArray.h"
#include "MetalBondedUtilities.h"
#include "MetalEvent.h"
#include "MetalFFT3D.h"
#include "MetalIntegrationUtilities.h"
#include "MetalKernels.h"
#include "MetalKernelSources.h"
#include "MetalNonbondedUtilities.h"
#include "MetalProgram.h"
#include "MetalQueue.h"
#include "MetalSort.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/MonteCarloFlexibleBarostat.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VirtualSite.h"
#include "MetalExpressionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stack>
#include <thread>
#include <typeinfo>
#include <sys/stat.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

using namespace OpenMM;
using namespace std;

const int MetalContext::ThreadBlockSize = 64;
const int MetalContext::TileSize = 32;

/**
 * Metal does not report the number of GPU cores, so read it from the IORegistry.
 */
static int getGpuCoreCount() {
    int cores = 0;
    io_iterator_t iterator;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iterator) != KERN_SUCCESS)
        return 0;
    io_object_t entry;
    while ((entry = IOIteratorNext(iterator)) != 0) {
        CFTypeRef value = IORegistryEntryCreateCFProperty(entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (value != NULL) {
            if (CFGetTypeID(value) == CFNumberGetTypeID())
                CFNumberGetValue((CFNumberRef) value, kCFNumberIntType, &cores);
            CFRelease(value);
        }
        IOObjectRelease(entry);
    }
    IOObjectRelease(iterator);
    return cores;
}

static bool isIdentifierChar(char c) {
    return isalnum((unsigned char) c) || c == '_';
}

static size_t findWord(const string& source, const string& word, size_t start) {
    size_t pos = source.find(word, start);
    while (pos != string::npos) {
        bool startsWord = (pos == 0 || !isIdentifierChar(source[pos-1]));
        bool endsWord = (pos+word.size() == source.size() || !isIdentifierChar(source[pos+word.size()]));
        if (startsWord && endsWord)
            return pos;
        pos = source.find(word, pos+1);
    }
    return string::npos;
}

static string replaceWord(const string& source, const string& word, const string& replacement) {
    string result;
    size_t pos = 0, found;
    while ((found = findWord(source, word, pos)) != string::npos) {
        result += source.substr(pos, found-pos)+replacement;
        pos = found+word.size();
    }
    return result+source.substr(pos);
}

/**
 * Rewrite one kernel parameter for Metal.  A pointer without an address space points to device
 * memory.  A value becomes a constant reference, so "real4 box" becomes "constant real4& _in_box",
 * and a statement copying it into a local variable with the original name is appended to the
 * prologue.  executeKernel() passes the arguments whose names start with _in_ by value.
 */
static string rewriteParameter(const string& param, string& prologue) {
    size_t first = param.find_first_not_of(" \t\r\n");
    if (first == string::npos)
        return param;
    size_t last = param.find_last_not_of(" \t\r\n");
    string declaration = param.substr(first, last-first+1);
    if (declaration.find_first_of("&[") != string::npos)
        return param;
    stringstream tokenStream(declaration);
    vector<string> tokens;
    string token;
    while (tokenStream >> token)
        if (token != "const")
            tokens.push_back(token);
    if (declaration.find('*') != string::npos) {
        for (string space : {"GLOBAL", "LOCAL", "LOCAL_ARG", "device", "threadgroup", "constant"})
            if (find(tokens.begin(), tokens.end(), space) != tokens.end())
                return param;
        return param.substr(0, first)+"device "+param.substr(first);
    }
    if (tokens.size() < 2)
        return param;
    string type;
    for (int i = 0; i < tokens.size()-1; i++)
        type += (i == 0 ? "" : " ")+tokens[i];
    const string& name = tokens.back();
    prologue += "    "+type+" "+name+" = _in_"+name+";\n";
    return param.substr(0, first)+"constant "+type+"& _in_"+name+param.substr(last+1);
}

/**
 * Rewrite the parameters of every kernel in a source file (see rewriteParameter()).  Preprocessor
 * directives inside a parameter list are copied into the prologue as well, so each copy statement
 * is compiled exactly when the parameter it reads is.
 */
static string rewriteKernelSignatures(const string& source) {
    string result;
    size_t pos = 0;
    size_t kernelStart;
    while ((kernelStart = min(findWord(source, "KERNEL", pos), findWord(source, "__global__", pos))) != string::npos) {
        size_t open = source.find('(', findWord(source, "void", kernelStart));
        if (open == string::npos)
            break;
        result += source.substr(pos, open+1-pos);
        string prologue, param;
        size_t i = open+1;
        int depth = 0;
        for (; i < source.size(); i++) {
            char c = source[i];
            if (c == '#') {
                size_t end = source.find('\n', i);
                string directive = source.substr(i, end-i);
                result += rewriteParameter(param, prologue)+directive;
                prologue += directive+"\n";
                param.clear();
                i = end-1;
            }
            else if (c == '(') {
                depth++;
                param += c;
            }
            else if (c == ')' && depth > 0) {
                depth--;
                param += c;
            }
            else if (c == ')' || (c == ',' && depth == 0)) {
                result += rewriteParameter(param, prologue)+c;
                param.clear();
                if (c == ')')
                    break;
            }
            else
                param += c;
        }
        size_t brace = source.find('{', i);
        if (brace == string::npos)
            throw OpenMMException("Error parsing kernel signature: "+source.substr(kernelStart, open-kernelStart));
        result += source.substr(i+1, brace+1-(i+1))+"\n"+prologue;
        pos = brace+1;
    }
    result += source.substr(pos);
    return result;
}

MetalContext::MetalContext(const System& system, int deviceIndex, bool useBlockingSync, const string& precision, const string& tempDir, MetalPlatform::PlatformData& platformData,
        MetalContext* originalContext) : ComputeContext(system), platformData(platformData), contextIsValid(false),
        pinnedBuffer(NULL), integration(NULL), expression(NULL), bonded(NULL), nonbonded(NULL),
        useBlockingSync(useBlockingSync), supportsHardwareFloatGlobalAtomicAdd(false) {
    if (precision == "single") {
        useDoublePrecision = false;
        useMixedPrecision = false;
    }
    else if (precision == "mixed") {
        useDoublePrecision = false;
        useMixedPrecision = true;
    }
    else if (precision == "double") {
        useDoublePrecision = true;
        useMixedPrecision = false;
    }
    else
        throw OpenMMException("Illegal value for Precision: "+precision);
    if (useDoublePrecision || useMixedPrecision)
        throw OpenMMException("This device does not support double precision");
#ifdef WIN32
    this->tempDir = tempDir+"\\";
#else
    this->tempDir = tempDir+"/";
#endif
    contextIndex = platformData.contexts.size();
    if (originalContext == NULL) {
        // A Mac has one GPU.  MTL::CopyAllDevices() would return NULL here, because metal-cpp
        // compiles it out below a 10.11 deployment target and OpenMM defaults to 10.7.

        device = MTL::CreateSystemDefaultDevice();
        int numDevices = (device == NULL ? 0 : 1);
        if (deviceIndex < -1 || deviceIndex >= numDevices)
            throw OpenMMException("Illegal value for DeviceIndex: "+intToString(deviceIndex));
        this->deviceIndex = 0;
        defaultQueue = shared_ptr<ComputeQueueImpl>(new MetalQueue(device));
    }
    else {
        this->deviceIndex = originalContext->deviceIndex;
        this->device = originalContext->device->retain();
        defaultQueue = originalContext->defaultQueue;
    }
    currentQueue = defaultQueue;

    // set device properties
    this->simdWidth = 32;
    this->sharedMemPerBlock = device->maxThreadgroupMemoryLength();

    // METAL-TODO: find a good value here
    int numThreadBlocksPerComputeUnit = 12;

    contextIsValid = true;
    ContextSelector selector(*this);
    numAtoms = system.getNumParticles();
    paddedNumAtoms = TileSize*((numAtoms+TileSize-1)/TileSize);
    numAtomBlocks = (paddedNumAtoms+(TileSize-1))/TileSize;
    multiprocessors = getGpuCoreCount();
    if (multiprocessors == 0)
        multiprocessors = 8; // The smallest Apple GPU with Metal 3.1 support has 7 or 8 cores.
    numThreadBlocks = numThreadBlocksPerComputeUnit*multiprocessors;

    compilationDefines["USE_METAL"] = "1";
    if (simdWidth == 32)
        compilationDefines["AMD_RDNA"] = "1";
    if (useDoublePrecision) {
        compilationDefines["USE_DOUBLE_PRECISION"] = "1";
        compilationDefines["make_real2"] = "make_double2";
        compilationDefines["make_real3"] = "make_double3";
        compilationDefines["make_real4"] = "make_double4";
        compilationDefines["make_mixed2"] = "make_double2";
        compilationDefines["make_mixed3"] = "make_double3";
        compilationDefines["make_mixed4"] = "make_double4";
    }
    else if (useMixedPrecision) {
        compilationDefines["USE_MIXED_PRECISION"] = "1";
        compilationDefines["make_real2"] = "make_float2";
        compilationDefines["make_real3"] = "make_float3";
        compilationDefines["make_real4"] = "make_float4";
        compilationDefines["make_mixed2"] = "make_double2";
        compilationDefines["make_mixed3"] = "make_double3";
        compilationDefines["make_mixed4"] = "make_double4";
    }
    else {
        compilationDefines["make_real2"] = "make_float2";
        compilationDefines["make_real3"] = "make_float3";
        compilationDefines["make_real4"] = "make_float4";
        compilationDefines["make_mixed2"] = "make_float2";
        compilationDefines["make_mixed3"] = "make_float3";
        compilationDefines["make_mixed4"] = "make_float4";
    }

    // Set defines based on the requested precision.

    compilationDefines["SQRT"] = useDoublePrecision ? "sqrt" : "__fsqrt_rn";
    compilationDefines["RSQRT"] = useDoublePrecision ? "rsqrt" : "__frsqrt_rn";
    compilationDefines["RECIP(x)"] = useDoublePrecision ? "(1.0/(x))" : "fast::divide(1.0f, (x))";
    compilationDefines["EXP"] = useDoublePrecision ? "exp" : "__expf";
    compilationDefines["LOG"] = useDoublePrecision ? "log" : "__logf";
    compilationDefines["POW"] = useDoublePrecision ? "pow" : "powf";
    compilationDefines["COS"] = useDoublePrecision ? "cos" : "cosf";
    compilationDefines["SIN"] = useDoublePrecision ? "sin" : "sinf";
    compilationDefines["TAN"] = useDoublePrecision ? "tan" : "tanf";
    compilationDefines["ACOS"] = useDoublePrecision ? "acos" : "acosf";
    compilationDefines["ASIN"] = useDoublePrecision ? "asin" : "asinf";
    compilationDefines["ATAN"] = useDoublePrecision ? "atan" : "atanf";
    compilationDefines["ERF"] = useDoublePrecision ? "erf" : "erff";
    compilationDefines["ERFC"] = useDoublePrecision ? "erfc" : "erfcf";
    compilationDefines["FMA"] = useDoublePrecision ? "fma" : "fmaf";
    compilationDefines["FABS"] = useDoublePrecision ? "fabs" : "fabsf";

    // Set defines for applying periodic boundary conditions.

    Vec3 boxVectors[3];
    system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    boxIsTriclinic = (boxVectors[0][1] != 0.0 || boxVectors[0][2] != 0.0 ||
                      boxVectors[1][0] != 0.0 || boxVectors[1][2] != 0.0 ||
                      boxVectors[2][0] != 0.0 || boxVectors[2][1] != 0.0);
    for (int i = 0; i < system.getNumForces(); i++)
        if (dynamic_cast<const MonteCarloFlexibleBarostat*>(&system.getForce(i)) != NULL)
            boxIsTriclinic = true;
    if (boxIsTriclinic) {
        compilationDefines["APPLY_PERIODIC_TO_DELTA(delta)"] =
            "{"
            "real scale3 = floor(delta.z*invPeriodicBoxSize.z+0.5f); \\\n"
            "delta.x -= scale3*periodicBoxVecZ.x; \\\n"
            "delta.y -= scale3*periodicBoxVecZ.y; \\\n"
            "delta.z -= scale3*periodicBoxVecZ.z; \\\n"
            "real scale2 = floor(delta.y*invPeriodicBoxSize.y+0.5f); \\\n"
            "delta.x -= scale2*periodicBoxVecY.x; \\\n"
            "delta.y -= scale2*periodicBoxVecY.y; \\\n"
            "real scale1 = floor(delta.x*invPeriodicBoxSize.x+0.5f); \\\n"
            "delta.x -= scale1*periodicBoxVecX.x;}";
        compilationDefines["APPLY_PERIODIC_TO_POS(pos)"] =
            "{"
            "real scale3 = floor(pos.z*invPeriodicBoxSize.z); \\\n"
            "pos.x -= scale3*periodicBoxVecZ.x; \\\n"
            "pos.y -= scale3*periodicBoxVecZ.y; \\\n"
            "pos.z -= scale3*periodicBoxVecZ.z; \\\n"
            "real scale2 = floor(pos.y*invPeriodicBoxSize.y); \\\n"
            "pos.x -= scale2*periodicBoxVecY.x; \\\n"
            "pos.y -= scale2*periodicBoxVecY.y; \\\n"
            "real scale1 = floor(pos.x*invPeriodicBoxSize.x); \\\n"
            "pos.x -= scale1*periodicBoxVecX.x;}";
        compilationDefines["APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)"] =
            "{"
            "real scale3 = floor((pos.z-center.z)*invPeriodicBoxSize.z+0.5f); \\\n"
            "pos.x -= scale3*periodicBoxVecZ.x; \\\n"
            "pos.y -= scale3*periodicBoxVecZ.y; \\\n"
            "pos.z -= scale3*periodicBoxVecZ.z; \\\n"
            "real scale2 = floor((pos.y-center.y)*invPeriodicBoxSize.y+0.5f); \\\n"
            "pos.x -= scale2*periodicBoxVecY.x; \\\n"
            "pos.y -= scale2*periodicBoxVecY.y; \\\n"
            "real scale1 = floor((pos.x-center.x)*invPeriodicBoxSize.x+0.5f); \\\n"
            "pos.x -= scale1*periodicBoxVecX.x;}";
    }
    else {
        compilationDefines["APPLY_PERIODIC_TO_DELTA(delta)"] =
            "{"
            "delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x; \\\n"
            "delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y; \\\n"
            "delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;}";
        compilationDefines["APPLY_PERIODIC_TO_POS(pos)"] =
            "{"
            "pos.x -= floor(pos.x*invPeriodicBoxSize.x)*periodicBoxSize.x; \\\n"
            "pos.y -= floor(pos.y*invPeriodicBoxSize.y)*periodicBoxSize.y; \\\n"
            "pos.z -= floor(pos.z*invPeriodicBoxSize.z)*periodicBoxSize.z;}";
        compilationDefines["APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)"] =
            "{"
            "pos.x -= floor((pos.x-center.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x; \\\n"
            "pos.y -= floor((pos.y-center.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y; \\\n"
            "pos.z -= floor((pos.z-center.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;}";
    }
    initializeKernels();

    // Create utilities objects.

    bonded = new MetalBondedUtilities(*this);
    nonbonded = new MetalNonbondedUtilities(*this);
    integration = new MetalIntegrationUtilities(*this, system);
    expression = new MetalExpressionUtilities(*this);
    clearBuffer(posq);
}

MetalContext::~MetalContext() {
    pushAsCurrent();
    for (auto force : forces)
        delete force;
    for (auto listener : reorderListeners)
        delete listener;
    for (auto computation : preComputations)
        delete computation;
    for (auto computation : postComputations)
        delete computation;
    if (pinnedBuffer != NULL)
        free(pinnedBuffer);
    if (integration != NULL)
        delete integration;
    if (expression != NULL)
        delete expression;
    if (bonded != NULL)
        delete bonded;
    if (nonbonded != NULL)
        delete nonbonded;
    for (auto module : loadedModules)
        module->release();
    for (auto& kernel : kernelArgumentSizes)
        kernel.first->release();
    popAsCurrent();
    contextIsValid = false;
    device->release();
}

void MetalContext::initialize() {
    ContextSelector selector(*this);
    int numEnergyBuffers = max(numThreadBlocks*ThreadBlockSize, nonbonded->getNumEnergyBuffers());
    if (useDoublePrecision) {
        energyBuffer.initialize<double>(*this, numEnergyBuffers, "energyBuffer");
        energySum.initialize<double>(*this, multiprocessors, "energySum");
        int pinnedBufferSize = max(paddedNumAtoms*4, numEnergyBuffers);
        pinnedBuffer = malloc(pinnedBufferSize*sizeof(double));
    }
    else if (useMixedPrecision) {
        energyBuffer.initialize<double>(*this, numEnergyBuffers, "energyBuffer");
        energySum.initialize<double>(*this, multiprocessors, "energySum");
        int pinnedBufferSize = max(paddedNumAtoms*4, numEnergyBuffers);
        pinnedBuffer = malloc(pinnedBufferSize*sizeof(double));
    }
    else {
        energyBuffer.initialize<float>(*this, numEnergyBuffers, "energyBuffer");
        energySum.initialize<float>(*this, multiprocessors, "energySum");
        int pinnedBufferSize = max(paddedNumAtoms*6, numEnergyBuffers);
        pinnedBuffer = malloc(pinnedBufferSize*sizeof(float));
    }
    for (int i = 0; i < numAtoms; i++) {
        double mass = system.getParticleMass(i);
        if (useDoublePrecision || useMixedPrecision)
            ((double4*) pinnedBuffer)[i] = make_double4(0.0, 0.0, 0.0, mass == 0.0 ? 0.0 : 1.0/mass);
        else
            ((float4*) pinnedBuffer)[i] = make_float4(0.0f, 0.0f, 0.0f, mass == 0.0 ? 0.0f : (float) (1.0/mass));
    }
    velm.upload(pinnedBuffer);
    bonded->initialize(system);
    addAutoclearBuffer(longForceBuffer);
    addAutoclearBuffer(energyBuffer);
    int numEnergyParamDerivs = energyParamDerivNames.size();
    if (numEnergyParamDerivs > 0) {
        if (useDoublePrecision || useMixedPrecision)
            energyParamDerivBuffer.initialize<double>(*this, numEnergyParamDerivs*numEnergyBuffers, "energyParamDerivBuffer");
        else
            energyParamDerivBuffer.initialize<float>(*this, numEnergyParamDerivs*numEnergyBuffers, "energyParamDerivBuffer");
        addAutoclearBuffer(energyParamDerivBuffer);
    }
    findMoleculeGroups();
    nonbonded->initialize(system);
}

void MetalContext::initializeContexts() {
    getPlatformData().initializeContexts(system);
}

void MetalContext::setAsCurrent() {
}

void MetalContext::pushAsCurrent() {
}

void MetalContext::popAsCurrent() {
}

string MetalContext::getTempFileName() const {
    stringstream tempFileName;
    tempFileName << tempDir;
    tempFileName << "openmmTempKernel" << this; // Include a pointer to this context as part of the filename to avoid collisions.
    tempFileName << "_" << std::this_thread::get_id();
    return tempFileName.str();
}

metalModule_t MetalContext::createModule(const string source) {
    return createModule(source, map<string, string>());
}

metalModule_t MetalContext::createModule(const string source, const map<string, string>& defines) {
    const char* saveTempsEnv = getenv("OPENMM_SAVE_TEMPS");
    const bool saveTemps = saveTempsEnv != nullptr && string(saveTempsEnv) == "1";

    stringstream src;
    for (auto& pair : compilationDefines) {
        // Query defines to avoid duplicate variables
        if (defines.find(pair.first) == defines.end()) {
            src << "#define " << pair.first;
            if (!pair.second.empty())
                src << " " << pair.second;
            src << endl;
        }
    }
    if (!compilationDefines.empty())
        src << endl;
    src << "#include <metal_stdlib>\n";
    src << "using namespace metal;\n";
    if (useDoublePrecision) {
        src << "typedef double real;\n";
        src << "typedef double2 real2;\n";
        src << "typedef double3 real3;\n";
        src << "typedef double4 real4;\n";
    }
    else {
        src << "typedef float real;\n";
        src << "typedef float2 real2;\n";
        src << "typedef float3 real3;\n";
        src << "typedef float4 real4;\n";
    }
    if (useDoublePrecision || useMixedPrecision) {
        src << "typedef double mixed;\n";
        src << "typedef double2 mixed2;\n";
        src << "typedef double3 mixed3;\n";
        src << "typedef double4 mixed4;\n";
    }
    else {
        src << "typedef float mixed;\n";
        src << "typedef float2 mixed2;\n";
        src << "typedef float3 mixed3;\n";
        src << "typedef float4 mixed4;\n";
    }
    src << "typedef unsigned int tileflags;\n";
    src << MetalKernelSources::common << endl;
    // Common kernels use "thread" as a variable name, but in MSL it is an address space.
    for (auto& pair : defines) {
        src << "#define " << pair.first;
        if (!pair.second.empty())
            src << " " << replaceWord(pair.second, "thread", "_mmThread");
        src << endl;
    }
    if (!defines.empty())
        src << endl;
    src << MetalKernelSources::intrinsics << endl;
    // MSL has no long long.  Its long is 64 bits.
    src << rewriteKernelSignatures(replaceWord(replaceWord(source, "thread", "_mmThread"), "long long", "long")) << endl;

    // Select names for the various temporary files.

    if (saveTemps) {
        string inputFile = getTempFileName()+"_"+to_string(hash<string>()(src.str()))+".metal";
        std::cout << "Source code: " << inputFile << std::endl;
        ofstream out(inputFile.c_str());
        out << src.str();
        out.close();
    }

    // Compile the program.  Autoreleased objects must not outlive the pool, so no exception may
    // be thrown until it is released.

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    options->setLanguageVersion(MTL::LanguageVersion3_2);
    options->setMathMode(MTL::MathModeSafe);
    options->setMathFloatingPointFunctions(MTL::MathFloatingPointFunctionsPrecise);
    NS::Error* error = NULL;
    metalModule_t module = device->newLibrary(NS::String::string(src.str().c_str(), NS::UTF8StringEncoding), options, &error);
    options->release();
    string log;
    if (module == NULL && error != NULL)
        log = error->localizedDescription()->utf8String();
    pool->release();
    if (module == NULL && log.find("no 'buffer' resource location available") != string::npos)
        throw OpenMMException("Error compiling program: a Metal kernel can have at most 31 array and value arguments, which this system exceeds.  "+log);
    if (module == NULL)
        throw OpenMMException("Error compiling program: "+log);
    loadedModules.push_back(module);
    return module;
}

metalFunction_t MetalContext::getKernel(metalModule_t module, const string& name) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Function* function = module->newFunction(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
    metalFunction_t kernel = NULL;
    MTL::ComputePipelineReflection* reflection = NULL;
    string log = "no kernel with this name";
    if (function != NULL) {
        NS::Error* error = NULL;
        kernel = device->newComputePipelineState(function, MTL::PipelineOptionBindingInfo, &reflection, &error);
        function->release();
        if (kernel == NULL && error != NULL)
            log = error->localizedDescription()->utf8String();
    }

    // Record which arguments are passed by value (see rewriteParameter()), and their sizes.

    vector<int> sizes;
    for (int i = 0; kernel != NULL && i < reflection->bindings()->count(); i++) {
        MTL::Binding* binding = reflection->bindings()->object<MTL::Binding>(i);
        if (binding->type() != MTL::BindingTypeBuffer)
            continue;
        sizes.resize(max((int) sizes.size(), (int) binding->index()+1), 0);
        if (string(binding->name()->utf8String()).rfind("_in_", 0) == 0)
            sizes[binding->index()] = ((MTL::BufferBinding*) binding)->bufferDataSize();
    }
    pool->release();
    if (kernel == NULL)
        throw OpenMMException("Error creating kernel "+name+": "+log);
    if (kernel->threadExecutionWidth() != getSIMDWidth()) {
        kernel->release();
        throw OpenMMException("Error creating kernel "+name+": the kernels require a SIMD width of "+intToString(getSIMDWidth()));
    }
    kernelArgumentSizes[kernel] = sizes;
    return kernel;
}

vector<ComputeContext*> MetalContext::getAllContexts() {
    vector<ComputeContext*> result;
    for (MetalContext* c : platformData.contexts)
        result.push_back(c);
    return result;
}

double& MetalContext::getEnergyWorkspace() {
    return platformData.contextEnergy[contextIndex];
}

ComputeQueue MetalContext::createQueue() {
    return shared_ptr<ComputeQueueImpl>(new MetalQueue(device));
}

MetalQueue& MetalContext::getCurrentStream() {
    return *dynamic_cast<MetalQueue*>(currentQueue.get());
}

MetalArray* MetalContext::createArray() {
    return new MetalArray();
}

ComputeEvent MetalContext::createEvent() {
    return shared_ptr<ComputeEventImpl>(new MetalEvent(*this));
}

ComputeSort MetalContext::createSort(ComputeSortImpl::SortTrait* trait, unsigned int length, bool uniform) {
    return shared_ptr<ComputeSortImpl>(new MetalSort(*this, trait, length, uniform));
}

FFT3D MetalContext::createFFT(int xsize, int ysize, int zsize, bool realToComplex) {
    return FFT3D(new MetalFFT3D(*this, xsize, ysize, zsize, realToComplex));
}

int MetalContext::findLegalFFTDimension(int minimum) {
    return MetalFFT3D::findLegalDimension(minimum);
}

ComputeProgram MetalContext::compileProgram(const std::string source, const std::map<std::string, std::string>& defines) {
    metalModule_t module = createModule(MetalKernelSources::vectorOps+source, defines);
    return shared_ptr<ComputeProgramImpl>(new MetalProgram(*this, module));
}

MetalArray& MetalContext::unwrap(ArrayInterface& array) const {
    MetalArray* cuarray;
    ComputeArray* wrapper = dynamic_cast<ComputeArray*>(&array);
    if (wrapper != NULL)
        cuarray = dynamic_cast<MetalArray*>(&wrapper->getArray());
    else
        cuarray = dynamic_cast<MetalArray*>(&array);
    if (cuarray == NULL)
        throw OpenMMException("Array argument is not an MetalArray");
    return *cuarray;
}

void MetalContext::executeKernel(metalFunction_t kernel, void** arguments, int threads, int blockSize, unsigned int sharedSize) {
    if (blockSize == -1)
        blockSize = ThreadBlockSize;
    int gridSize = std::min((threads+blockSize-1)/blockSize, numThreadBlocks);
    launchKernel(kernel, arguments, gridSize, blockSize, sharedSize);
}

void MetalContext::executeKernelFlat(metalFunction_t kernel, void** arguments, int threads, int blockSize, unsigned int sharedSize) {
    if (blockSize == -1)
        blockSize = ThreadBlockSize;
    int gridSize = (threads+blockSize-1)/blockSize;
    launchKernel(kernel, arguments, gridSize, blockSize, sharedSize);
}

void MetalContext::launchKernel(metalFunction_t kernel, void** arguments, int gridSize, int blockSize, unsigned int sharedSize) {
    MetalQueue& queue = getCurrentStream();
    lock_guard<recursive_mutex> guard(queue.getLock());
    MTL::ComputeCommandEncoder* encoder = queue.getEncoder();
    encoder->setComputePipelineState(kernel);
    const vector<int>& sizes = kernelArgumentSizes[kernel];
    for (int i = 0; i < sizes.size(); i++) {
        if (sizes[i] == 0)
            encoder->setBuffer(*(MTL::Buffer**) arguments[i], 0, i);
        else
            encoder->setBytes(arguments[i], sizes[i], i);
    }
    if (sharedSize > 0)
        encoder->setThreadgroupMemoryLength((sharedSize+15)/16*16, 0);
    encoder->dispatchThreadgroups(MTL::Size(gridSize, 1, 1), MTL::Size(blockSize, 1, 1));
}

int MetalContext::computeThreadBlockSize(double memory) const {
    int maxShared = this->sharedMemPerBlock;
    int max = (int) (maxShared/memory);
    if (max < MetalContext::ThreadBlockSize) {
        throw OpenMMException("Too much shared memory requested!");
    }
    int threads = this->simdWidth;
    while (threads+this->simdWidth < max)
        threads += this->simdWidth;
    return threads;
}

double MetalContext::reduceEnergy() {
    int workGroupSize = getMaxThreadBlockSize();
    reduceEnergyKernel->setArg(0, energyBuffer);
    reduceEnergyKernel->setArg(1, energySum);
    reduceEnergyKernel->setArg(2, energyBuffer.getSize());
    reduceEnergyKernel->setArg(3, workGroupSize);
    reduceEnergyKernel->execute(workGroupSize*energySum.getSize(), workGroupSize);
    energySum.download(pinnedBuffer);
    double result = 0;
    if (getUseDoublePrecision() || getUseMixedPrecision()) {
        for (int i = 0; i < energySum.getSize(); i++)
            result += ((double*) pinnedBuffer)[i];
    }
    else {
        for (int i = 0; i < energySum.getSize(); i++)
            result += ((float*) pinnedBuffer)[i];
    }
    return result;
}

void MetalContext::flushQueue() {
    getCurrentStream().commit();
}
