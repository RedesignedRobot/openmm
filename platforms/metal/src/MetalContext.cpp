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
#include "SHA1.h"
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
#include <IOKit/IOKitLib.h>
#include <regex>

using namespace OpenMM;
using namespace std;

const int MetalContext::ThreadBlockSize = 64;
const int MetalContext::TileSize = 32;

static string rewriteParameter(const string& param, string& prologue) {
    smatch m;
    if (regex_search(param, regex("\\b(GLOBAL|LOCAL|LOCAL_ARG|device|threadgroup|constant)\\b|[&\\[]")))
        return param;
    if (param.find('*') != string::npos)
        return "device "+param;
    if (!regex_match(param, m, regex("\\s*(const\\s+)?(\\S.*?)\\s+(\\w+)((\\s+[A-Z_]+)*)\\s*")))
        return param;
    prologue += m[2].str()+" "+m[3].str()+" = _in_"+m[3].str()+";\n";
    return "constant "+m[2].str()+"& _in_"+m[3].str()+m[4].str();
}

static string rewriteKernelSignatures(const string& source) {
    string result;
    size_t pos = 0;
    smatch m;
    while (regex_search(source.begin()+pos, source.end(), m, regex("\\b(KERNEL|__global__)\\b.*?\\bvoid\\s+\\w+\\s*\\("))) {
        size_t i = pos+m.position()+m.length();
        result += source.substr(pos, i-pos);
        string prologue, param;
        for (int depth = 0; depth >= 0; i++) {
            if (source[i] == '#') {
                size_t end = source.find('\n', i)+1;
                result += rewriteParameter(param, prologue)+"\n"+source.substr(i, end-i);
                prologue += source.substr(i, end-i);
                param.clear();
                i = end-1;
                continue;
            }
            depth += (source[i] == '(')-(source[i] == ')');
            if (depth < 0 || (source[i] == ',' && depth == 0)) {
                result += rewriteParameter(param, prologue)+source[i];
                param.clear();
            }
            else
                param += source[i];
        }
        pos = source.find('{', i)+1;
        result += source.substr(i, pos-i)+"\n"+prologue;
    }
    return result+source.substr(pos);
}

static string addThreadAddressSpace(const string& param) {
    if (param.find('*') == string::npos || regex_search(param, regex("\\b(GLOBAL|LOCAL|LOCAL_ARG|PRIVATE|device|threadgroup|constant|thread)\\b")))
        return param;
    return "thread "+param;
}

/**
 * Common kernels follow OpenCL, where a pointer parameter with no address space is private.
 * Metal requires an address space on every pointer, so give those parameters of device
 * functions the thread address space.
 */
static string rewriteDeviceSignatures(const string& source) {
    string result;
    size_t pos = 0;
    smatch m;
    while (regex_search(source.begin()+pos, source.end(), m, regex("\\b(DEVICE|__device__)\\b[^;{}()=]*?\\b\\w+\\s*\\("))) {
        size_t i = pos+m.position()+m.length();
        result += source.substr(pos, i-pos);
        string param;
        for (int depth = 0; depth >= 0; i++) {
            if (source[i] == '#') {
                size_t end = source.find('\n', i)+1;
                result += addThreadAddressSpace(param)+source.substr(i, end-i);
                param.clear();
                i = end-1;
                continue;
            }
            depth += (source[i] == '(')-(source[i] == ')');
            if (depth < 0 || (source[i] == ',' && depth == 0)) {
                result += addThreadAddressSpace(param)+source[i];
                param.clear();
            }
            else
                param += source[i];
        }
        pos = i;
    }
    return result+source.substr(pos);
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
#ifdef WIN32
    this->tempDir = tempDir+"\\";
#else
    this->tempDir = tempDir+"/";
#endif
    contextIndex = platformData.contexts.size();
    if (originalContext == NULL) {
        device = MTL::CreateSystemDefaultDevice();
        int numDevices = (device == NULL ? 0 : 1);
        if (deviceIndex < -1 || deviceIndex >= numDevices)
            throw OpenMMException("Illegal value for DeviceIndex: "+intToString(deviceIndex));
        this->deviceIndex = 0;
        defaultQueue = shared_ptr<ComputeQueueImpl>(new MetalQueue(device->newCommandQueue()));
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
    io_service_t gpu = IOServiceGetMatchingService(MACH_PORT_NULL, IOServiceMatching("AGXAccelerator"));
    CFNumberRef cores = (CFNumberRef) IORegistryEntryCreateCFProperty(gpu, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
    if (cores == NULL)
        throw OpenMMException("Error initializing Metal: the IORegistry has no GPU core count");
    CFNumberGetValue(cores, kCFNumberIntType, &multiprocessors);
    CFRelease(cores);
    IOObjectRelease(gpu);
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

string MetalContext::getHash(const string& src) const {
    CSHA1 sha1;
    sha1.Update((const UINT_8*) src.c_str(), src.size());
    sha1.Final();
    UINT_8 hash[20];
    sha1.GetHash(hash);
    stringstream cacheHash;
    cacheHash.flags(ios::hex);
    for (int i = 0; i < 20; i++)
        cacheHash << setw(2) << setfill('0') << (int) hash[i];
    return cacheHash.str();
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
    if (useMixedPrecision)
        src << MetalKernelSources::df64 << endl;
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
        // Metal has no double, so single precision kernels that name it get float.
        src << "#define double float\n";
        src << "typedef float mixed;\n";
        src << "typedef float2 mixed2;\n";
        src << "typedef float3 mixed3;\n";
        src << "typedef float4 mixed4;\n";
    }
    src << "typedef unsigned int tileflags;\n";
    src << MetalKernelSources::common << endl;
    for (auto& pair : defines) {
        src << "#define " << pair.first;
        if (!pair.second.empty())
            src << " " << regex_replace(pair.second, regex("\\bthread\\b"), "_mmThread");
        src << endl;
    }
    if (!defines.empty())
        src << endl;
    src << MetalKernelSources::intrinsics << endl;
    src << rewriteKernelSignatures(rewriteDeviceSignatures(regex_replace(regex_replace(source, regex("\\bthread\\b"), "_mmThread"), regex("\\blong long\\b"), "long"))) << endl;

    // Select names for the various temporary files.

    if (saveTemps) {
        stringstream tempFileName;
        const char* saveTempsPrefixEnv = getenv("OPENMM_SAVE_TEMPS_PREFIX");
        if (saveTempsPrefixEnv) {
            tempFileName << saveTempsPrefixEnv;
        }
        tempFileName << getHash(src.str());

        string inputFile = (tempFileName.str()+".metal");
        std::cout << "Source code: " << inputFile << std::endl;
        ofstream out(inputFile.c_str());
        out << src.str();
        out.close();
    }

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    options->setLanguageVersion(MTL::LanguageVersion3_2);
    options->setMathMode(MTL::MathModeSafe);
    options->setMathFloatingPointFunctions(MTL::MathFloatingPointFunctionsPrecise);
    NS::Error* error = NULL;
    metalModule_t module = device->newLibrary(NS::String::string(src.str().c_str(), NS::UTF8StringEncoding), options, &error);
    options->release();
    string log;
    if (module == NULL)
        log = error->localizedDescription()->utf8String();
    pool->release();
    if (module == NULL && log.find("no 'buffer' resource location available") != string::npos)
        throw OpenMMException("Error compiling program: a Metal kernel can have at most 31 array and value arguments, which this system exceeds.  "+log);
    if (module == NULL)
        throw OpenMMException("Error compiling program: "+log);
    loadedModules.push_back(module);
    return module;
}

metalFunction_t MetalContext::getKernel(metalModule_t& module, const string& name) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Function* function = module->newFunction(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
    MTL::ComputePipelineReflection* reflection = NULL;
    metalFunction_t kernel = (function == NULL ? NULL : device->newComputePipelineState(function, MTL::PipelineOptionBindingInfo, &reflection, NULL));
    vector<int> sizes;
    for (int i = 0; kernel != NULL && i < reflection->bindings()->count(); i++) {
        MTL::Binding* binding = reflection->bindings()->object<MTL::Binding>(i);
        if (binding->type() != MTL::BindingTypeBuffer)
            continue;
        sizes.resize(max((int) sizes.size(), (int) binding->index()+1), 0);
        if (string(binding->name()->utf8String()).rfind("_in_", 0) == 0)
            sizes[binding->index()] = ((MTL::BufferBinding*) binding)->bufferDataSize();
    }
    if (function != NULL)
        function->release();
    pool->release();
    if (kernel == NULL)
        throw OpenMMException("Error creating kernel "+name);
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
    return shared_ptr<ComputeQueueImpl>(new MetalQueue(device->newCommandQueue()));
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

string MetalContext::doubleToString(double value, bool mixedIsDouble) const {
    float hi = (float) value;
    if (!useMixedPrecision || !mixedIsDouble || hi == value || !isfinite(hi))
        return ComputeContext::doubleToString(value, mixedIsDouble);
    return "df64("+ComputeContext::doubleToString(hi)+", "+ComputeContext::doubleToString((float) (value-hi))+")";
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
    executeKernelFlat(kernel, arguments, gridSize*blockSize, blockSize, sharedSize);
}

void MetalContext::executeKernelFlat(metalFunction_t kernel, void** arguments, int threads, int blockSize, unsigned int sharedSize) {
    if (blockSize == -1)
        blockSize = ThreadBlockSize;
    int gridSize = (threads+blockSize-1)/blockSize;
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
