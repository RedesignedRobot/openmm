/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2026 Stanford University and the Authors.      *
 * Portions copyright (c) 2026 the Authors.                                   *
 * Authors: Peter Eastman                                                     *
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
#include "MetalContext.h"
#include "MetalArray.h"
#include "MetalEvent.h"
#include "MetalFFT3D.h"
#include "MetalIntegrationUtilities.h"
#include "MetalKernelSources.h"
#include "MetalNonbondedUtilities.h"
#include "MetalProgram.h"
#include "MetalQueue.h"
#include "MetalSort.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/MonteCarloFlexibleBarostat.h"
#include "openmm/System.h"
#include <Metal/Metal.hpp>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace OpenMM;
using namespace std;

const int MetalContext::ThreadBlockSize = 64;
const int MetalContext::TileSize = 32;

/**
 * The types and constructors that depend on the precision mode.  They come right after the Metal
 * standard library and before the prelude, which uses them.
 */
static const char* singlePrecisionDefinitions = R"(
typedef float real;
typedef float2 real2;
typedef float3 real3;
typedef float4 real4;
typedef float mixed;
typedef float2 mixed2;
typedef float3 mixed3;
typedef float4 mixed4;
#define make_real2 make_float2
#define make_real3 make_float3
#define make_real4 make_float4
#define make_mixed2 make_float2
#define make_mixed3 make_float3
#define make_mixed4 make_float4
)";

/**
 * Apple GPUs have no double precision, so in mixed precision mode mixed and double are df64, an
 * unevaluated sum of two floats, defined by the df64 source that precedes this block.  Device and
 * constant memory hold df64 values as IEEE doubles, so arrays and kernel arguments have the same
 * format as on devices with double precision.  metal_stdlib reserves double2 to double4 as type
 * names, so the double types are macros.
 */
static const char* mixedPrecisionDefinitions = R"(
typedef float real;
typedef float2 real2;
typedef float3 real3;
typedef float4 real4;
typedef df64 mixed;
typedef df64_2 mixed2;
typedef df64_3 mixed3;
typedef df64_4 mixed4;
#define make_real2 make_float2
#define make_real3 make_float3
#define make_real4 make_float4
#define make_mixed2 df64_2
#define make_mixed3 df64_3
#define make_mixed4 df64_4
#define double df64
#define double2 df64_2
#define double3 df64_3
#define double4 df64_4
#define make_double2 df64_2
#define make_double3 df64_3
#define make_double4 df64_4
)";

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

/**
 * Rewrite one kernel parameter if it is passed by value.  Metal only allows buffers,
 * textures, and constant references as kernel arguments, so "real4 box" becomes
 * "constant real4& _in_box", and a statement copying it into a local variable with the
 * original name is appended to the prologue.  Pointers and single tokens (macros that
 * expand to other parameters) are left alone.  Surrounding whitespace is preserved,
 * since a parameter may be followed by a preprocessor directive that needs its own line.
 */
static string rewriteParameter(const string& param, string& prologue) {
    size_t first = param.find_first_not_of(" \t\r\n");
    if (first == string::npos)
        return param;
    size_t last = param.find_last_not_of(" \t\r\n");
    string declaration = param.substr(first, last-first+1);
    if (declaration.find_first_of("*&[") != string::npos)
        return param;
    stringstream tokenStream(declaration);
    vector<string> tokens;
    string token;
    while (tokenStream >> token)
        if (token != "const")
            tokens.push_back(token);
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
 * Rewrite the signature of every KERNEL in a source file so that its parameters passed by
 * value become constant references (see rewriteParameter()).  Preprocessor directives inside
 * a parameter list are copied into the prologue as well, so each copy statement is compiled
 * exactly when the parameter it reads is.
 */
static string rewriteKernelSignatures(const string& source) {
    string result;
    size_t pos = 0;
    size_t kernelStart;
    while ((kernelStart = findWord(source, "KERNEL", pos)) != string::npos) {
        size_t open = source.find('(', kernelStart);
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

MetalContext::MetalContext(const System& system, const string& precision, const string& tempDir, MetalPlatform::PlatformData& platformData,
        MetalContext* originalContext) : ComputeContext(system), platformData(platformData), tempDir(tempDir+"/"),
        integration(NULL), expression(NULL), bonded(NULL), nonbonded(NULL) {
    if (precision != "single" && precision != "mixed")
        throw OpenMMException("Illegal value for Precision: "+precision);
    useDoublePrecision = false;
    useMixedPrecision = (precision == "mixed");
    contextIndex = platformData.contexts.size();
    if (!MetalPlatform::isPlatformSupported())
        throw OpenMMException("The Metal platform requires macOS 15 or later and an Apple GPU of family Apple7 (M1) or later");
    if (originalContext == NULL) {
        device = MTL::CreateSystemDefaultDevice();
        if (device == NULL)
            throw OpenMMException("No compatible Metal device is available");
        defaultQueue = shared_ptr<ComputeQueueImpl>(new MetalQueue(device));
    }
    else {
        device = originalContext->device->retain();
        defaultQueue = originalContext->defaultQueue;
    }
    currentQueue = defaultQueue;
    numCores = getGpuCoreCount();
    if (numCores == 0)
        numCores = 8; // The smallest Apple GPU with Metal 3.1 support has 7 or 8 cores.
    numAtoms = system.getNumParticles();
    paddedNumAtoms = TileSize*((numAtoms+TileSize-1)/TileSize);
    numAtomBlocks = (paddedNumAtoms+(TileSize-1))/TileSize;
    numThreadBlocks = 12*numCores;

    // Decide whether the fast versions of rsqrt(), reciprocal, exp(), and log() are sufficiently accurate
    // to use.  sqrt() stays correctly rounded.

    ComputeArray valuesArray;
    const int numValues = 20;
    valuesArray.initialize<float>(*this, 5*numValues, "values");
    vector<float> values(valuesArray.getSize(), 0.0f);
    float nextValue = 1e-4f;
    for (int i = 0; i < numValues; i++) {
        values[5*i] = nextValue;
        nextValue *= (float) M_PI;
    }
    valuesArray.upload(values);
    ComputeKernel accuracyKernel = compileProgram(MetalKernelSources::utilities)->createKernel("determineFastAccuracy");
    accuracyKernel->addArg(valuesArray);
    accuracyKernel->addArg(numValues);
    accuracyKernel->execute(numValues);
    valuesArray.download(values);
    double maxRsqrtError = 0.0, maxRecipError = 0.0, maxExpError = 0.0, maxLogError = 0.0;
    for (int i = 0; i < numValues; i++) {
        double v = values[5*i];
        maxRsqrtError = max(maxRsqrtError, fabs(1.0/sqrt(v)-values[5*i+1])*sqrt(v));
        maxRecipError = max(maxRecipError, fabs(1.0/v-values[5*i+2])/values[5*i+2]);
        maxExpError = max(maxExpError, fabs(exp(v)-values[5*i+3])/values[5*i+3]);
        maxLogError = max(maxLogError, fabs(log(v)-values[5*i+4])/values[5*i+4]);
    }
    compilationDefines["SQRT"] = "precise::sqrt";
    compilationDefines["RSQRT"] = (maxRsqrtError < 1e-6) ? "fast::rsqrt" : "precise::rsqrt";
    compilationDefines["RECIP(x)"] = (maxRecipError < 1e-6) ? "fast::divide(1.0f, (x))" : "(1.0f/(x))";
    compilationDefines["EXP"] = (maxExpError < 1e-6) ? "fast::exp" : "precise::exp";
    compilationDefines["LOG"] = (maxLogError < 1e-6) ? "fast::log" : "precise::log";
    compilationDefines["POW"] = "pow";
    compilationDefines["COS"] = "cos";
    compilationDefines["SIN"] = "sin";
    compilationDefines["TAN"] = "tan";
    compilationDefines["ACOS"] = "acos";
    compilationDefines["ASIN"] = "asin";
    compilationDefines["ATAN"] = "atan";
    compilationDefines["ERF"] = "erf";
    compilationDefines["ERFC"] = "erfc";
    compilationDefines["FMA"] = "fma";
    compilationDefines["FABS"] = "fabs";

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

    bonded = new BondedUtilities(*this);
    nonbonded = new MetalNonbondedUtilities(*this);
    integration = new MetalIntegrationUtilities(*this, system);
    expression = new ExpressionUtilities(*this);
    clearBuffer(posq);
}

MetalContext::~MetalContext() {
    for (auto force : forces)
        delete force;
    for (auto listener : reorderListeners)
        delete listener;
    for (auto computation : preComputations)
        delete computation;
    for (auto computation : postComputations)
        delete computation;
    if (integration != NULL)
        delete integration;
    if (expression != NULL)
        delete expression;
    if (bonded != NULL)
        delete bonded;
    if (nonbonded != NULL)
        delete nonbonded;
    device->release();
}

void MetalContext::initialize() {
    int numEnergyBuffers = max(numThreadBlocks*ThreadBlockSize, nonbonded->getNumEnergyBuffers());
    if (useMixedPrecision) {
        energyBuffer.initialize<double>(*this, numEnergyBuffers, "energyBuffer");
        energySum.initialize<double>(*this, numCores, "energySum");
    }
    else {
        energyBuffer.initialize<float>(*this, numEnergyBuffers, "energyBuffer");
        energySum.initialize<float>(*this, numCores, "energySum");
    }
    pinnedBuffer.resize(max(paddedNumAtoms*4, numEnergyBuffers));
    for (int i = 0; i < numAtoms; i++) {
        double mass = system.getParticleMass(i);
        if (useMixedPrecision)
            ((mm_double4*) getPinnedBuffer())[i] = mm_double4(0.0, 0.0, 0.0, mass == 0.0 ? 0.0 : 1.0/mass);
        else
            ((mm_float4*) getPinnedBuffer())[i] = mm_float4(0.0f, 0.0f, 0.0f, mass == 0.0 ? 0.0f : (float) (1.0/mass));
    }
    velm.upload(getPinnedBuffer());
    bonded->initialize(system);
    addAutoclearBuffer(longForceBuffer);
    addAutoclearBuffer(energyBuffer);
    int numEnergyParamDerivs = energyParamDerivNames.size();
    if (numEnergyParamDerivs > 0) {
        if (useMixedPrecision)
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

MTL::Library* MetalContext::createLibrary(const string& source, const map<string, string>& defines) {
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
    src << endl;
    src << "#include <metal_stdlib>\n";
    src << "using namespace metal;\n";
    if (useMixedPrecision) {
        src << "#define USE_MIXED_PRECISION 1\n";
        src << "#define SUPPORTS_DOUBLE_PRECISION\n";
        src << MetalKernelSources::df64 << endl;
        src << mixedPrecisionDefinitions << endl;
    }
    else
        src << singlePrecisionDefinitions << endl;
    src << "typedef unsigned int tileflags;\n";
    src << MetalKernelSources::common << endl;
    for (auto& pair : defines) {
        src << "#define " << pair.first;
        if (!pair.second.empty())
            src << " " << pair.second;
        src << endl;
    }
    if (!defines.empty())
        src << endl;
    src << rewriteKernelSignatures(source) << endl;
    const char* saveTemps = getenv("OPENMM_SAVE_TEMPS");
    if (saveTemps != NULL && string(saveTemps) == "1") {
        stringstream fileName;
        fileName << tempDir << "openmmTempKernel" << this << "_" << hash<string>()(src.str()) << ".metal";
        ofstream(fileName.str()) << src.str();
        cout << "Source code: " << fileName.str() << endl;
    }

    // Autoreleased objects must not outlive the pool, so no exception may be thrown until it is released.

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    options->setLanguageVersion(MTL::LanguageVersion3_1);
    options->setMathMode(MTL::MathModeSafe);
    options->setMathFloatingPointFunctions(MTL::MathFloatingPointFunctionsPrecise);
    NS::Error* error = NULL;
    MTL::Library* library = device->newLibrary(NS::String::string(src.str().c_str(), NS::UTF8StringEncoding), options, &error);
    options->release();
    string log;
    if (library == NULL && error != NULL)
        log = error->localizedDescription()->utf8String();
    pool->release();
    if (library == NULL && log.find("no 'buffer' resource location available") != string::npos)
        throw OpenMMException("Error compiling program: a Metal kernel can have at most 31 array and value arguments, which this system exceeds.  "+log);
    if (library == NULL)
        throw OpenMMException("Error compiling program: "+log);
    return library;
}

MTL::ComputePipelineState* MetalContext::createPipeline(MTL::Library* library, const string& name) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Function* function = library->newFunction(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pipeline = NULL;
    string log = "no kernel with this name";
    if (function != NULL) {
        NS::Error* error = NULL;
        pipeline = device->newComputePipelineState(function, &error);
        function->release();
        if (pipeline == NULL && error != NULL)
            log = error->localizedDescription()->utf8String();
    }
    pool->release();
    if (pipeline == NULL)
        throw OpenMMException("Error creating kernel "+name+": "+log);
    if (pipeline->threadExecutionWidth() != getSIMDWidth()) {
        pipeline->release();
        throw OpenMMException("Error creating kernel "+name+": the kernels require a SIMD width of "+intToString(getSIMDWidth()));
    }
    return pipeline;
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

MetalQueue& MetalContext::getCurrentMetalQueue() {
    return dynamic_cast<MetalQueue&>(*currentQueue);
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
    return shared_ptr<ComputeProgramImpl>(new MetalProgram(*this, createLibrary(source, defines)));
}

MetalArray& MetalContext::unwrap(ArrayInterface& array) const {
    MetalArray* metalArray;
    ComputeArray* wrapper = dynamic_cast<ComputeArray*>(&array);
    if (wrapper != NULL)
        metalArray = dynamic_cast<MetalArray*>(&wrapper->getArray());
    else
        metalArray = dynamic_cast<MetalArray*>(&array);
    if (metalArray == NULL)
        throw OpenMMException("Array argument is not a MetalArray");
    return *metalArray;
}

int MetalContext::computeThreadBlockSize(double memory) const {
    const int maxThreadgroupMemory = 32768;
    int max = (int) (maxThreadgroupMemory/memory);
    if (max < ThreadBlockSize)
        throw OpenMMException("Too much threadgroup memory requested!");
    int threads = getSIMDWidth();
    while (threads+getSIMDWidth() < max)
        threads += getSIMDWidth();
    return threads;
}

double MetalContext::reduceEnergy() {
    int workGroupSize = getMaxThreadBlockSize();
    reduceEnergyKernel->setArg(0, energyBuffer);
    reduceEnergyKernel->setArg(1, energySum);
    reduceEnergyKernel->setArg(2, (int) energyBuffer.getSize());
    reduceEnergyKernel->setArg(3, workGroupSize);
    reduceEnergyKernel->execute(workGroupSize*energySum.getSize(), workGroupSize);
    energySum.download(getPinnedBuffer());
    double result = 0;
    if (useMixedPrecision) {
        for (int i = 0; i < energySum.getSize(); i++)
            result += ((double*) getPinnedBuffer())[i];
    }
    else {
        for (int i = 0; i < energySum.getSize(); i++)
            result += ((float*) getPinnedBuffer())[i];
    }
    return result;
}

void MetalContext::flushQueue() {
    getCurrentMetalQueue().commit();
}
