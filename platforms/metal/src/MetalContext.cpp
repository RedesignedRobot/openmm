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
#include <cstring>
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
const int MetalContext::SpreadBlockSize = 128;

/**
 * Rewrite one kernel parameter.  A value parameter becomes a constant reference, or, when members is
 * given, a member of the kernel's argument struct, and then this returns only its trailing macros.
 * Each other parameter that takes a buffer binding gets a one byte marker member, so the host can
 * recover the position of every argument from reflection.
 */
static string rewriteParameter(const string& param, string& prologue, string* members, int& index) {
    smatch m;
    bool isValue = !regex_search(param, regex("\\b(GLOBAL|LOCAL|LOCAL_ARG|device|threadgroup|constant)\\b|[&\\[*]"));
    if (!isValue || !regex_match(param, m, regex("\\s*(const\\s+)?(\\S.*?)\\s+(\\w+)((\\s+[A-Z_]+)*)\\s*"))) {
        if (members != NULL && !isValue && !regex_search(param, regex("\\b(LOCAL|LOCAL_ARG|threadgroup)\\b|\\[\\[")))
            *members += "char _mmB"+to_string(index++)+";\n";
        if (isValue || regex_search(param, regex("\\b(GLOBAL|LOCAL|LOCAL_ARG|device|threadgroup|constant)\\b|[&\\[]")))
            return param;
        return "device "+param;
    }
    if (members == NULL) {
        prologue += m[2].str()+" "+m[3].str()+" = _in_"+m[3].str()+";\n";
        return "constant "+m[2].str()+"& _in_"+m[3].str()+m[4].str();
    }
    string member = "_mmV"+to_string(index);
    *members += m[2].str()+" "+member+";\nchar _mmE"+to_string(index++)+";\n";
    prologue += m[2].str()+" "+m[3].str()+" = _in__mmArgs."+member+";\n";
    return m[4].str();
}

static size_t lineEnd(const string& source, size_t i) {
    size_t end = source.find('\n', i);
    return (end == string::npos ? source.size() : end+1);
}

/**
 * Metal needs an address space on every kernel pointer, and passes values by constant reference.
 * With packValues, value parameters instead go into one struct per kernel, because Metal allows
 * only 31 buffer bindings per kernel.  A signature that never closes is left for the compiler to
 * report.
 */
static string rewriteKernelSignatures(const string& source, bool packValues) {
    string result;
    size_t pos = 0;
    smatch m;
    while (regex_search(source.begin()+pos, source.end(), m, regex("\\b(KERNEL|__global__)\\b.*?\\bvoid\\s+(\\w+)\\s*\\("))) {
        size_t i = pos+m.position()+m.length();
        string argsType = "_mmArgs_"+m[2].str();
        string signature = m.str()+(packValues ? "constant "+argsType+"& _in__mmArgs, " : "");
        result += source.substr(pos, m.position());
        string prologue, members, param;
        int index = 0;
        int depth = 0;
        for (; depth >= 0 && i < source.size(); i++) {
            // A value moved into the struct leaves the signature, along with one comma next to it.
            size_t prologueSize = prologue.size();
            if (source[i] == '#') {
                size_t end = lineEnd(source, i);
                string rewritten = rewriteParameter(param, prologue, packValues ? &members : NULL, index);
                if (packValues && prologue.size() > prologueSize)
                    signature.erase(signature.rfind(','), 1);
                signature += rewritten+"\n"+source.substr(i, end-i);
                prologue += source.substr(i, end-i);
                members += source.substr(i, end-i);
                param.clear();
                i = end-1;
                continue;
            }
            depth += (source[i] == '(')-(source[i] == ')');
            if (depth < 0 || (source[i] == ',' && depth == 0)) {
                string rewritten = rewriteParameter(param, prologue, packValues ? &members : NULL, index);
                bool moved = packValues && prologue.size() > prologueSize;
                if (moved && source[i] == ')')
                    signature.erase(signature.rfind(','), 1);
                signature += rewritten;
                if (!moved || source[i] == ')')
                    signature += source[i];
                param.clear();
            }
            else
                param += source[i];
        }
        size_t body = source.find('{', i);
        if (depth >= 0 || body == string::npos)
            return source;
        if (packValues)
            result += "struct "+argsType+" {\n"+members+"};\n";
        pos = body+1;
        result += signature+source.substr(i, pos-i)+"\n"+prologue;
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
 * functions the thread address space.  A signature that never closes is left for the compiler
 * to report.
 */
static string rewriteDeviceSignatures(const string& source) {
    string result;
    size_t pos = 0;
    smatch m;
    while (regex_search(source.begin()+pos, source.end(), m, regex("\\b(DEVICE|__device__)\\b[^;{}()=]*?\\b\\w+\\s*\\("))) {
        size_t i = pos+m.position()+m.length();
        result += source.substr(pos, i-pos);
        string param;
        int depth = 0;
        for (; depth >= 0 && i < source.size(); i++) {
            if (source[i] == '#') {
                size_t end = lineEnd(source, i);
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
        if (depth >= 0)
            return source;
        pos = i;
    }
    return result+source.substr(pos);
}

MetalContext::MetalContext(const System& system, int deviceIndex, bool useBlockingSync, const string& precision, const string& tempDir, MetalPlatform::PlatformData& platformData,
        MetalContext* originalContext) : ComputeContext(system), platformData(platformData), contextIsValid(false),
        pinnedBuffer(NULL), integration(NULL), expression(NULL), bonded(NULL), nonbonded(NULL),
        useBlockingSync(useBlockingSync), supportsHardwareFloatGlobalAtomicAdd(false), autoclearWorkCount(-1),
        autoclearBufferCount(0), forceMemoEnabled(false), skippingForcePass(false), forceMemoWorkCount(-1), forcePassCount(0),
        forcePassesSkipped(0), forceMemoGroups(0) {
    const char* forceMemoEnv = getenv("OPENMM_METAL_FORCE_MEMO");
    forceMemoAllowed = (forceMemoEnv == NULL || string(forceMemoEnv) != "0");
    reportForceMemo = (forceMemoEnv != NULL && string(forceMemoEnv) == "stats");
    if (precision == "single") {
        useDoublePrecision = false;
        useMixedPrecision = false;
    }
    else if (precision == "mixed") {
        useDoublePrecision = false;
        useMixedPrecision = true;
    }
    else if (precision == "double")
        throw OpenMMException("The Metal platform does not support Precision=double, because Apple GPUs have no 64-bit floating point.  Use single or mixed.");
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

    // Apple7 and Apple8 GPUs add floats atomically too, but on an M2 the fixed-point PME spread
    // measured slightly faster.  From Apple9 on, float atomics spread charge about 3x faster.
    supportsHardwareFloatGlobalAtomicAdd = device->supportsFamily(MTL::GPUFamilyApple9);

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

    // PME spreading with float atomics sums spans of up to 16 consecutive atoms in a threadgroup slab
    // per SIMD group and issues one device atomic per grid point a span touches (pme.cc).  1024
    // points hold 50% of the 16-atom spans on apoa1pme and 95% of the 8-atom halves.
    compilationDefines["SPREAD_SPAN"] = "16";
    compilationDefines["SPREAD_SLAB"] = "1024";
    compilationDefines["SPREAD_GROUPS"] = intToString(SpreadBlockSize/32);
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
    for (auto& kernel : kernelArguments)
        kernel.first->release();
    popAsCurrent();
    contextIsValid = false;
    device->release();
    if (reportForceMemo)
        fprintf(stderr, "OpenMM Metal: force memo skipped %lld of %lld force passes\n", forcePassesSkipped, forcePassCount);
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
    unwrap(velm).setPairStorage(useMixedPrecision);
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
    if (useMixedPrecision) {
        // The program's own defines come after this header, too late for df64 to see.
        if (defines.find("DF64_FINITE") != defines.end())
            src << "#define DF64_FINITE\n";
        src << MetalKernelSources::df64 << endl;
    }
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
        // mixed4s is the integration state (velm, posDelta) as stored: float pairs in mixed, with no IEEE decode.
        src << (useMixedPrecision ? "typedef df64s_4 mixed4s;\n" : "typedef double4 mixed4s;\n");
    }
    else {
        src << "typedef float mixed;\n";
        src << "typedef float2 mixed2;\n";
        src << "typedef float3 mixed3;\n";
        src << "typedef float4 mixed4;\n";
        src << "typedef float4 mixed4s;\n";
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
    string header = src.str();
    // Common kernels use "thread" as a variable name, a reserved word in MSL, so the program source and its defines
    // rename it.  That also renames the address space qualifier: program sources must write PRIVATE, never thread.
    string body = rewriteDeviceSignatures(regex_replace(regex_replace(source, regex("\\bthread\\b"), "_mmThread"), regex("\\blong long\\b"), "long"));
    string log;
    metalModule_t module = createLibrary(header+rewriteKernelSignatures(body, false)+"\n", log, saveTemps);
    if (module == NULL) {
        // A kernel past Metal's 31 buffer bindings compiles once its values are packed.  Any other
        // error fails both ways, and the first log reports it.
        string packedLog;
        module = createLibrary(header+rewriteKernelSignatures(body, true)+"\n", packedLog, saveTemps);
        if (module != NULL)
            packedModules.insert(module);
    }
    if (module == NULL && log.find("no 'buffer' resource location available") != string::npos)
        throw OpenMMException("Error compiling program: a Metal kernel can have at most 31 array and value arguments, which this system exceeds.  "+log);
    if (module == NULL)
        throw OpenMMException("Error compiling program: "+log);
    loadedModules.push_back(module);
    return module;
}

metalModule_t MetalContext::createLibrary(const string& source, string& log, bool saveTemps) {
    // Select names for the various temporary files.

    if (saveTemps) {
        stringstream tempFileName;
        const char* saveTempsPrefixEnv = getenv("OPENMM_SAVE_TEMPS_PREFIX");
        if (saveTempsPrefixEnv) {
            tempFileName << saveTempsPrefixEnv;
        }
        tempFileName << getHash(source);

        string inputFile = (tempFileName.str()+".metal");
        std::cout << "Source code: " << inputFile << std::endl;
        ofstream out(inputFile.c_str());
        out << source;
        out.close();
    }

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    options->setLanguageVersion(MTL::LanguageVersion3_2);
    options->setMathMode(MTL::MathModeSafe);
    options->setMathFloatingPointFunctions(MTL::MathFloatingPointFunctionsPrecise);
    NS::Error* error = NULL;
    metalModule_t module = device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), options, &error);
    options->release();
    if (module == NULL)
        log = error->localizedDescription()->utf8String();
    pool->release();
    return module;
}

metalFunction_t MetalContext::getKernel(metalModule_t& module, const string& name) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Function* function = module->newFunction(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
    MTL::ComputePipelineReflection* reflection = NULL;
    bool packed = (packedModules.find(module) != packedModules.end());
    bool typeInfo = (packed || useMixedPrecision);
    MTL::PipelineOption reflectionOptions = (typeInfo ? MTL::PipelineOptionBindingInfo | MTL::PipelineOptionBufferTypeInfo : MTL::PipelineOptionBindingInfo);
    metalFunction_t kernel = (function == NULL ? NULL : device->newComputePipelineState(function, reflectionOptions, &reflection, NULL));
    KernelArguments args;
    MTL::BufferBinding* argsBinding = NULL;
    vector<BufferLayout> bindingLayouts;
    for (int i = 0; kernel != NULL && i < reflection->bindings()->count(); i++) {
        MTL::Binding* binding = reflection->bindings()->object<MTL::Binding>(i);
        if (binding->type() != MTL::BindingTypeBuffer)
            continue;
        string bindingName = binding->name()->utf8String();
        args.sizes.resize(max((int) args.sizes.size(), (int) binding->index()+1), 0);
        bindingLayouts.resize(args.sizes.size(), PlainLayout);
        if (bindingName == "_in__mmArgs")
            argsBinding = (MTL::BufferBinding*) binding;
        else if (bindingName.rfind("_in_", 0) == 0)
            args.sizes[binding->index()] = ((MTL::BufferBinding*) binding)->bufferDataSize();
        else if (useMixedPrecision)
            bindingLayouts[binding->index()] = getBufferLayout((MTL::BufferBinding*) binding);
    }
    bool validArguments = (!packed || (argsBinding != NULL && getPackedArguments(argsBinding, args.sizes.size()-1, args)));
    if (validArguments) {
        // Unpacked arguments bind at their own index, packed buffers in order from index 1.
        args.layouts.resize(args.sizes.size(), PlainLayout);
        int bufferIndex = 1;
        for (int i = 0; i < args.sizes.size(); i++) {
            int bindingIndex = (packed ? bufferIndex : i);
            if (args.sizes[i] != 0)
                continue;
            if (bindingIndex < bindingLayouts.size())
                args.layouts[i] = bindingLayouts[bindingIndex];
            bufferIndex++;
        }
    }
    if (function != NULL)
        function->release();
    pool->release();
    if (kernel == NULL)
        throw OpenMMException("Error creating kernel "+name);
    if (!validArguments)
        throw OpenMMException("Error creating kernel "+name+": its packed arguments don't match its buffer bindings");
    kernelArguments[kernel] = args;
    return kernel;
}

MetalContext::BufferLayout MetalContext::getBufferLayout(MTL::BufferBinding* binding) {
    MTL::DataType type = binding->bufferDataType();
    if (type == MTL::DataTypeInt || type == MTL::DataTypeUInt)
        return AnyLayout;
    if (type != MTL::DataTypeStruct)
        return PlainLayout;
    // A df64s_4 element is the only struct whose member x is a struct with a member shi.
    MTL::StructMember* x = binding->bufferStructType()->memberByName(NS::String::string("x", NS::UTF8StringEncoding));
    if (x == NULL || x->dataType() != MTL::DataTypeStruct)
        return PlainLayout;
    return (x->structType()->memberByName(NS::String::string("shi", NS::UTF8StringEncoding)) != NULL ? PairLayout : PlainLayout);
}

bool MetalContext::getPackedArguments(MTL::BufferBinding* binding, int numBuffers, KernelArguments& args) {
    // The struct has a marker member for each declared argument, numbered in source order: _mmB<n> for
    // a buffer, and _mmV<n> followed by _mmE<n> for a value.  Arguments past the last marker came from
    // macros, and are buffers.
    MTL::StructType* type = binding->bufferStructType();
    if (type == NULL)
        return false;
    map<int, MTL::StructMember*> markers;
    for (int i = 0; i < type->members()->count(); i++) {
        MTL::StructMember* member = type->members()->object<MTL::StructMember>(i);
        string memberName = member->name()->utf8String();
        if (memberName.rfind("_mmB", 0) == 0 || memberName.rfind("_mmV", 0) == 0)
            markers[stoi(memberName.substr(4))] = member;
    }
    args = KernelArguments();
    for (auto& marker : markers) {
        string memberName = marker.second->name()->utf8String();
        int offset = marker.second->offset();
        int size = 0;
        if (memberName[3] == 'V') {
            string endName = "_mmE"+to_string(marker.first);
            MTL::StructMember* end = type->memberByName(NS::String::string(endName.c_str(), NS::UTF8StringEncoding));
            if (end == NULL)
                return false;
            size = end->offset()-offset;
        }
        else
            numBuffers--;
        args.sizes.push_back(size);
        args.offsets.push_back(offset);
    }
    if (numBuffers < 0)
        return false;
    args.sizes.resize(args.sizes.size()+numBuffers, 0);
    args.offsets.resize(args.sizes.size(), 0);
    args.packedSize = binding->bufferDataSize();
    return true;
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
    // MetalFFT3D's PME convolution kernels take 30% longer at 55^3 than at 56^3 because of the radix 11 stages, so a
    // size those kernels would run on a cubic grid avoids the factors 11 and 13.
    int size = MetalFFT3D::findLegalDimension(minimum);
    int smooth = MetalFFT3D::findLegalDimension(minimum, 7);
    if (smooth != size && MetalFFT3D::canUsePmeConvolution(*this, smooth, smooth, smooth))
        return smooth;
    return size;
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

void MetalContext::executeKernelFlat(metalFunction_t kernel, void** arguments, int threads, int blockSize, unsigned int sharedSize, MTL::Buffer* indirect, size_t indirectOffset) {
    if (blockSize == -1)
        blockSize = ThreadBlockSize;
    int gridSize = (threads+blockSize-1)/blockSize;
    MetalQueue& queue = getCurrentStream();
    lock_guard<recursive_mutex> guard(queue.getLock());
    MTL::ComputeCommandEncoder* encoder = queue.getEncoder();
    encoder->setComputePipelineState(kernel);
    const KernelArguments& args = kernelArguments[kernel];
    if (args.offsets.empty()) {
        for (int i = 0; i < args.sizes.size(); i++) {
            if (args.sizes[i] == 0)
                encoder->setBuffer(*(MTL::Buffer**) arguments[i], 0, i);
            else
                encoder->setBytes(arguments[i], args.sizes[i], i);
        }
    }
    else {
        vector<char> packed(args.packedSize);
        int bufferIndex = 1;
        for (int i = 0; i < args.sizes.size(); i++) {
            if (args.sizes[i] == 0)
                encoder->setBuffer(*(MTL::Buffer**) arguments[i], 0, bufferIndex++);
            else
                memcpy(&packed[args.offsets[i]], arguments[i], args.sizes[i]);
        }
        encoder->setBytes(packed.data(), packed.size(), 0);
    }
    encoder->setThreadgroupMemoryLength((sharedSize+15)/16*16, 0);
    if (indirect != NULL)
        encoder->dispatchThreadgroups(indirect, indirectOffset, MTL::Size(blockSize, 1, 1));
    else
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

bool MetalContext::getSupportsForcePassKernels() const {
    return device->supportsFamily(MTL::GPUFamilyApple9);
}

void MetalContext::executeForcePassKernels() {
    for (auto& kernel : forcePassKernels)
        kernel.first->execute(kernel.second);
    forcePassKernels.clear();
}

void MetalContext::markAutoclearBuffersCleared() {
    autoclearWorkCount = MetalQueue::getWorkCount();
    autoclearBufferCount = autoclearBuffers.size();
}

bool MetalContext::getAutoclearBuffersMarkedCleared() const {
    return autoclearWorkCount == MetalQueue::getWorkCount() && autoclearBufferCount == autoclearBuffers.size();
}

void MetalContext::clearAutoclearBuffersIfNeeded() {
    bool cleared = getAutoclearBuffersMarkedCleared();
    autoclearWorkCount = -1;
    if (!cleared)
        clearAutoclearBuffers();
}

void MetalContext::setForceMemoEnabled(bool enabled) {
    forceMemoEnabled = (enabled && forceMemoAllowed);
}

void MetalContext::armForceMemo(int groups) {
    forceMemoWorkCount = MetalQueue::getWorkCount();
    forceMemoGroups = groups;
    getPeriodicBoxVectors(forceMemoBox[0], forceMemoBox[1], forceMemoBox[2]);
}

bool MetalContext::getForceMemoCurrent() const {
    return forceMemoWorkCount == MetalQueue::getWorkCount();
}

bool MetalContext::getForceMemoHolds(int groups) const {
    if (!getForceMemoCurrent() || groups != forceMemoGroups)
        return false;
    Vec3 box[3];
    getPeriodicBoxVectors(box[0], box[1], box[2]);
    return box[0] == forceMemoBox[0] && box[1] == forceMemoBox[1] && box[2] == forceMemoBox[2];
}

void MetalContext::rearmForceMemo() {
    armForceMemo(forceMemoGroups);
}

void MetalContext::disarmForceMemo() {
    forceMemoWorkCount = -1;
}

void MetalContext::setSkippingForcePass(bool skipping) {
    skippingForcePass = skipping;
}

void MetalContext::noteForcePass(bool skipped) {
    forcePassCount++;
    if (skipped)
        forcePassesSkipped++;
}
