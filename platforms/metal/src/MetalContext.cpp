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
#include <metal/metalrtc.h>


#define CHECK_RESULT(result) CHECK_RESULT2(result, errorMessage);
#define CHECK_RESULT2(result, prefix) \
    if (result != metalSuccess) { \
        std::stringstream m; \
        m<<prefix<<": "<<getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }

#define METALRTC_CHECK_RESULT(result, prefix) \
    if (result != METALRTC_SUCCESS) { \
        stringstream m; \
        m<<prefix<<": "<<metalrtcGetErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }

using namespace OpenMM;
using namespace std;

const int MetalContext::ThreadBlockSize = 64;
const int MetalContext::TileSize = 32;
bool MetalContext::hasInitializedMetal = false;


MetalContext::MetalContext(const System& system, int deviceIndex, bool useBlockingSync, const string& precision, const string& tempDir, MetalPlatform::PlatformData& platformData,
        MetalContext* originalContext) : ComputeContext(system), platformData(platformData), contextIsValid(false),
        pinnedBuffer(NULL), integration(NULL), expression(NULL), bonded(NULL), nonbonded(NULL),
        useBlockingSync(useBlockingSync), supportsHardwareFloatGlobalAtomicAdd(false) {
    if (!hasInitializedMetal) {
        CHECK_RESULT2(metalInit(0), "Error initializing METAL");
        hasInitializedMetal = true;
    }
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
    char* cacheVariable = getenv("OPENMM_CACHE_DIR");
    cacheDir = (cacheVariable == NULL ? tempDir : string(cacheVariable));
#ifdef WIN32
    this->tempDir = tempDir+"\\";
    cacheDir = cacheDir+"\\";
#else
    this->tempDir = tempDir+"/";
    cacheDir = cacheDir+"/";
#endif
    contextIndex = platformData.contexts.size();
    string errorMessage = "Error initializing Context";
    if (originalContext == NULL) {
        isLinkedContext = false;
        int numDevices;
        CHECK_RESULT(metalGetDeviceCount(&numDevices));
        if (deviceIndex < -1 || deviceIndex >= numDevices)
            throw OpenMMException("Illegal value for DeviceIndex: "+intToString(deviceIndex));

        vector<int> devicePrecedence;
        if (deviceIndex == -1) {
            devicePrecedence = getDevicePrecedence();
        } else {
            devicePrecedence.push_back(deviceIndex);
        }

        this->deviceIndex = -1;
        for (int i = 0; i < static_cast<int>(devicePrecedence.size()); i++) {
            int trialDeviceIndex = devicePrecedence[i];
            CHECK_RESULT(metalDeviceGet(&device, trialDeviceIndex));
            // try setting device
            if (metalSetDevice(device) == metalSuccess) {
                this->deviceIndex = trialDeviceIndex;
                break;
            }

        }
        if (this->deviceIndex == -1) {
            if (deviceIndex != -1)
                throw OpenMMException("The requested METAL device could not be loaded");
            else
                throw OpenMMException("No compatible METAL device is available");
        }
        defaultQueue = shared_ptr<ComputeQueueImpl>(new MetalQueue());
    }
    else {
        isLinkedContext = true;
        this->deviceIndex = originalContext->deviceIndex;
        this->device = originalContext->device;
        defaultQueue = originalContext->defaultQueue;
    }
    currentQueue = defaultQueue;

    metalDeviceProp_t props;
    CHECK_RESULT(metalGetDeviceProperties(&props, device));

    // set device properties
    this->simdWidth = props.warpSize;
    this->sharedMemPerBlock = props.sharedMemPerBlock;

    gpuArchitecture = props.gcnArchName;
    // METAL-TODO: find a good value here
    int numThreadBlocksPerComputeUnit = 6;

    // GPUs starting from CDNA1 and RDNA3 support atomic add for floats (global_atomic_add_f32),
    // which can be used in PME. Older GPUs use fixed point charge spreading instead.
    // RDNA4 also has this instruction but benchmarks show that it is very slow compared to
    // global_atomic_add_u64.
    this->supportsHardwareFloatGlobalAtomicAdd = false;
    if (gpuArchitecture.find("gfx908") == 0 ||
        gpuArchitecture.find("gfx90a") == 0 ||
        gpuArchitecture.find("gfx94") == 0 ||
        gpuArchitecture.find("gfx11") == 0) {
        this->supportsHardwareFloatGlobalAtomicAdd = true;
    }

    hostMallocFlags = metalHostMallocDefault;
#if !defined(WIN32)
    // metalHostMallocNumaUser may not be allowed in some conditions, for example, if docker container
    // is created without --security-opt seccomp=unconfined or --cap-add=SYS_NICE
    int* tmpHostBuffer;
    if(metalHostMalloc(&tmpHostBuffer, sizeof(*tmpHostBuffer), metalHostMallocNumaUser) == metalSuccess) {
        CHECK_RESULT(metalHostFree(tmpHostBuffer));
        hostMallocFlags = metalHostMallocNumaUser;
    }
#endif

    contextIsValid = true;
    ContextSelector selector(*this);
    if (contextIndex > 0 && originalContext == NULL) {
        int canAccess;
        CHECK_RESULT(metalDeviceCanAccessPeer(&canAccess, getDevice(), platformData.contexts[0]->getDevice()));
        if (canAccess) {
            {
                ContextSelector selector2(*platformData.contexts[0]);
                metalError_t result = metalDeviceEnablePeerAccess(getDevice(), 0);
                if (result != metalErrorPeerAccessAlreadyEnabled) {
                    CHECK_RESULT(result);
                }
            }
            metalError_t result = metalDeviceEnablePeerAccess(platformData.contexts[0]->getDevice(), 0);
            if (result != metalErrorPeerAccessAlreadyEnabled) {
                CHECK_RESULT(result);
            }
        }
    }
    numAtoms = system.getNumParticles();
    paddedNumAtoms = TileSize*((numAtoms+TileSize-1)/TileSize);
    numAtomBlocks = (paddedNumAtoms+(TileSize-1))/TileSize;
    CHECK_RESULT(metalDeviceGetAttribute(&multiprocessors, metalDeviceAttributeMultiprocessorCount, device));
    // For RDNA GPUs metalDeviceAttributeMultiprocessorCount means WGP (work-group processors, two compute units), not CUs.
    if (simdWidth == 32)
        multiprocessors *= 2;
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
    compilationDefines["RECIP(x)"] = useDoublePrecision ? "(1.0/(x))" : "(1.0f/(x))";
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
        metalHostFree(pinnedBuffer);
    if (integration != NULL)
        delete integration;
    if (expression != NULL)
        delete expression;
    if (bonded != NULL)
        delete bonded;
    if (nonbonded != NULL)
        delete nonbonded;
    for (auto module : loadedModules)
        metalModuleUnload(module);
    popAsCurrent();
    contextIsValid = false;
}

void MetalContext::initialize() {
    ContextSelector selector(*this);
    string errorMessage = "Error initializing Context";
    int numEnergyBuffers = max(numThreadBlocks*ThreadBlockSize, nonbonded->getNumEnergyBuffers());
    if (useDoublePrecision) {
        energyBuffer.initialize<double>(*this, numEnergyBuffers, "energyBuffer");
        energySum.initialize<double>(*this, multiprocessors, "energySum");
        int pinnedBufferSize = max(paddedNumAtoms*4, numEnergyBuffers);
        CHECK_RESULT(metalHostMalloc(&pinnedBuffer, pinnedBufferSize*sizeof(double), getHostMallocFlags()));
    }
    else if (useMixedPrecision) {
        energyBuffer.initialize<double>(*this, numEnergyBuffers, "energyBuffer");
        energySum.initialize<double>(*this, multiprocessors, "energySum");
        int pinnedBufferSize = max(paddedNumAtoms*4, numEnergyBuffers);
        CHECK_RESULT(metalHostMalloc(&pinnedBuffer, pinnedBufferSize*sizeof(double), getHostMallocFlags()));
    }
    else {
        energyBuffer.initialize<float>(*this, numEnergyBuffers, "energyBuffer");
        energySum.initialize<float>(*this, multiprocessors, "energySum");
        int pinnedBufferSize = max(paddedNumAtoms*6, numEnergyBuffers);
        CHECK_RESULT(metalHostMalloc(&pinnedBuffer, pinnedBufferSize*sizeof(float), getHostMallocFlags()));
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
    if (contextIsValid)
        metalSetDevice(device);
}

thread_local std::stack<metalDevice_t> outerScopeDevices;

void MetalContext::pushAsCurrent() {
    if (contextIsValid) {
        // Emulate cuCtxPushCurrent's behavior because metalCtxPushCurrent is deprecated
        metalDevice_t outerScopeDevice;
        metalGetDevice(&outerScopeDevice);
        outerScopeDevices.push(outerScopeDevice);
        if (device != outerScopeDevice) {
            metalSetDevice(device);
        }
    }
}

void MetalContext::popAsCurrent() {
    if (contextIsValid) {
        // Emulate cuCtxPopCurrent's behavior because metalCtxPopCurrent is deprecated
        metalDevice_t outerScopeDevice = outerScopeDevices.top();
        outerScopeDevices.pop();
        if (outerScopeDevice != device) {
            metalSetDevice(outerScopeDevice);
        }
    }
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

string MetalContext::getCacheFileName(const string& src) const {
    stringstream cacheFile;
    cacheFile << cacheDir << "openmm-metal-" << getHash(src + gpuArchitecture);
    return cacheFile.str();
}

metalModule_t MetalContext::createModule(const string source) {
    return createModule(source, map<string, string>());
}

metalModule_t MetalContext::createModule(const string source, const map<string, string>& defines) {
    const char* saveTempsEnv = getenv("OPENMM_SAVE_TEMPS");
    const bool saveTemps = saveTempsEnv != nullptr && string(saveTempsEnv) == "1";

    int runtimeVersion;
    CHECK_RESULT2(metalRuntimeGetVersion(&runtimeVersion), "Error getting METAL runtime version");

    string options = "-O3 -ffast-math -munsafe-fp-atomics -Wall -Wno-metal-only";
    options += " --offload-arch=" + gpuArchitecture;
    if (gpuArchitecture.find("gfx90a") == 0 ||
        gpuArchitecture.find("gfx94") == 0) {
        // METAL-TODO: Remove it when the compiler does a better job
        // Disable SLP vectorization as it may generate unoptimal packed math instructions on
        // >=MI200 (gfx90a, gfx942): more v_mov, higher register usage etc.
        options += " -fno-slp-vectorize";
    }
    if (getMaxThreadBlockSize() < 1024) {
        options += " --gpu-max-threads-per-block=" + std::to_string(getMaxThreadBlockSize());
    }
    if (runtimeVersion < 60140092) {
        // Workaround for operator* defined for complex types (typedefs for float2, double2) in
        // ROCm 6.0 headers. This issue has been fixed in 6.1. metalRTC includes amd_hip_complex.h
        // by default, we fool the include guard into thinking the header is already included.
        options += " -D METAL_INCLUDE_METAL_AMD_DETAIL_METAL_COMPLEX_H";
    }
    stringstream src;
    src << "// Compilation Options: " << options << endl << endl;
    src << "// METAL Runtime Version: " << runtimeVersion << endl << endl;
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
    for (auto& pair : defines) {
        src << "#define " << pair.first;
        if (!pair.second.empty())
            src << " " << pair.second;
        src << endl;
    }
    if (!defines.empty())
        src << endl;
    src << MetalKernelSources::intrinsics << endl;
    src << source << endl;

    // See whether we already have PTX for this kernel cached.

    string cacheFile = getCacheFileName(src.str());
    metalModule_t module;
    if (metalModuleLoad(&module, cacheFile.c_str()) == metalSuccess) {
        loadedModules.push_back(module);
        return module;
    }
    else {
        // Need to call metalGetLastError to clear the error code
        metalError_t last_error = metalGetLastError();
        if (last_error != metalErrorFileNotFound) {
            std::ostringstream msg;
            msg << "Detected METAL error: " << metalGetErrorString(last_error)
                << "(" << last_error << ") at " << __FILE__ << ":" << __LINE__;
            throw OpenMMException(msg.str());
        }
    }

    // Select names for the various temporary files.

    if (saveTemps) {
        stringstream tempFileName;
        const char* saveTempsPrefixEnv = getenv("OPENMM_SAVE_TEMPS_PREFIX");
        if (saveTempsPrefixEnv) {
            tempFileName << saveTempsPrefixEnv;
        }
        tempFileName << getHash(src.str());

        options += " --save-temps";

        string inputFile = (tempFileName.str()+".metal");
        std::cout << "Source code: " << inputFile << std::endl;
        std::cout << "Compile options: " << options << std::endl;
        ofstream out(inputFile.c_str());
        out << src.str();
        out.close();
    }

    // Split the command line options into an array of options.

    stringstream flagsStream(options);
    string flag;
    vector<string> splitFlags;
    while (flagsStream >> flag)
        splitFlags.push_back(flag);
    int numOptions = splitFlags.size();
    vector<const char*> optionsVec(numOptions);
    for (int i = 0; i < numOptions; i++)
        optionsVec[i] = &splitFlags[i][0];

    // Compile the program to CO.

    metalrtcProgram program;
    METALRTC_CHECK_RESULT(metalrtcCreateProgram(&program, src.str().c_str(), NULL, 0, NULL, NULL), "Error creating program");
    try {
        metalrtcResult result = metalrtcCompileProgram(program, optionsVec.size(), &optionsVec[0]);
        if (result != METALRTC_SUCCESS || saveTemps) {
            size_t logSize;
            metalrtcGetProgramLogSize(program, &logSize);
            std::string log(logSize, '\0');
            if (logSize > 0) {
                metalrtcGetProgramLog(program, &log[0]);
                if (saveTemps) {
                    std::cout << "Log: " << log << std::endl;
                }
            }
            if (result != METALRTC_SUCCESS) {
                throw OpenMMException("Error compiling program: "+log);
            }
        }
        size_t codeSize;
        metalrtcGetCodeSize(program, &codeSize);
        vector<char> code(codeSize);
        metalrtcGetCode(program, &code[0]);
        metalrtcDestroyProgram(&program);

        // If possible, write the CO out to a cache file for later use.

        try {
            ofstream out(cacheFile.c_str(), ios::out | ios::binary);
            out.write(&code[0], code.size());
            out.close();
        }
        catch (...) {
            // An error occurred.  Possibly we don't have permission to write to the temp directory.
            // Ignore.
        }
        CHECK_RESULT2(metalModuleLoadDataEx(&module, &code[0], 0, NULL, NULL), "Error loading METAL module");
        loadedModules.push_back(module);
        return module;
    }
    catch (...) {
        metalrtcDestroyProgram(&program);
        throw;
    }
}

metalFunction_t MetalContext::getKernel(metalModule_t& module, const string& name) {
    metalFunction_t function;
    metalError_t result = metalModuleGetFunction(&function, module, name.c_str());
    if (result != metalSuccess) {
        std::stringstream m;
        m<<"Error creating kernel "<<name<<": "<<getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(m.str());
    }
    return function;
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
    return shared_ptr<ComputeQueueImpl>(new MetalQueue());
}

metalStream_t MetalContext::getCurrentStream() {
    return dynamic_cast<MetalQueue*>(currentQueue.get())->getStream();
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

std::string MetalContext::getErrorString(metalError_t result) {
    return string(metalGetErrorName(result));
}

void MetalContext::executeKernel(metalFunction_t kernel, void** arguments, int threads, int blockSize, unsigned int sharedSize) {
    if (blockSize == -1)
        blockSize = ThreadBlockSize;
    int gridSize = std::min((threads+blockSize-1)/blockSize, numThreadBlocks);
    metalError_t result = metalModuleLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, sharedSize, getCurrentStream(), arguments, NULL);
    if (result != metalSuccess) {
        stringstream str;
        str<<"Error invoking kernel: "<<getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(str.str());
    }
}

void MetalContext::executeKernelFlat(metalFunction_t kernel, void** arguments, int threads, int blockSize, unsigned int sharedSize) {
    if (blockSize == -1)
        blockSize = ThreadBlockSize;
    int gridSize = (threads+blockSize-1)/blockSize;
    metalError_t result = metalModuleLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, sharedSize, getCurrentStream(), arguments, NULL);
    if (result != metalSuccess) {
        stringstream str;
        str<<"Error invoking kernel: "<<getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(str.str());
    }
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
    metalStreamSynchronize(getCurrentStream());
}

vector<int> MetalContext::getDevicePrecedence() {
    int numDevices;
    metalDeviceProp_t thisDevice;
    string errorMessage = "Error initializing Context";
    vector<pair<int, int> > devices;

    CHECK_RESULT(metalGetDeviceCount(&numDevices));
    for (int i = 0; i < numDevices; i++) {
        CHECK_RESULT(metalGetDeviceProperties(&thisDevice, i));
        int clock, multiprocessors, speed;
        clock = thisDevice.clockRate;
        multiprocessors = thisDevice.multiProcessorCount;
        speed = clock*multiprocessors;
        devices.push_back(std::make_pair(speed, -i));
    }

    // sort first by speed (higher is better), and finally device index (lower is better)
    std::sort(devices.begin(), devices.end());
    std::reverse(devices.begin(), devices.end());

    vector<int> precedence;
    for (int i = 0; i < static_cast<int>(devices.size()); i++) {
        precedence.push_back(-devices[i].second);
    }

    return precedence;
}

unsigned int MetalContext::getEventFlags() {
    unsigned int flags = metalEventDisableTiming;
    return flags;
}

unsigned int MetalContext::getHostMallocFlags() {
    return hostMallocFlags;
}
