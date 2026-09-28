/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2026 Stanford University and the Authors.      *
 * Portions copyright (c) 2020 Advanced Micro Devices, Inc.                   *
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

#include "MetalContext.h"
#include "MetalExpressionUtilities.h"
#include "MetalPlatform.h"
#include "MetalKernelFactory.h"
#include "MetalKernels.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/hardware.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <cstdio>
#ifdef _MSC_VER
    #include <Windows.h>
#endif
using namespace OpenMM;
using namespace std;

static bool isPlatformSupported() {
    if (__builtin_available(macOS 15.0, *)) {
        NS::SharedPtr<MTL::Device> device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
        return device && device->supportsFamily(MTL::GPUFamilyApple7);
    }
    return false;
}

static string getDeviceName(MTL::Device* device) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    string name = device->name()->utf8String();
    pool->release();
    return name;
}

// On macOS 14 or an older GPU the platform stays unregistered, so OpenCL or CPU gets picked.
#ifdef OPENMM_COMMON_BUILDING_STATIC_LIBRARY
extern "C" void registerMetalPlatform() {
    if (isPlatformSupported())
        Platform::registerPlatform(new MetalPlatform());
}
#else
extern "C" OPENMM_EXPORT_COMMON void registerPlatforms() {
    if (isPlatformSupported())
        Platform::registerPlatform(new MetalPlatform());
}
#endif

MetalPlatform::MetalPlatform() {
    deprecatedPropertyReplacements["MetalDeviceIndex"] = MetalDeviceIndex();
    deprecatedPropertyReplacements["MetalDeviceName"] = MetalDeviceName();
    deprecatedPropertyReplacements["MetalUseBlockingSync"] = MetalUseBlockingSync();
    deprecatedPropertyReplacements["MetalPrecision"] = MetalPrecision();
    deprecatedPropertyReplacements["MetalUseCpuPme"] = MetalUseCpuPme();
    deprecatedPropertyReplacements["MetalTempDirectory"] = MetalTempDirectory();
    deprecatedPropertyReplacements["MetalDisablePmeStream"] = MetalDisablePmeStream();
    deprecatedPropertyReplacements["MetalDeterministicForces"] = MetalDeterministicForces();
    MetalKernelFactory* factory = new MetalKernelFactory();
    registerKernelFactory(CalcForcesAndEnergyKernel::Name(), factory);
    registerKernelFactory(UpdateStateDataKernel::Name(), factory);
    registerKernelFactory(ApplyConstraintsKernel::Name(), factory);
    registerKernelFactory(VirtualSitesKernel::Name(), factory);
    registerKernelFactory(MinimizeKernel::Name(), factory);
    registerKernelFactory(CalcHarmonicBondForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomBondForceKernel::Name(), factory);
    registerKernelFactory(CalcHarmonicAngleForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomAngleForceKernel::Name(), factory);
    registerKernelFactory(CalcPeriodicTorsionForceKernel::Name(), factory);
    registerKernelFactory(CalcRBTorsionForceKernel::Name(), factory);
    registerKernelFactory(CalcCMAPTorsionForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomTorsionForceKernel::Name(), factory);
    registerKernelFactory(CalcNonbondedForceKernel::Name(), factory);
    registerKernelFactory(CalcConstantPotentialForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomNonbondedForceKernel::Name(), factory);
    registerKernelFactory(CalcGBSAOBCForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomGBForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomExternalForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomHbondForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomCentroidBondForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomCompoundBondForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomCPPForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomCVForceKernel::Name(), factory);
    registerKernelFactory(CalcOrientationRestraintForceKernel::Name(), factory);
    registerKernelFactory(CalcPythonForceKernel::Name(), factory);
    registerKernelFactory(CalcRGForceKernel::Name(), factory);
    registerKernelFactory(CalcRMSDForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomManyParticleForceKernel::Name(), factory);
    registerKernelFactory(CalcGayBerneForceKernel::Name(), factory);
    registerKernelFactory(CalcLCPOForceKernel::Name(), factory);
    registerKernelFactory(IntegrateVerletStepKernel::Name(), factory);
    registerKernelFactory(IntegrateNoseHooverStepKernel::Name(), factory);
    registerKernelFactory(IntegrateLangevinMiddleStepKernel::Name(), factory);
    registerKernelFactory(IntegrateBrownianStepKernel::Name(), factory);
    registerKernelFactory(IntegrateVariableVerletStepKernel::Name(), factory);
    registerKernelFactory(IntegrateVariableLangevinStepKernel::Name(), factory);
    registerKernelFactory(IntegrateCustomStepKernel::Name(), factory);
    registerKernelFactory(IntegrateDPDStepKernel::Name(), factory);
    registerKernelFactory(IntegrateQTBStepKernel::Name(), factory);
    registerKernelFactory(ApplyAndersenThermostatKernel::Name(), factory);
    registerKernelFactory(ApplyMonteCarloBarostatKernel::Name(), factory);
    registerKernelFactory(RemoveCMMotionKernel::Name(), factory);
    registerKernelFactory(CalcATMForceKernel::Name(), factory);
    platformProperties.push_back(MetalDeviceIndex());
    platformProperties.push_back(MetalDeviceName());
    platformProperties.push_back(MetalUseBlockingSync());
    platformProperties.push_back(MetalPrecision());
    platformProperties.push_back(MetalUseCpuPme());
    platformProperties.push_back(MetalTempDirectory());
    platformProperties.push_back(MetalDisablePmeStream());
    platformProperties.push_back(MetalDeterministicForces());
    setPropertyDefaultValue(MetalDeviceIndex(), "");
    setPropertyDefaultValue(MetalDeviceName(), "");
    setPropertyDefaultValue(MetalUseBlockingSync(), "true");
    setPropertyDefaultValue(MetalPrecision(), "single");
    setPropertyDefaultValue(MetalUseCpuPme(), "false");
    setPropertyDefaultValue(MetalDisablePmeStream(), "true");
    setPropertyDefaultValue(MetalDeterministicForces(), "false");
#ifdef _MSC_VER
    setPropertyDefaultValue(MetalTempDirectory(), string(getenv("TEMP")));
#else
    char* tmpdir = getenv("TMPDIR");
    string tmp = (tmpdir == NULL ? string(P_tmpdir) : string(tmpdir));
    setPropertyDefaultValue(MetalTempDirectory(), tmp);
#endif
}

double MetalPlatform::getSpeed() const {
    // Reduce the speed of the METAL platform if there are no METAL devices in the system,
    // so the OpenCL plaform can be selected as default
    return isPlatformSupported() ? 100 : 40;
}

bool MetalPlatform::supportsDoublePrecision() const {
    return false;
}

const string& MetalPlatform::getPropertyValue(const Context& context, const string& property) const {
    const ContextImpl& impl = getContextImpl(context);
    const PlatformData* data = reinterpret_cast<const PlatformData*>(impl.getPlatformData());
    string propertyName = property;
    if (deprecatedPropertyReplacements.find(property) != deprecatedPropertyReplacements.end())
        propertyName = deprecatedPropertyReplacements.find(property)->second;
    map<string, string>::const_iterator value = data->propertyValues.find(propertyName);
    if (value != data->propertyValues.end())
        return value->second;
    return Platform::getPropertyValue(context, property);
}

void MetalPlatform::setPropertyValue(Context& context, const string& property, const string& value) const {
}

vector<map<string, string> > MetalPlatform::getDevices(const map<string, string>& filters) const {
    // Check for properties that might act as filters.

    int deviceIndex = -1;
    if (filters.find(MetalDeviceIndex()) != filters.end())
        stringstream(filters.at(MetalDeviceIndex())) >> deviceIndex;
    string deviceName = (filters.find(MetalDeviceName()) == filters.end() ? "" : filters.at(MetalDeviceName()));

    // Loop over devices.

    vector<map<string, string> > results;
    NS::SharedPtr<MTL::Device> device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    int numDevices = (device ? 1 : 0);
    for (int i = 0; i < numDevices; i++) {
        if (deviceIndex != -1 && deviceIndex != i)
            continue;
        stringstream deviceNameStr;
        deviceNameStr << getDeviceName(device.get());
        if (deviceName.size() > 0 && deviceName != deviceNameStr.str())
            continue;
        stringstream deviceIndexStr;
        deviceIndexStr << i;
        map<string, string> properties = {{MetalDeviceIndex(), deviceIndexStr.str()},
                                          {MetalDeviceName(), deviceNameStr.str()}};
        results.push_back(properties);
    }
    return results;
}

void MetalPlatform::contextCreated(ContextImpl& context, const map<string, string>& properties) const {
    // The kernels compile as MSL 3.2 with MTLCompileOptions mathMode, both new in macOS 15.  On
    // macOS 14 setMathMode: is an unrecognized selector, which aborts the process instead of throwing.
    // registerPlatforms() already skips such Macs; this covers a MetalPlatform created by hand.
    if (!isPlatformSupported())
        throw OpenMMException("The Metal platform needs macOS 15 or later and an Apple7-family GPU (M1 or later)");
    const string& devicePropValue = (properties.find(MetalDeviceIndex()) == properties.end() ?
            getPropertyDefaultValue(MetalDeviceIndex()) : properties.find(MetalDeviceIndex())->second);
    string blockingPropValue = (properties.find(MetalUseBlockingSync()) == properties.end() ?
            getPropertyDefaultValue(MetalUseBlockingSync()) : properties.find(MetalUseBlockingSync())->second);
    string precisionPropValue = (properties.find(MetalPrecision()) == properties.end() ?
            getPropertyDefaultValue(MetalPrecision()) : properties.find(MetalPrecision())->second);
    string cpuPmePropValue = (properties.find(MetalUseCpuPme()) == properties.end() ?
            getPropertyDefaultValue(MetalUseCpuPme()) : properties.find(MetalUseCpuPme())->second);
    const string& tempPropValue = (properties.find(MetalTempDirectory()) == properties.end() ?
            getPropertyDefaultValue(MetalTempDirectory()) : properties.find(MetalTempDirectory())->second);
    string pmeStreamPropValue = (properties.find(MetalDisablePmeStream()) == properties.end() ?
            getPropertyDefaultValue(MetalDisablePmeStream()) : properties.find(MetalDisablePmeStream())->second);
    string deterministicForcesValue = (properties.find(MetalDeterministicForces()) == properties.end() ?
            getPropertyDefaultValue(MetalDeterministicForces()) : properties.find(MetalDeterministicForces())->second);
    transform(blockingPropValue.begin(), blockingPropValue.end(), blockingPropValue.begin(), ::tolower);
    transform(precisionPropValue.begin(), precisionPropValue.end(), precisionPropValue.begin(), ::tolower);
    transform(cpuPmePropValue.begin(), cpuPmePropValue.end(), cpuPmePropValue.begin(), ::tolower);
    transform(pmeStreamPropValue.begin(), pmeStreamPropValue.end(), pmeStreamPropValue.begin(), ::tolower);
    transform(deterministicForcesValue.begin(), deterministicForcesValue.end(), deterministicForcesValue.begin(), ::tolower);
    vector<string> pmeKernelName;
    pmeKernelName.push_back(CalcPmeReciprocalForceKernel::Name());
    if (!supportsKernels(pmeKernelName))
        cpuPmePropValue = "false";
    int threads = getNumProcessors();
    char* threadsEnv = getenv("OPENMM_CPU_THREADS");
    if (threadsEnv != NULL)
        stringstream(threadsEnv) >> threads;
    context.setPlatformData(new PlatformData(&context, context.getSystem(), devicePropValue, blockingPropValue, precisionPropValue, cpuPmePropValue, tempPropValue,
            pmeStreamPropValue, deterministicForcesValue, threads, NULL));
}

void MetalPlatform::linkedContextCreated(ContextImpl& context, ContextImpl& originalContext) const {
    Platform& platform = originalContext.getPlatform();
    string devicePropValue = platform.getPropertyValue(originalContext.getOwner(), MetalDeviceIndex());
    string blockingPropValue = platform.getPropertyValue(originalContext.getOwner(), MetalUseBlockingSync());
    string precisionPropValue = platform.getPropertyValue(originalContext.getOwner(), MetalPrecision());
    string cpuPmePropValue = platform.getPropertyValue(originalContext.getOwner(), MetalUseCpuPme());
    string tempPropValue = platform.getPropertyValue(originalContext.getOwner(), MetalTempDirectory());
    string pmeStreamPropValue = platform.getPropertyValue(originalContext.getOwner(), MetalDisablePmeStream());
    string deterministicForcesValue = platform.getPropertyValue(originalContext.getOwner(), MetalDeterministicForces());
    int threads = reinterpret_cast<PlatformData*>(originalContext.getPlatformData())->threads.getNumThreads();
    context.setPlatformData(new PlatformData(&context, context.getSystem(), devicePropValue, blockingPropValue, precisionPropValue, cpuPmePropValue, tempPropValue,
            pmeStreamPropValue, deterministicForcesValue, threads, &originalContext));
}

void MetalPlatform::contextDestroyed(ContextImpl& context) const {
    PlatformData* data = reinterpret_cast<PlatformData*>(context.getPlatformData());
    delete data;
}

MetalPlatform::PlatformData::PlatformData(ContextImpl* context, const System& system, const string& deviceIndexProperty, const string& blockingProperty, const string& precisionProperty,
            const string& cpuPmeProperty, const string& tempProperty, const string& pmeStreamProperty,
            const string& deterministicForcesProperty, int numThreads, ContextImpl* originalContext) :
                context(context), removeCM(false), stepCount(0), computeForceCount(0), time(0.0), hasInitializedContexts(false),
                threads(numThreads) {
    bool blocking = (blockingProperty == "true");
    vector<string> devices;
    size_t searchPos = 0, nextPos;
    while ((nextPos = deviceIndexProperty.find_first_of(", ", searchPos)) != string::npos) {
        devices.push_back(deviceIndexProperty.substr(searchPos, nextPos-searchPos));
        searchPos = nextPos+1;
    }
    devices.push_back(deviceIndexProperty.substr(searchPos));
    PlatformData* originalData = NULL;
    if (originalContext != NULL)
        originalData = reinterpret_cast<PlatformData*>(originalContext->getPlatformData());
    try {
        for (int i = 0; i < (int) devices.size(); i++) {
            if (devices[i].length() > 0) {
                int deviceIndex;
                stringstream(devices[i]) >> deviceIndex;
                contexts.push_back(new MetalContext(system, deviceIndex, blocking, precisionProperty, tempProperty, *this, (originalData == NULL ? NULL : originalData->contexts[i])));
            }
        }
        if (contexts.size() == 0)
            contexts.push_back(new MetalContext(system, -1, blocking, precisionProperty, tempProperty, *this, (originalData == NULL ? NULL : originalData->contexts[0])));
        if (contexts.size() > 1)
            throw OpenMMException("The METAL platform does not support multiple devices");
    }
    catch (...) {
        // If an exception was thrown, do our best to clean up memory.

        for (int i = 0; i < (int) contexts.size(); i++)
            delete contexts[i];
        throw;
    }
    stringstream deviceIndex, deviceName;
    for (int i = 0; i < (int) contexts.size(); i++) {
        if (i > 0) {
            deviceIndex << ',';
            deviceName << ',';
        }
        deviceIndex << contexts[i]->getDeviceIndex();
        deviceName << getDeviceName(contexts[i]->getDevice());
    }

    useCpuPme = (cpuPmeProperty == "true" && !contexts[0]->getUseDoublePrecision());
    disablePmeStream = (pmeStreamProperty == "true");
    deterministicForces = (deterministicForcesProperty == "true");
    propertyValues[MetalPlatform::MetalDeviceIndex()] = deviceIndex.str();
    propertyValues[MetalPlatform::MetalDeviceName()] = deviceName.str();
    propertyValues[MetalPlatform::MetalUseBlockingSync()] = blocking ? "true" : "false";
    propertyValues[MetalPlatform::MetalPrecision()] = precisionProperty;
    propertyValues[MetalPlatform::MetalUseCpuPme()] = useCpuPme ? "true" : "false";
    propertyValues[MetalPlatform::MetalTempDirectory()] = tempProperty;
    propertyValues[MetalPlatform::MetalDisablePmeStream()] = disablePmeStream ? "true" : "false";
    propertyValues[MetalPlatform::MetalDeterministicForces()] = deterministicForces ? "true" : "false";
    contextEnergy.resize(contexts.size());
}

MetalPlatform::PlatformData::~PlatformData() {
    for (int i = 0; i < (int) contexts.size(); i++)
        delete contexts[i];
}

void MetalPlatform::PlatformData::initializeContexts(const System& system) {
    if (hasInitializedContexts)
        return;
    for (int i = 0; i < (int) contexts.size(); i++)
        contexts[i]->initialize();
    hasInitializedContexts = true;
}

void MetalPlatform::PlatformData::syncContexts() {
    for (int i = 0; i < (int) contexts.size(); i++)
        contexts[i]->getWorkThread().flush();
}
