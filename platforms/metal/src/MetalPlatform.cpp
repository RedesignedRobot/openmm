/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2026 Stanford University and the Authors.      *
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
#include "MetalPlatform.h"
#include "MetalKernelFactory.h"
#include "MetalKernels.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/hardware.h"
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <cstdio>

using namespace OpenMM;
using namespace std;

#ifdef OPENMM_COMMON_BUILDING_STATIC_LIBRARY
extern "C" void registerMetalPlatform() {
    Platform::registerPlatform(new MetalPlatform());
}
#else
extern "C" OPENMM_EXPORT_COMMON void registerPlatforms() {
    if (MetalPlatform::isPlatformSupported())
        Platform::registerPlatform(new MetalPlatform());
}
#endif

static string getDeviceName(MTL::Device* device) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    string name = device->name()->utf8String();
    pool->release();
    return name;
}

MetalPlatform::MetalPlatform() {
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
    platformProperties.push_back(MetalDeviceName());
    platformProperties.push_back(MetalPrecision());
    platformProperties.push_back(MetalUseCpuPme());
    platformProperties.push_back(MetalTempDirectory());
    platformProperties.push_back(MetalDeterministicForces());
    setPropertyDefaultValue(MetalDeviceName(), "");
    setPropertyDefaultValue(MetalPrecision(), "single");
    setPropertyDefaultValue(MetalUseCpuPme(), "false");
    setPropertyDefaultValue(MetalDeterministicForces(), "false");
    char* tmpdir = getenv("TMPDIR");
    string tmp = (tmpdir == NULL ? string(P_tmpdir) : string(tmpdir));
    setPropertyDefaultValue(MetalTempDirectory(), tmp);
}

double MetalPlatform::getSpeed() const {
    return 100;
}

bool MetalPlatform::isPlatformSupported() {
    // Compiling kernels needs macOS 15 (MTLCompileOptions.mathMode), and the kernels assume the
    // 32 wide SIMD groups of Apple GPUs.

    if (__builtin_available(macOS 15.0, *)) {
        MTL::Device* device = MTL::CreateSystemDefaultDevice();
        if (device == NULL)
            return false;
        bool supported = device->supportsFamily(MTL::GPUFamilyApple7);
        device->release();
        return supported;
    }
    return false;
}

bool MetalPlatform::supportsDoublePrecision() const {
    return false;
}

const string& MetalPlatform::getPropertyValue(const Context& context, const string& property) const {
    const ContextImpl& impl = getContextImpl(context);
    const PlatformData* data = reinterpret_cast<const PlatformData*>(impl.getPlatformData());
    map<string, string>::const_iterator value = data->propertyValues.find(property);
    if (value != data->propertyValues.end())
        return value->second;
    return Platform::getPropertyValue(context, property);
}

void MetalPlatform::setPropertyValue(Context& context, const string& property, const string& value) const {
}

void MetalPlatform::contextCreated(ContextImpl& context, const map<string, string>& properties) const {
    string precisionPropValue = (properties.find(MetalPrecision()) == properties.end() ?
            getPropertyDefaultValue(MetalPrecision()) : properties.find(MetalPrecision())->second);
    string cpuPmePropValue = (properties.find(MetalUseCpuPme()) == properties.end() ?
            getPropertyDefaultValue(MetalUseCpuPme()) : properties.find(MetalUseCpuPme())->second);
    const string& tempPropValue = (properties.find(MetalTempDirectory()) == properties.end() ?
            getPropertyDefaultValue(MetalTempDirectory()) : properties.find(MetalTempDirectory())->second);
    string deterministicForcesValue = (properties.find(MetalDeterministicForces()) == properties.end() ?
            getPropertyDefaultValue(MetalDeterministicForces()) : properties.find(MetalDeterministicForces())->second);
    transform(precisionPropValue.begin(), precisionPropValue.end(), precisionPropValue.begin(), ::tolower);
    transform(cpuPmePropValue.begin(), cpuPmePropValue.end(), cpuPmePropValue.begin(), ::tolower);
    transform(deterministicForcesValue.begin(), deterministicForcesValue.end(), deterministicForcesValue.begin(), ::tolower);
    vector<string> pmeKernelName;
    pmeKernelName.push_back(CalcPmeReciprocalForceKernel::Name());
    if (!supportsKernels(pmeKernelName))
        cpuPmePropValue = "false";
    int threads = getNumProcessors();
    char* threadsEnv = getenv("OPENMM_CPU_THREADS");
    if (threadsEnv != NULL)
        stringstream(threadsEnv) >> threads;
    context.setPlatformData(new PlatformData(&context, context.getSystem(), precisionPropValue, cpuPmePropValue, tempPropValue,
            deterministicForcesValue, threads, NULL));
}

void MetalPlatform::linkedContextCreated(ContextImpl& context, ContextImpl& originalContext) const {
    Platform& platform = originalContext.getPlatform();
    string precisionPropValue = platform.getPropertyValue(originalContext.getOwner(), MetalPrecision());
    string cpuPmePropValue = platform.getPropertyValue(originalContext.getOwner(), MetalUseCpuPme());
    string tempPropValue = platform.getPropertyValue(originalContext.getOwner(), MetalTempDirectory());
    string deterministicForcesValue = platform.getPropertyValue(originalContext.getOwner(), MetalDeterministicForces());
    int threads = reinterpret_cast<PlatformData*>(originalContext.getPlatformData())->threads.getNumThreads();
    context.setPlatformData(new PlatformData(&context, context.getSystem(), precisionPropValue, cpuPmePropValue, tempPropValue,
            deterministicForcesValue, threads, &originalContext));
}

void MetalPlatform::contextDestroyed(ContextImpl& context) const {
    PlatformData* data = reinterpret_cast<PlatformData*>(context.getPlatformData());
    delete data;
}

MetalPlatform::PlatformData::PlatformData(ContextImpl* context, const System& system, const string& precisionProperty, const string& cpuPmeProperty,
            const string& tempProperty, const string& deterministicForcesProperty, int numThreads, ContextImpl* originalContext) :
                context(context), hasInitializedContexts(false), threads(numThreads) {
    MetalContext* originalMetalContext = NULL;
    if (originalContext != NULL)
        originalMetalContext = reinterpret_cast<PlatformData*>(originalContext->getPlatformData())->contexts[0];
    contexts.push_back(new MetalContext(system, precisionProperty, tempProperty, *this, originalMetalContext));
    useCpuPme = (cpuPmeProperty == "true");
    deterministicForces = (deterministicForcesProperty == "true");
    propertyValues[MetalPlatform::MetalDeviceName()] = getDeviceName(contexts[0]->getDevice());
    propertyValues[MetalPlatform::MetalPrecision()] = precisionProperty;
    propertyValues[MetalPlatform::MetalUseCpuPme()] = useCpuPme ? "true" : "false";
    propertyValues[MetalPlatform::MetalTempDirectory()] = tempProperty;
    propertyValues[MetalPlatform::MetalDeterministicForces()] = deterministicForces ? "true" : "false";
    contextEnergy.resize(contexts.size());
}

MetalPlatform::PlatformData::~PlatformData() {
    for (MetalContext* context : contexts)
        delete context;
}

void MetalPlatform::PlatformData::initializeContexts(const System& system) {
    if (hasInitializedContexts)
        return;
    for (MetalContext* context : contexts)
        context->initialize();
    hasInitializedContexts = true;
}
