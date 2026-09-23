/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2011-2026 Stanford University and the Authors.      *
 * Portions copyright (c) 2020-2021 Advanced Micro Devices, Inc.              *
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

#include "MetalParallelKernels.h"
#include "MetalKernelSources.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/timer.h"

using namespace OpenMM;
using namespace std;


#define CHECK_RESULT(result, prefix) \
if (result != metalSuccess) { \
    std::stringstream m; \
    m<<prefix<<": "<<cu.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
    throw OpenMMException(m.str());\
}

class MetalParallelCalcForcesAndEnergyKernel::BeginComputationTask : public MetalContext::WorkTask {
public:
    BeginComputationTask(ContextImpl& context, MetalContext& cu, MetalCalcForcesAndEnergyKernel& kernel,
            bool includeForce, bool includeEnergy, int groups, void* pinnedMemory, metalEvent_t event) : context(context), cu(cu), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), groups(groups), pinnedMemory(pinnedMemory), event(event) {
    }
    void execute() {
        // Copy coordinates over to this device and execute the kernel.

        ContextSelector selector(cu);
        if (cu.getContextIndex() > 0) {
            metalStreamWaitEvent(cu.getCurrentStream(), event, 0);
            if (!cu.getPlatformData().peerAccessSupported)
                cu.getPosq().upload(pinnedMemory, false);
        }
        kernel.beginComputation(context, includeForce, includeEnergy, groups);
    }
private:
    ContextImpl& context;
    MetalContext& cu;
    MetalCalcForcesAndEnergyKernel& kernel;
    bool includeForce, includeEnergy;
    int groups;
    void* pinnedMemory;
    metalEvent_t event;
};

class MetalParallelCalcForcesAndEnergyKernel::FinishComputationTask : public MetalContext::WorkTask {
public:
    FinishComputationTask(ContextImpl& context, MetalContext& cu, MetalCalcForcesAndEnergyKernel& kernel,
            bool includeForce, bool includeEnergy, int groups, double& energy, double& completionTime, long long* pinnedMemory, MetalArray& contextForces,
            bool& valid, metalStream_t stream, metalEvent_t event, metalEvent_t localEvent) :
            context(context), cu(cu), kernel(kernel), includeForce(includeForce), includeEnergy(includeEnergy), groups(groups), energy(energy),
            completionTime(completionTime), pinnedMemory(pinnedMemory), contextForces(contextForces), valid(valid),
            stream(stream), event(event), localEvent(localEvent) {
    }
    void execute() {
        // Execute the kernel, then download forces.

        ContextSelector selector(cu);
        energy += kernel.finishComputation(context, includeForce, includeEnergy, groups, valid);
        if (cu.getComputeForceCount() < 200) {
            // Record timing information for load balancing.  Since this takes time, only do it at the start of the simulation.

            CHECK_RESULT(metalStreamSynchronize(cu.getCurrentStream()), "Error synchronizing METAL context");
            completionTime = getCurrentTime();
        }
        if (includeForce) {
            if (cu.getContextIndex() > 0) {
                metalEventRecord(localEvent, cu.getCurrentStream());
                metalStreamWaitEvent(stream, localEvent, 0);
                int numAtoms = cu.getPaddedNumAtoms();
                if (cu.getPlatformData().peerAccessSupported) {
                    int numBytes = numAtoms*3*sizeof(long long);
                    int offset = (cu.getContextIndex()-1)*numBytes;
                    CHECK_RESULT(metalMemcpyAsync(static_cast<char*>(contextForces.getDevicePointer())+offset,
                                           cu.unwrap(cu.getLongForceBuffer()).getDevicePointer(), numBytes, metalMemcpyDeviceToDevice, stream), "Error copying forces");
                    metalEventRecord(event, stream);
                }
                else
                    cu.getForce().download(&pinnedMemory[(cu.getContextIndex()-1)*numAtoms*3]);
            }
        }
    }
private:
    ContextImpl& context;
    MetalContext& cu;
    MetalCalcForcesAndEnergyKernel& kernel;
    bool includeForce, includeEnergy;
    int groups;
    double& energy;
    double& completionTime;
    long long* pinnedMemory;
    MetalArray& contextForces;
    bool& valid;
    metalStream_t stream;
    metalEvent_t event;
    metalEvent_t localEvent;
};

MetalParallelCalcForcesAndEnergyKernel::MetalParallelCalcForcesAndEnergyKernel(string name, const Platform& platform, MetalPlatform::PlatformData& data) :
        CalcForcesAndEnergyKernel(name, platform), data(data), completionTimes(data.contexts.size()), contextNonbondedFractions(data.contexts.size()),
        pinnedPositionBuffer(NULL), pinnedForceBuffer(NULL) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new MetalCalcForcesAndEnergyKernel(name, platform, *data.contexts[i])));
}

MetalParallelCalcForcesAndEnergyKernel::~MetalParallelCalcForcesAndEnergyKernel() {
    ContextSelector selector(*data.contexts[0]);
    if (pinnedPositionBuffer != NULL)
        metalHostFree(pinnedPositionBuffer);
    if (pinnedForceBuffer != NULL)
        metalHostFree(pinnedForceBuffer);
    metalEventDestroy(event);
    for (int i = 0; i < peerCopyEvent.size(); i++)
        metalEventDestroy(peerCopyEvent[i]);
    for (int i = 0; i < peerCopyEventLocal.size(); i++)
        metalEventDestroy(peerCopyEventLocal[i]);
    for (int i = 0; i < peerCopyStream.size(); i++)
        metalStreamDestroy(peerCopyStream[i]);
}

void MetalParallelCalcForcesAndEnergyKernel::initialize(const System& system) {
    MetalContext& cu = *data.contexts[0];
    ContextSelector selector(cu);
    metalModule_t module = cu.createModule(MetalKernelSources::parallel);
    sumKernel = cu.getKernel(module, "sumForces");
    int numContexts = data.contexts.size();
    for (int i = 0; i < numContexts; i++)
        getKernel(i).initialize(system);
    for (int i = 0; i < numContexts; i++)
        contextNonbondedFractions[i] = 1/(double) numContexts;
    CHECK_RESULT(metalEventCreateWithFlags(&event, cu.getEventFlags()), "Error creating event");
    peerCopyEvent.resize(numContexts);
    peerCopyEventLocal.resize(numContexts);
    peerCopyStream.resize(numContexts);
    for (int i = 0; i < numContexts; i++) {
        MetalContext& cuLocal = *data.contexts[i];
        ContextSelector selectorLocal(cuLocal);
        CHECK_RESULT(metalEventCreateWithFlags(&peerCopyEvent[i], cu.getEventFlags()), "Error creating event");
        CHECK_RESULT(metalStreamCreateWithFlags(&peerCopyStream[i], metalStreamNonBlocking), "Error creating stream");
        CHECK_RESULT(metalEventCreateWithFlags(&peerCopyEventLocal[i], cu.getEventFlags()), "Error creating event");
    }
}

void MetalParallelCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups) {
    MetalContext& cu = *data.contexts[0];
    ContextSelector selector(cu);
    if (!contextForces.isInitialized()) {
        contextForces.initialize<long long>(cu, 3*(data.contexts.size()-1)*cu.getPaddedNumAtoms(), "contextForces");
        if (!cu.getPlatformData().peerAccessSupported) {
            CHECK_RESULT(metalHostMalloc((void**) &pinnedForceBuffer, 3*(data.contexts.size()-1)*cu.getPaddedNumAtoms()*sizeof(long long), metalHostMallocPortable), "Error allocating pinned memory");
            CHECK_RESULT(metalHostMalloc(&pinnedPositionBuffer, cu.getPaddedNumAtoms()*(cu.getUseDoublePrecision() ? sizeof(double4) : sizeof(float4)), metalHostMallocPortable), "Error allocating pinned memory");
        }
    }

    // Copy coordinates over to each device and execute the kernel.

    if (!cu.getPlatformData().peerAccessSupported) {
        cu.getPosq().download(pinnedPositionBuffer, false);
        metalEventRecord(event, cu.getCurrentStream());
    }
    else {
        int numBytes = cu.getPosq().getSize()*cu.getPosq().getElementSize();
        metalEventRecord(event, cu.getCurrentStream());
        for (int i = 1; i < (int) data.contexts.size(); i++) {
            metalStreamWaitEvent(peerCopyStream[i], event, 0);
            CHECK_RESULT(metalMemcpyAsync(
                cu.unwrap(data.contexts[i]->getPosq()).getDevicePointer(),
                cu.unwrap(cu.getPosq()).getDevicePointer(), numBytes,
                metalMemcpyDeviceToDevice, peerCopyStream[i]), "Error copying positions");
            metalEventRecord(peerCopyEvent[i], peerCopyStream[i]);
        }
    }
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        data.contextEnergy[i] = 0.0;
        MetalContext& cu = *data.contexts[i];
        ComputeContext::WorkThread& thread = cu.getWorkThread();
        metalEvent_t waitEvent = (cu.getPlatformData().peerAccessSupported ? peerCopyEvent[i] : event);
        thread.addTask(new BeginComputationTask(context, cu, getKernel(i), includeForce, includeEnergy, groups, pinnedPositionBuffer, waitEvent));
    }
    data.syncContexts();
}

double MetalParallelCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups, bool& valid) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cu = *data.contexts[i];
        ComputeContext::WorkThread& thread = cu.getWorkThread();
        thread.addTask(new FinishComputationTask(context, cu, getKernel(i), includeForce, includeEnergy, groups, data.contextEnergy[i], completionTimes[i],
                pinnedForceBuffer, contextForces, valid, peerCopyStream[i], peerCopyEvent[i], peerCopyEventLocal[i]));
    }
    data.syncContexts();
    MetalContext& cu = *data.contexts[0];
    ContextSelector selector(cu);
    if (cu.getPlatformData().peerAccessSupported)
        for (int i = 1; i < data.contexts.size(); i++)
            metalStreamWaitEvent(cu.getCurrentStream(), peerCopyEvent[i], 0);
    double energy = 0.0;
    for (int i = 0; i < (int) data.contextEnergy.size(); i++)
        energy += data.contextEnergy[i];
    if (includeForce && valid) {
        // Sum the forces from all devices.

        if (!cu.getPlatformData().peerAccessSupported)
            contextForces.upload(pinnedForceBuffer, false);
        int bufferSize = 3*cu.getPaddedNumAtoms();
        int numBuffers = data.contexts.size()-1;
        void* args[] = {&cu.unwrap(cu.getLongForceBuffer()).getDevicePointer(), &contextForces.getDevicePointer(), &bufferSize, &numBuffers};
        cu.executeKernel(sumKernel, args, bufferSize);

        // Balance work between the contexts by transferring a little nonbonded work from the context that
        // finished last to the one that finished first.

        if (cu.getComputeForceCount() < 200) {
            int firstIndex = 0, lastIndex = 0;
            const double eps = 0.001;
            for (int i = 0; i < (int) completionTimes.size(); i++) {
                if (completionTimes[i] < completionTimes[firstIndex])
                    firstIndex = i;
                if (contextNonbondedFractions[lastIndex] < eps || completionTimes[i] > completionTimes[lastIndex])
                    lastIndex = i;
            }
            double fractionToTransfer = min(cu.getComputeForceCount() < 100 ? 0.01 : 0.001, contextNonbondedFractions[lastIndex]);
            contextNonbondedFractions[firstIndex] += fractionToTransfer;
            contextNonbondedFractions[lastIndex] -= fractionToTransfer;
            double startFraction = 0.0;
            for (int i = 0; i < (int) contextNonbondedFractions.size(); i++) {
                double endFraction = startFraction+contextNonbondedFractions[i];
                if (i == contextNonbondedFractions.size()-1)
                    endFraction = 1.0; // Avoid roundoff error
                data.contexts[i]->getNonbondedUtilities().setAtomBlockRange(startFraction, endFraction);
                startFraction = endFraction;
            }
        }
    }
    return energy;
}

class MetalParallelCalcNonbondedForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, MetalCalcNonbondedForceKernel& kernel, bool includeForce,
            bool includeEnergy, bool includeDirect, bool includeReciprocal, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), includeDirect(includeDirect), includeReciprocal(includeReciprocal), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy, includeDirect, includeReciprocal);
    }
private:
    ContextImpl& context;
    MetalCalcNonbondedForceKernel& kernel;
    bool includeForce, includeEnergy, includeDirect, includeReciprocal;
    double& energy;
};

MetalParallelCalcNonbondedForceKernel::MetalParallelCalcNonbondedForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcNonbondedForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new MetalCalcNonbondedForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cu = *data.contexts[i];
        ComputeContext::WorkThread& thread = cu.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, includeDirect, includeReciprocal, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force, int firstParticle, int lastParticle, int firstException, int lastException) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force, firstParticle, lastParticle, firstException, lastException);
}

void MetalParallelCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const MetalCalcNonbondedForceKernel&>(kernels[0].getImpl()).getPMEParameters(alpha, nx, ny, nz);
}

void MetalParallelCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const MetalCalcNonbondedForceKernel&>(kernels[0].getImpl()).getLJPMEParameters(alpha, nx, ny, nz);
}
