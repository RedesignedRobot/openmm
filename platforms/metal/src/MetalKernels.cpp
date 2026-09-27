/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.      *
 * Portions copyright (c) 2020-2022 Advanced Micro Devices, Inc.              *
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

#include "MetalKernels.h"
#include "MetalForceInfo.h"
#include "openmm/Context.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/CommonKernelUtilities.h"
#include "CommonKernelSources.h"
#include "MetalBondedUtilities.h"
#include "MetalExpressionUtilities.h"
#include "MetalIntegrationUtilities.h"
#include "MetalNonbondedUtilities.h"
#include "MetalKernelSources.h"
#include "MetalQueue.h"
#include "SimTKOpenMMRealType.h"
#include "SimTKOpenMMUtilities.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <assert.h>

using namespace OpenMM;
using namespace std;

static void getMetalPmeParameters(MetalContext& cu, bool& usePmeQueue, bool& useFixedPointChargeSpreading) {
    usePmeQueue = (!cu.getPlatformData().disablePmeStream && !cu.getPlatformData().useCpuPme);
    useFixedPointChargeSpreading = cu.getUseDoublePrecision() || !cu.getSupportsHardwareFloatGlobalAtomicAdd() || cu.getPlatformData().deterministicForces;
}

// Sets the queue's dispatch type for a scope and restores serial dispatch on the way out, even when an
// exception leaves the scope.
struct ConcurrentDispatchScope {
    ConcurrentDispatchScope(MetalQueue& queue, bool concurrent) : queue(queue) {
        queue.setConcurrentDispatch(concurrent);
    }
    ~ConcurrentDispatchScope() {
        queue.setConcurrentDispatch(false);
    }
    MetalQueue& queue;
};

void MetalCalcForcesAndEnergyKernel::initialize(const System& system) {
}

void MetalCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    cu.setForcesValid(true);
    ContextSelector selector(cu);
    cu.clearAutoclearBuffers();
    cu.updateGlobalParamValues();
    for (auto computation : cu.getPreComputations())
        computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    MetalNonbondedUtilities& nb = cu.getNonbondedUtilities();
    cu.setComputeForceCount(cu.getComputeForceCount()+1);
    nb.prepareInteractions(groups);
    map<string, double>& derivs = cu.getEnergyParamDerivWorkspace();
    for (auto& param : context.getParameters())
        derivs[param.first] = 0;
}

double MetalCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups, bool& valid) {
    ContextSelector selector(cu);

    // The bonded and nonbonded kernels share no written buffer except the force buffer, and both only add
    // to it with integer atomics, so they can run side by side and still give the same bits.  Each also does
    // a plain += into the energy buffer (bonded on every step, nonbonded only with energy), and each writes
    // energy parameter derivatives when there are any, so those steps keep them in order.  The pass
    // boundaries on either side order them against the clear before and the readers after.

    {
        bool overlap = (includeForces && !includeEnergy && cu.getEnergyParamDerivNames().empty());
        ConcurrentDispatchScope scope(cu.getCurrentStream(), overlap);
        cu.getBondedUtilities().computeInteractions(groups);
        cu.getNonbondedUtilities().computeInteractions(groups, includeForces, includeEnergy);
    }
    double sum = 0.0;
    for (auto computation : cu.getPostComputations())
        sum += computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    cu.getIntegrationUtilities().distributeForcesFromVirtualSites();
    if (includeEnergy)
        sum += cu.reduceEnergy();
    if (!cu.getForcesValid())
        valid = false;
    return sum;
}

void MetalCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
    bool usePmeQueue, useFixedPointChargeSpreading;
    getMetalPmeParameters(cu, usePmeQueue, useFixedPointChargeSpreading);
    // Sorting atoms by PME grid index costs more than it saves: reorderAtoms already keeps the stored order spatially coherent.
    commonInitialize(system, force, usePmeQueue, false, useFixedPointChargeSpreading, cu.getPlatformData().useCpuPme, false);
}

void MetalCalcConstantPotentialForceKernel::initialize(const System& system, const ConstantPotentialForce& force) {
    bool usePmeQueue, useFixedPointChargeSpreading;
    getMetalPmeParameters(cu, usePmeQueue, useFixedPointChargeSpreading);
    commonInitialize(system, force, false, useFixedPointChargeSpreading);
}

void MetalIntegrateLangevinMiddleStepKernel::integrate(double tol) {
    if (!cu.getIntegrationUtilities().integrateLangevinMiddle(params, tol))
        CommonIntegrateLangevinMiddleStepKernel::integrate(tol);
}

// Lanes that share one atom in the reduction kernels.
static const int REDUCE_LANES = 8;

// The block-pair kernels win when the half-list launch leaves the GPU short of work, so that its tail after the last
// atomics is exposed, and lose when the launch already fills the GPU, since they read and write one partial per atom
// per block.  gbsa (78 blocks) has 1.27 block pairs per SIMD group of the launch on the M3 Ultra (60 cores), where
// they gain, and 7.6 on the M2 (10 cores), where they cost 12-14%.  The count per SIMD group comes from the core
// count, so this picks the path per device and system.
static const double MAX_OWNED_BLOCK_PAIRS_PER_WARP = 2.0;

bool MetalCalcGBSAOBCForceKernel::createOwnedKernels() {
    MetalNonbondedUtilities& nb = cu.getNonbondedUtilities();
    if (!nb.getUseCutoff() || cu.getUseDoublePrecision())
        return false;

    // The partial buffers hold one slot per atom per block.  Keep them inside 1% of the working set.

    int numBlocks = cu.getNumAtomBlocks();
    int paddedAtoms = cu.getPaddedNumAtoms();
    int threads = nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize();
    int numWarps = threads/32;
    if ((double) numBlocks*numBlocks > MAX_OWNED_BLOCK_PAIRS_PER_WARP*numWarps)
        return false;
    long long partialBytes = (long long) numBlocks*paddedAtoms*(sizeof(float)+sizeof(mm_float4));
    if (partialBytes > cu.getDevice()->recommendedMaxWorkingSetSize()/100)
        return false;
    partialBornSum.initialize<float>(cu, numBlocks*paddedAtoms, "partialBornSum");
    partialForce.initialize<mm_float4>(cu, numBlocks*paddedAtoms, "partialForce");
    atomData.initialize<mm_float4>(cu, paddedAtoms, "gbsaAtomData");
    map<string, string> defines;
    if (nb.getUsePeriodic())
        defines["USE_PERIODIC"] = "1";
    defines["CUTOFF_SQUARED"] = cu.doubleToString(cutoff*cutoff);
    defines["CUTOFF"] = cu.doubleToString(cutoff);
    defines["PREFACTOR"] = cu.doubleToString(prefactor);
    defines["SURFACE_AREA_FACTOR"] = cu.doubleToString(surfaceAreaFactor);
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(paddedAtoms);
    defines["NUM_BLOCKS"] = cu.intToString(numBlocks);
    defines["NUM_BLOCK_PAIRS"] = cu.intToString(numBlocks*(numBlocks+1)/2);
    defines["TILE_SIZE"] = "32";
    defines["REDUCE_LANES"] = cu.intToString(REDUCE_LANES);
    defines["DIELECTRIC_OFFSET"] = "0.009f";
    defines["PROBE_RADIUS"] = "0.14f";
    ComputeProgram program = cu.compileProgram(MetalKernelSources::gbsaObcOwned, defines);
    ownedBornSumKernel = program->createKernel("computeBornSumOwned");
    ownedBornSumKernel->addArg(partialBornSum);
    ownedBornSumKernel->addArg(cu.getPosq());
    ownedBornSumKernel->addArg(params);
    for (int i = 0; i < 5; i++)
        ownedBornSumKernel->addArg(); // The periodic box size arguments are set when the kernel is executed.
    ownedBornSumKernel->addArg(nb.getBlockCenters());
    ownedBornSumKernel->addArg(nb.getBlockBoundingBoxes());
    ownedReduceBornSumKernel = program->createKernel("reduceBornSumOwned");
    ownedReduceBornSumKernel->addArg(1.0f);
    ownedReduceBornSumKernel->addArg(0.8f);
    ownedReduceBornSumKernel->addArg(4.85f);
    ownedReduceBornSumKernel->addArg(partialBornSum);
    ownedReduceBornSumKernel->addArg(params);
    ownedReduceBornSumKernel->addArg(charges);
    ownedReduceBornSumKernel->addArg(bornRadii);
    ownedReduceBornSumKernel->addArg(obcChain);
    ownedReduceBornSumKernel->addArg(atomData);
    ownedForce1Kernel = program->createKernel("computeGBSAForce1Owned");
    ownedForce1Kernel->addArg(partialForce);
    ownedForce1Kernel->addArg(cu.getEnergyBuffer());
    ownedForce1Kernel->addArg(cu.getPosq());
    ownedForce1Kernel->addArg(atomData);
    ownedForce1Kernel->addArg(); // Whether to include energy.
    for (int i = 0; i < 5; i++)
        ownedForce1Kernel->addArg(); // The periodic box size arguments are set when the kernel is executed.
    ownedForce1Kernel->addArg(nb.getBlockCenters());
    ownedForce1Kernel->addArg(nb.getBlockBoundingBoxes());
    ownedReduceBornForceKernel = program->createKernel("reduceBornForceOwned");
    ownedReduceBornForceKernel->addArg(partialForce);
    ownedReduceBornForceKernel->addArg(cu.getLongForceBuffer());
    ownedReduceBornForceKernel->addArg(bornForce);
    ownedReduceBornForceKernel->addArg(cu.getEnergyBuffer());
    ownedReduceBornForceKernel->addArg(params);
    ownedReduceBornForceKernel->addArg(bornRadii);
    ownedReduceBornForceKernel->addArg(obcChain);
    return true;
}

double MetalCalcGBSAOBCForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    ContextSelector selector(cu);
    if (!hasCheckedOwned) {
        // The nonbonded utilities are initialized only after initialize() returns, so this can't happen earlier.
        hasCheckedOwned = true;
        useOwned = createOwnedKernels();
    }
    if (!useOwned)
        return CommonCalcGBSAOBCForceKernel::execute(context, includeForces, includeEnergy);
    MetalNonbondedUtilities& nb = cu.getNonbondedUtilities();
    int threads = nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize();
    ownedForce1Kernel->setArg(4, (int) includeEnergy);
    setPeriodicBoxArgs(cu, ownedBornSumKernel, 3);
    setPeriodicBoxArgs(cu, ownedForce1Kernel, 5);
    ownedBornSumKernel->execute(threads, nb.getForceThreadBlockSize());
    ownedReduceBornSumKernel->execute(cu.getPaddedNumAtoms()*REDUCE_LANES);
    ownedForce1Kernel->execute(threads, nb.getForceThreadBlockSize());
    ownedReduceBornForceKernel->execute(cu.getPaddedNumAtoms()*REDUCE_LANES);
    return 0.0;
}
