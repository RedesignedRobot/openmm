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
#include "ReferenceTabulatedFunction.h"
#include "SimTKOpenMMRealType.h"
#include "SimTKOpenMMUtilities.h"
#include "lepton/CustomFunction.h"
#include "lepton/Operation.h"
#include "lepton/Parser.h"
#include "lepton/ParsedExpression.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <sstream>
#include <typeinfo>
#include <set>
#include <random>
#include <iostream>
#include "MetalAtomReorder.h"
#include <assert.h>

using namespace OpenMM;
using namespace std;

static void getMetalPmeParameters(MetalContext& cu, bool& usePmeQueue, bool& useFixedPointChargeSpreading) {
    usePmeQueue = (!cu.getPlatformData().disablePmeStream && !cu.getPlatformData().useCpuPme);
    useFixedPointChargeSpreading = cu.getUseFixedPointChargeSpreading();
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
    // A new evaluation supersedes any deferred result.  Drain its private words before the clear.
    cu.getNonbondedUtilities().scatterRoleForces();
    cu.clearAutoclearBuffersIfNeeded();
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
    // boundaries on either side order them against the clear before and the readers after.  A pending
    // calcCenterOfMassVelocity joins the pass, since it only reads the velocities, which no force kernel
    // writes.  It goes first so its single thread group starts before the force kernels fill the GPU.  Force
    // pass kernels either write only a buffer no other pass kernel touches (PME charge spreading into its
    // grid) or add to the force buffer only with integer atomics (PME interpolation).  Work that reads what the
    // pass wrote (the rest of the PME chain after spreading) runs right after it, before the post computations
    // and virtual site force distribution.  A refill of the random number pool joins too: it writes only the
    // pool and its seeds, which only the integrator reads, after the pass.

    {
        bool overlap = (includeForces && !includeEnergy && cu.getEnergyParamDerivNames().empty());
        ConcurrentDispatchScope scope(cu.getCurrentStream(), overlap);
        cu.getIntegrationUtilities().calcPendingCenterOfMassMomentum();
        cu.getIntegrationUtilities().refillExhaustedRandomNumbers();
        cu.executeForcePassKernels();
        cu.getBondedUtilities().computeInteractions(groups);
        // The kinetic energy kernel reads only the velocities and writes only its own result, so in a pass with
        // energy it runs beside the nonbonded kernel.  Its single thread group goes first, as above.
        MetalIntegrationUtilities& integration = cu.getIntegrationUtilities();
        if (includeEnergy && integration.getPrefetchKineticEnergy()) {
            cu.getCurrentStream().setConcurrentDispatch(true);
            integration.prefetchKineticEnergy();
        }
        cu.getNonbondedUtilities().computeInteractions(groups, includeForces, includeEnergy);
    }
    cu.executeAfterForcePass();
    // Only the prepared fused LangevinMiddle consumer can read the private words.  Energy evaluations,
    // other integrators (including CompoundIntegrator), and unfused virtual sites keep materialization.
    if (!includeForces || includeEnergy || dynamic_cast<LangevinMiddleIntegrator*>(&context.getIntegrator()) == NULL ||
            !cu.getIntegrationUtilities().canFoldOrderedForces())
        cu.getNonbondedUtilities().scatterRoleForces();
    double sum = 0.0;
    for (auto computation : cu.getPostComputations())
        sum += computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    cu.getIntegrationUtilities().distributeForcesFromVirtualSites();
    if (includeEnergy) {
        sum += cu.reduceEnergy();
        cu.getIntegrationUtilities().finishKineticEnergyPrefetch();
    }
    // Without a neighbor list nothing else inside step() commits, and the GPU would sit idle until the host
    // next waits for a result.
    if (!cu.getNonbondedUtilities().getUseNeighborList())
        cu.getCurrentStream().commit();
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

void MetalCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force, int firstParticle, int lastParticle, int firstException, int lastException) {
    cu.getNonbondedUtilities().checkRoleOrder(force);
    CommonCalcNonbondedForceKernel::copyParametersToContext(context, force, firstParticle, lastParticle, firstException, lastException);
}

void MetalCalcConstantPotentialForceKernel::initialize(const System& system, const ConstantPotentialForce& force) {
    bool usePmeQueue, useFixedPointChargeSpreading;
    getMetalPmeParameters(cu, usePmeQueue, useFixedPointChargeSpreading);
    commonInitialize(system, force, false, useFixedPointChargeSpreading);
}

/**
 * Find an operation in an expression that Metal mixed precision cannot evaluate per DOF, where every value is a
 * double-single pair.  Integer powers are multiplied out, so only other exponents need pow().  Returns an empty
 * string if there is none.
 */
static string findUnsupportedMixedOperation(const Lepton::ExpressionTreeNode& node) {
    // Functions are matched by name: SimTKOpenMMRealType.h defines TAN, ASIN and the rest as macros, which would
    // rewrite the Lepton::Operation names.
    static const set<string> unsupportedFunctions = {"tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "erf", "erfc"};
    const Lepton::Operation& op = node.getOperation();
    if (unsupportedFunctions.count(op.getName()) != 0)
        return op.getName();
    switch (op.getId()) {
        case Lepton::Operation::POWER:
            return "^ with a variable exponent";
        case Lepton::Operation::POWER_CONSTANT: {
            double exponent = dynamic_cast<const Lepton::Operation::PowerConstant&>(op).getValue();
            if (exponent != (int) exponent) {
                stringstream name;
                name << "^" << exponent;
                return name.str();
            }
            break;
        }
        default:
            break;
    }
    for (auto& child : node.getChildren()) {
        string found = findUnsupportedMixedOperation(child);
        if (!found.empty())
            return found;
    }
    return "";
}

static void checkMixedPrecisionExpression(const string& expression, const string& where, const map<string, Lepton::CustomFunction*>& functions) {
    string operation = findUnsupportedMixedOperation(Lepton::Parser::parse(expression, functions).optimize().getRootNode());
    if (!operation.empty())
        throw OpenMMException("Metal mixed precision cannot evaluate "+operation+" in the CustomIntegrator "+where+" '"+expression+
            "'.  In mixed precision, per-DOF expressions cannot use tan, asin, acos, atan, atan2, sinh, cosh, tanh, erf, erfc or a non-integer power."
            "  Use Precision=single, or compute the value in a global expression.");
}

void MetalIntegrateCustomStepKernel::initialize(const System& system, const CustomIntegrator& integrator) {
    if (cu.getUseMixedPrecision()) {
        // Parse per-DOF expressions the way CustomIntegratorUtilities::analyzeComputations does.

        vector<unique_ptr<Lepton::CustomFunction> > tables;
        map<string, Lepton::CustomFunction*> functions;
        for (int i = 0; i < integrator.getNumTabulatedFunctions(); i++) {
            tables.emplace_back(createReferenceTabulatedFunction(integrator.getTabulatedFunction(i)));
            functions[integrator.getTabulatedFunctionName(i)] = tables.back().get();
        }
        Lepton::PlaceholderFunction fn1(1), fn2(2), fn3(3);
        for (string name : {"_x", "_y", "_z"})
            functions[name] = &fn1;
        for (string name : {"deriv", "dot", "cross"})
            functions[name] = &fn2;
        functions["vector"] = &fn3;
        for (int step = 0; step < integrator.getNumComputations(); step++) {
            CustomIntegrator::ComputationType type;
            string variable, expression;
            integrator.getComputationStep(step, type, variable, expression);
            if (type == CustomIntegrator::ComputePerDof || type == CustomIntegrator::ComputeSum)
                checkMixedPrecisionExpression(expression, "step "+cu.intToString(step)+" expression", functions);
        }
        checkMixedPrecisionExpression(integrator.getKineticEnergyExpression(), "kinetic energy expression", functions);
    }
    CommonIntegrateCustomStepKernel::initialize(system, integrator);
}

/**
 * Get whether a force leaves the velocities alone in updateContextState() and calcForcesAndEnergy().  Unknown forces,
 * thermostats, other barostats and forces that evaluate an inner Context count as touching them.
 *
 * MonteCarloBarostat counts as leaving them alone.  Its trial reads only potential energies (the kinetic energy that
 * getState() adds is dropped), and a rejected trial restores velm whole, so the velocities the integrator sees are the
 * same whether the center of mass motion comes off before the trial or inside the integrator.  openmm.app adds the
 * barostat after CMMotionRemover, so without this every system with one pays two extra dispatches per step.  The
 * anisotropic, flexible and membrane barostats stay out: some of them read the kinetic energy inside a trial.
 */
static bool ignoresVelocities(const Force& force) {
    return typeid(force) == typeid(MonteCarloBarostat) ||
           dynamic_cast<const HarmonicBondForce*>(&force) != NULL || dynamic_cast<const HarmonicAngleForce*>(&force) != NULL ||
           dynamic_cast<const PeriodicTorsionForce*>(&force) != NULL || dynamic_cast<const RBTorsionForce*>(&force) != NULL ||
           dynamic_cast<const CMAPTorsionForce*>(&force) != NULL || dynamic_cast<const NonbondedForce*>(&force) != NULL ||
           dynamic_cast<const GBSAOBCForce*>(&force) != NULL || dynamic_cast<const CustomBondForce*>(&force) != NULL ||
           dynamic_cast<const CustomAngleForce*>(&force) != NULL || dynamic_cast<const CustomTorsionForce*>(&force) != NULL ||
           dynamic_cast<const CustomNonbondedForce*>(&force) != NULL || dynamic_cast<const CustomGBForce*>(&force) != NULL ||
           dynamic_cast<const CustomExternalForce*>(&force) != NULL || dynamic_cast<const CustomCompoundBondForce*>(&force) != NULL;
}

void MetalRemoveCMMotionKernel::initialize(const System& system, const CMMotionRemover& force) {
    ContextSelector selector(cu);
    int forceIndex = 0;
    while (&system.getForce(forceIndex) != &force)
        forceIndex++;
    laterForcesIgnoreVelocities = true;
    for (int i = forceIndex+1; i < system.getNumForces(); i++)
        laterForcesIgnoreVelocities &= ignoresVelocities(system.getForce(i));
    int numAtoms = cu.getNumAtoms();
    cmMomentum.initialize<mm_float4>(cu, cu.getPaddedNumAtoms(), "cmMomentum");
    double totalMass = 0.0;
    for (int i = 0; i < numAtoms; i++)
        totalMass += system.getParticleMass(i);
    map<string, string> defines;
    defines["INVERSE_TOTAL_MASS"] = cu.doubleToString(totalMass == 0 ? 0.0 : 1.0/totalMass);
    defines["CM_INVERSE_TOTAL_MASS"] = cu.doubleToString(totalMass == 0 ? 0.0 : 1.0/totalMass, true);
    defines["CM_BLOCK_SIZE"] = cu.intToString(MaxCMBlockSize);
    ComputeProgram program = cu.compileProgram(CommonKernelSources::removeCM+MetalKernelSources::centerOfMass, defines);
    kernel1 = program->createKernel("calcCenterOfMassMomentum");
    kernel1->addArg(numAtoms);
    kernel1->addArg(cu.getVelm());
    kernel1->addArg(cmMomentum);
    kernel2 = program->createKernel("removeCenterOfMassMomentum");
    kernel2->addArg(numAtoms);
    kernel2->addArg(cu.getVelm());
    kernel2->addArg(cmMomentum);
    cmVelocity.initialize<mm_float4>(cu, 1, "cmVelocity");
    kernel3 = program->createKernel("calcCenterOfMassVelocity");
    kernel3->addArg(numAtoms);
    kernel3->addArg(cu.getVelm());
    kernel3->addArg(cmVelocity);
    // calcCenterOfMassVelocity runs as one thread group, so it gets as many threads as its pipeline allows.
    cmBlockSize = MaxCMBlockSize;
    while (cmBlockSize > kernel3->getMaxBlockSize())
        cmBlockSize /= 2;
}

void MetalRemoveCMMotionKernel::execute(ContextImpl& context) {
    ContextSelector selector(cu);
    int numAtoms = cu.getNumAtoms();
    MetalIntegrationUtilities& integration = cu.getIntegrationUtilities();
    if (laterForcesIgnoreVelocities && integration.canRemoveCenterOfMassMomentum() &&
            dynamic_cast<LangevinMiddleIntegrator*>(&context.getIntegrator()) != NULL) {
        integration.removeCenterOfMassMomentum(kernel3, cmBlockSize, cmVelocity);
        return;
    }
    kernel1->execute(numAtoms, 64);
    kernel2->execute(numAtoms, 64);
}

void MetalUpdateStateDataKernel::initialize(const System& system) {
    CommonUpdateStateDataKernel::initialize(system);
    ContextSelector selector(cu);
    bool mixed = cu.getUseMixedPrecision();
    cellOffsets.initialize<mm_int4>(cu, cu.getPaddedNumAtoms(), "cellOffsets");
    boxVectors.initialize(cu, 3, mixed ? sizeof(mm_double4) : sizeof(mm_float4), "unwrapBoxVectors");
    ComputeProgram program = cu.compileProgram(MetalKernelSources::unwrapPositions);
    unwrapKernel = program->createKernel("unwrapPositions");
    unwrapKernel->addArg(cu.getNumAtoms());
    unwrapKernel->addArg(cu.getPosq());
    if (mixed)
        unwrapKernel->addArg(cu.getPosqCorrection());
    else
        unwrapKernel->addArg(nullptr);
    unwrapKernel->addArg(cellOffsets);
    unwrapKernel->addArg(boxVectors);
}

void MetalApplyMonteCarloBarostatKernel::saveCoordinates(ContextImpl& context) {
    cu.getNonbondedUtilities().scatterRoleForces();
    CommonApplyMonteCarloBarostatKernel::saveCoordinates(context);
}

void MetalUpdateStateDataKernel::getForces(ContextImpl& context, vector<Vec3>& forces) {
    cu.getIntegrationUtilities().ensureVirtualSiteForcesDistributed();
    CommonUpdateStateDataKernel::getForces(context, forces);
    cu.resolveDeviceBox();
}

void MetalUpdateStateDataKernel::setPeriodicBoxVectors(ContextImpl& context, const Vec3& a, const Vec3& b, const Vec3& c) {
    if (cu.getUseDeviceBox()) {
        if (!cu.getBoxIsTriclinic() && (b[0] != 0 || c[0] != 0 || c[1] != 0))
            throw OpenMMException("The box shape has changed from rectangular to triclinic.  Reinitialize the Context with a triclinic default box.");
        ContextSelector selector(cu);
        // Context construction stages the default box before the reorder object exists.
        if (cu.getNonbondedUtilities().getAtomReorder() != nullptr)
            cu.unwrapDevicePositions();
        cu.setPeriodicBoxVectors(a, b, c);
        return;
    }
    vector<mm_int4>& offsets = cu.getPosCellOffsets();
    bool wrapped = any_of(offsets.begin(), offsets.end(), [] (const mm_int4& o) { return o.x != 0 || o.y != 0 || o.z != 0; });
    if (wrapped) {
        // Unwrap with the current box, then zero the offsets, so the common kernel below only sets the box.
        ContextSelector selector(cu);
        // The enabled unwrap kernel reads the old box directly from the device buffer.
        if (!cu.getUseDeviceBox()) {
            Vec3 box[3];
            cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
            if (cu.getUseMixedPrecision()) {
                vector<mm_double4> vectors;
                for (const Vec3& v : box)
                    vectors.push_back(mm_double4(v[0], v[1], v[2], 0));
                boxVectors.upload(vectors);
            }
            else {
                vector<mm_float4> vectors;
                for (const Vec3& v : box)
                    vectors.push_back(mm_float4((float) v[0], (float) v[1], (float) v[2], 0));
                boxVectors.upload(vectors);
            }
        }
        cellOffsets.upload(offsets);
        unwrapKernel->execute(cu.getNumAtoms());
        for (auto& offset : offsets)
            offset = mm_int4(0, 0, 0, 0);
    }
    CommonUpdateStateDataKernel::setPeriodicBoxVectors(context, a, b, c);
}

void MetalIntegrateLangevinMiddleStepKernel::initialize(const System& system, const LangevinMiddleIntegrator& integrator) {
    CommonIntegrateLangevinMiddleStepKernel::initialize(system, integrator);
    // Its kinetic energy is computeKineticEnergy(0), which a prefetch in the energy pass can answer.
    cu.getIntegrationUtilities().setPrefetchKineticEnergy(true);
}

void MetalIntegrateLangevinMiddleStepKernel::integrate(double tol) {
    if (!cu.getIntegrationUtilities().integrateLangevinMiddle(params, tol))
        CommonIntegrateLangevinMiddleStepKernel::integrate(tol);
}

// Threads per threadgroup in the pair kernels, which launch one SIMD group per block pair.
static const int PAIR_GROUP_SIZE = 256;

// The block-pair kernels win when the half-list launch leaves the GPU short of work, so that its tail after the last
// atomics is exposed, and lose when the launch already fills the GPU.  gbsa (78 blocks) has 1.27 block pairs per SIMD
// group of the launch on the M3 Ultra (60 cores), where they gain, and 7.6 on the M2 (10 cores), where they cost
// 12-14%.  The count per SIMD group comes from the core count, so this picks the path per device and system.  Those
// numbers predate Force1 adding with atomics, which gained another 3.7% single and 4.5% mixed on gbsa on the M3 Ultra
// by removing the kernel that reduced its per-block partials; the M2 side wasn't measured again.
static const double MAX_OWNED_BLOCK_PAIRS_PER_WARP = 2.0;

bool MetalCalcGBSAOBCForceKernel::createOwnedKernels() {
    MetalNonbondedUtilities& nb = cu.getNonbondedUtilities();
    if (!nb.getUseCutoff() || cu.getUseDoublePrecision())
        return false;

    int numBlocks = cu.getNumAtomBlocks();
    int paddedAtoms = cu.getPaddedNumAtoms();
    int threads = nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize();
    int numWarps = threads/32;
    if ((double) numBlocks*numBlocks > MAX_OWNED_BLOCK_PAIRS_PER_WARP*numWarps)
        return false;
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
    defines["DIELECTRIC_OFFSET"] = "0.009f";
    defines["OBC_ALPHA"] = "1.0f";
    defines["OBC_BETA"] = "0.8f";
    defines["OBC_GAMMA"] = "4.85f";
    defines["PROBE_RADIUS"] = "0.14f";
    ComputeProgram program = cu.compileProgram(MetalKernelSources::gbsaObcOwned, defines);
    ownedBornSumKernel = program->createKernel("computeBornSumOwned");
    ownedBornSumKernel->addArg(bornSum);
    ownedBornSumKernel->addArg(cu.getPosq());
    ownedBornSumKernel->addArg(params);
    for (int i = 0; i < 5; i++)
        ownedBornSumKernel->addArg(); // The periodic box size arguments are set when the kernel is executed.
    ownedBornSumKernel->addArg(nb.getBlockCenters());
    ownedBornSumKernel->addArg(nb.getBlockBoundingBoxes());
    ownedForce1Kernel = program->createKernel("computeGBSAForce1Owned");
    ownedForce1Kernel->addArg(cu.getLongForceBuffer());
    ownedForce1Kernel->addArg(bornForce);
    ownedForce1Kernel->addArg(cu.getEnergyBuffer());
    ownedForce1Kernel->addArg(cu.getPosq());
    ownedForce1Kernel->addArg(bornSum);
    ownedForce1Kernel->addArg(params);
    ownedForce1Kernel->addArg(charges);
    ownedForce1Kernel->addArg(); // Whether to include energy.
    for (int i = 0; i < 5; i++)
        ownedForce1Kernel->addArg(); // The periodic box size arguments are set when the kernel is executed.
    ownedForce1Kernel->addArg(nb.getBlockCenters());
    ownedForce1Kernel->addArg(nb.getBlockBoundingBoxes());
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
    int numBlocks = cu.getNumAtomBlocks();
    int pairThreads = min(threads, (numBlocks*(numBlocks+1)/2*32+PAIR_GROUP_SIZE-1)/PAIR_GROUP_SIZE*PAIR_GROUP_SIZE);
    ownedForce1Kernel->setArg(7, (int) includeEnergy);
    setPeriodicBoxArgs(cu, ownedBornSumKernel, 3);
    setPeriodicBoxArgs(cu, ownedForce1Kernel, 8);
    ownedBornSumKernel->execute(pairThreads, PAIR_GROUP_SIZE);
    ownedForce1Kernel->execute(pairThreads, PAIR_GROUP_SIZE);
    return 0.0;
}

void MetalApplyMonteCarloBarostatKernel::initialize(const System& system, const Force& barostat, int components, bool rigidMolecules) {
    CommonApplyMonteCarloBarostatKernel::initialize(system, barostat, components, rigidMolecules);
    const char* deviceSetting = getenv("OPENMM_METAL_DEVICE_BAROSTAT");
    deviceRequested = deviceSetting != nullptr && string(deviceSetting) == "1";
    rigidTrial = rigidMolecules;
    if (typeid(barostat) == typeid(MonteCarloBarostat))
        trialSeed = static_cast<const MonteCarloBarostat&>(barostat).getRandomNumberSeed();
    if (deviceRequested && trialSeed == 0)
        trialSeed = (int) std::random_device{}();
    const char* setting = getenv("OPENMM_METAL_DEFER_BAROSTAT_ENERGIES");
    deferEnergies = typeid(barostat) == typeid(MonteCarloBarostat) && cu.getNumContexts() == 1 && !cu.getPlatformData().useCpuPme;
    // These forces have no nested Context evaluation or host callback.  Other systems retain getState().
    for (int i = 0; i < system.getNumForces(); i++) {
        const type_info& type = typeid(system.getForce(i));
        if (type != typeid(HarmonicBondForce) && type != typeid(HarmonicAngleForce) &&
                type != typeid(PeriodicTorsionForce) && type != typeid(NonbondedForce) &&
                type != typeid(CMMotionRemover) && type != typeid(MonteCarloBarostat))
            deferEnergies = false;
    }
    deviceEligible = deferEnergies;
    // Neighbor list allocation and molecule tables are not ready until context initialization finishes.
    cu.deviceBarostatKernel = this;
    deferEnergies = deferEnergies && (setting == nullptr || string(setting) != "0");
}

bool MetalApplyMonteCarloBarostatKernel::computeTrialEnergies(ContextImpl& context, int groups,
        const function<void()>& propose, double& initialEnergy, double& finalEnergy) {
    // LangevinMiddle's kinetic energy is an unshifted, read-only reduction.  Omitting its unused
    // getState() result cannot change coordinates, forces, velocities, or random number consumption.
    // Custom and force-shifted integrators retain the complete synchronous getState() path.
    if (!deferEnergies || typeid(context.getIntegrator()) != typeid(LangevinMiddleIntegrator))
        return false;
    ContextSelector selector(cu);
    double initialForce, initialPlatform, finalForce, finalPlatform;
    cu.enqueueEnergy(initialEnergySum, groups, initialForce, initialPlatform);
    propose();
    cu.enqueueEnergy(finalEnergySum, groups, finalForce, finalPlatform);
    // The first download waits for both reductions.  Keep finishComputation's addition inside
    // ContextImpl's addition: hostForce + (hostPlatform + deviceReduction), never reassociate it.
    initialEnergy = initialForce+(initialPlatform+cu.readEnergy(initialEnergySum));
    finalEnergy = finalForce+(finalPlatform+cu.readEnergy(finalEnergySum));
    testInitialEnergy = initialEnergy;
    testFinalEnergy = finalEnergy;
    return true;
}

bool MetalApplyMonteCarloBarostatKernel::getTrialUniforms(double& proposal, double& acceptance) {
    proposal = testProposal;
    acceptance = testAcceptance;
    return injectedUniforms;
}

void MetalApplyMonteCarloBarostatKernel::setTrialUniforms(double proposal, double acceptance) {
    if (proposal < 0 || proposal >= 1 || acceptance < 0 || acceptance >= 1)
        throw OpenMMException("Barostat test uniforms must be in [0, 1)");
    injectedUniforms = true;
    testProposal = proposal;
    testAcceptance = acceptance;
}

vector<double> MetalApplyMonteCarloBarostatKernel::getTrialState() {
    vector<double> result;
    if (trialState.isInitialized())
        trialState.download(result);
    return result;
}

void MetalApplyMonteCarloBarostatKernel::readTrialEnergies(double& initial, double& final) {
    initial = testInitialEnergy+cu.readEnergy(initialEnergySum);
    final = testFinalEnergy+cu.readEnergy(finalEnergySum);
}

bool MetalApplyMonteCarloBarostatKernel::prepareDeviceBarostat() {
    cu.deviceBarostatState = nullptr;
    if (!deviceRequested)
        return false;
    string reason;
    if (!deviceEligible)
        reason = "requires an isotropic FAH force set, one context and GPU PME";
    else if (!cu.getUseDeviceBox())
        reason = "requires OPENMM_METAL_DEVICE_BOX=1 and an eligible device box";
    else if (typeid(cu.getContextImpl()->getIntegrator()) != typeid(LangevinMiddleIntegrator))
        reason = "requires LangevinMiddleIntegrator";
    else if (!rigidTrial)
        reason = "requires rigid molecule scaling";
    else if (!cu.getNonbondedUtilities().getAtomReorder())
        reason = "requires periodic GPU atom reordering";
    else if (!cu.getNonbondedUtilities().hasBoundedNeighborList())
        reason = "neighbor list requires host overflow checks";
    else if (!cu.getNonbondedUtilities().getAtomReorder()->hasUsableTables())
        reason = "reorder tables do not support this molecule layout";
    if (!reason.empty()) {
        if (!reportedFallback)
            cerr << "OpenMM Metal: device barostat disabled: " << reason << endl;
        reportedFallback = true;
        return false;
    }
    ContextSelector selector(cu);
    if (!trialState.isInitialized()) {
        trialState.initialize(cu, 16, sizeof(double), "deviceBarostatState");
        vector<double> initial(16, 0);
        Vec3 box[3];
        cu.getContextImpl()->getSystem().getDefaultPeriodicBoxVectors(box[0], box[1], box[2]);
        initial[0] = 0.01*box[0][0]*box[1][1]*box[2][2];
        initial[12] = (unsigned int) trialSeed;
        trialState.upload(initial);
    }
    cu.deviceBarostatState = &trialState;
    return true;
}

bool MetalApplyMonteCarloBarostatKernel::executeTrial(ContextImpl& context, int groups, double pressure, double kT) {
    if (!prepareDeviceBarostat())
        return false;
    ContextSelector selector(cu);
    if (!proposeTrial) {
        scaleFactor.initialize(cu, 1, sizeof(double), "barostatScale");
        map<string, string> defines;
        defines["DEVICE_BOX_ENERGY"] = "1"; // Enable df64 in single mode as well.
        ComputeProgram program = cu.compileProgram(MetalKernelSources::deviceBarostat, defines);
        proposeTrial = program->createKernel("proposeBarostat");
        decideTrial = program->createKernel("decideBarostat");
        restoreTrial = program->createKernel("restoreBarostat");
        for (int i = 0; i < 5; i++) proposeTrial->addArg();
        for (int i = 0; i < 11; i++) decideTrial->addArg();
        for (int i = 0; i < 4; i++) restoreTrial->addArg();
    }
    // Energy-only evaluations do not reorder atoms.  Save the complete device transaction anyway,
    // including offsets and index, so rejection preserves bits without host observation/listeners.
    cu.getIntegrationUtilities().ensureVirtualSiteForcesDistributed();
    cu.getNonbondedUtilities().scatterRoleForces();
    vector<ArrayInterface*> arrays = {&cu.getPosq(), &cu.getVelm(), &cu.getLongForceBuffer(),
                                    &cu.getAtomIndexArray()};
    if (cu.getUseMixedPrecision()) arrays.push_back(&cu.getPosqCorrection());
    vector<MetalArray*> targets;
    for (auto array : arrays) targets.push_back(&cu.unwrap(*array));
    targets.push_back(&cu.getDeviceCellOffsets());
    targets.push_back(&cu.getDeviceBoxState());
    for (int i = 0; i < targets.size(); i++) {
        if (i == trialSaved.size())
            trialSaved.emplace_back(new MetalArray(cu, targets[i]->getSize(), targets[i]->getElementSize(), "barostatSaved"));
        targets[i]->copyTo(*trialSaved[i]);
    }
    proposeTrial->setArg(0, trialState);
    proposeTrial->setArg(1, cu.getDeviceBoxState());
    proposeTrial->setArg(2, scaleFactor);
    proposeTrial->setArg(3, (int) injectedUniforms);
    proposeTrial->setArg(4, testProposal);
    proposeTrial->execute(1, 1);
    double initialForce, initialPlatform, finalForce, finalPlatform;
    cu.enqueueEnergy(initialEnergySum, groups, initialForce, initialPlatform);
    cu.scaleDeviceBox(scaleFactor);
    cu.enqueueEnergy(finalEnergySum, groups, finalForce, finalPlatform);
    testInitialEnergy = initialForce+initialPlatform;
    testFinalEnergy = finalForce+finalPlatform;
    decideTrial->setArg(0, trialState);
    decideTrial->setArg(1, initialEnergySum);
    decideTrial->setArg(2, finalEnergySum);
    decideTrial->setArg(3, (int) initialEnergySum.getSize());
    decideTrial->setArg(4, initialForce+initialPlatform);
    decideTrial->setArg(5, finalForce+finalPlatform);
    decideTrial->setArg(6, pressure);
    decideTrial->setArg(7, kT);
    decideTrial->setArg(8, (int) context.getMolecules().size());
    decideTrial->setArg(9, (int) injectedUniforms);
    decideTrial->setArg(10, testAcceptance);
    decideTrial->execute(1, 1);
    restoreTrial->setArg(0, trialState);
    for (int i = 0; i < targets.size(); i++) {
        int words = targets[i]->getSize()*targets[i]->getElementSize()/sizeof(unsigned int);
        restoreTrial->setArg(1, *trialSaved[i]);
        restoreTrial->setArg(2, *targets[i]);
        restoreTrial->setArg(3, words);
        restoreTrial->execute(words);
    }
    cu.deviceCellOffsetsChanged();
    cu.getNonbondedUtilities().invalidateNeighborList();
    cu.setForcesValid(false);
    return true;
}

void MetalUpdateStateDataKernel::createCheckpoint(ContextImpl& context, ostream& stream) {
    if (cu.deviceBarostatKernel != nullptr)
        cu.deviceBarostatKernel->prepareDeviceBarostat();
    if (cu.deviceBarostatState != nullptr) {
        const int version = 0xDBA0001;
        stream.write((const char*) &version, sizeof(version));
    }
    CommonUpdateStateDataKernel::createCheckpoint(context, stream);
    if (cu.deviceBarostatState != nullptr) {
        vector<double> state;
        cu.deviceBarostatState->download(state);
        stream.write((char*) state.data(), state.size()*sizeof(double));
        // Save the raw device box: its derived fields come from device df64 scaling, which
        // setPeriodicBoxVectors() cannot reproduce, so continuation would not be bitwise.
        vector<unsigned char> box(cu.getDeviceBoxState().getSize());
        cu.getDeviceBoxState().download(box.data());
        stream.write((char*) box.data(), box.size());
    }
}

void MetalUpdateStateDataKernel::loadCheckpoint(ContextImpl& context, istream& stream) {
    if (cu.deviceBarostatKernel != nullptr)
        cu.deviceBarostatKernel->prepareDeviceBarostat();
    const int deviceVersion = 0xDBA0001;
    // The marker's first byte differs from Common's version on either byte order.
    // Peek so ordinary checkpoints also work with non-seekable input streams.
    if (cu.deviceBarostatState == nullptr && stream.peek() == *(const unsigned char*) &deviceVersion)
        throw OpenMMException("Checkpoint requires matching OPENMM_METAL_DEVICE_BAROSTAT and device eligibility");
    if (cu.deviceBarostatState != nullptr) {
        int version = 0;
        stream.read((char*) &version, sizeof(version));
        if (version != deviceVersion)
            throw OpenMMException("Checkpoint requires matching OPENMM_METAL_DEVICE_BAROSTAT and device eligibility");
    }
    CommonUpdateStateDataKernel::loadCheckpoint(context, stream);
    if (cu.deviceBarostatState != nullptr) {
        vector<double> state(16);
        stream.read((char*) state.data(), state.size()*sizeof(double));
        vector<unsigned char> box(cu.getDeviceBoxState().getSize());
        stream.read((char*) box.data(), box.size());
        if (!stream) throw OpenMMException("Missing device barostat checkpoint state (flags must match)");
        cu.deviceBarostatState->upload(state);
        cu.restoreDeviceBox(box.data());
    }
}
