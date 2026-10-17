/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2026 Stanford University and the Authors.      *
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

#include "MetalKernels.h"
#include "MetalQueue.h"
#include "CommonKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/CMAPTorsionForce.h"
#include "openmm/CustomAngleForce.h"
#include "openmm/CustomBondForce.h"
#include "openmm/CustomCompoundBondForce.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/CustomGBForce.h"
#include "openmm/CustomNonbondedForce.h"
#include "openmm/CustomTorsionForce.h"
#include "openmm/GBSAOBCForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/MonteCarloBarostat.h"
#include "openmm/NonbondedForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/RBTorsionForce.h"
#include <typeinfo>

using namespace OpenMM;
using namespace std;

void MetalCalcForcesAndEnergyKernel::initialize(const System& system) {
}

void MetalCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    mc.setForcesValid(true);
    mc.clearAutoclearBuffers();
    mc.updateGlobalParamValues();
    for (auto computation : mc.getPreComputations())
        computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    MetalNonbondedUtilities& nb = mc.getNonbondedUtilities();
    mc.setComputeForceCount(mc.getComputeForceCount()+1);
    nb.prepareInteractions(groups);
    map<string, double>& derivs = mc.getEnergyParamDerivWorkspace();
    for (auto& param : context.getParameters())
        derivs[param.first] = 0;
}

double MetalCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups, bool& valid) {
    mc.getBondedUtilities().computeInteractions(groups);
    mc.getNonbondedUtilities().computeInteractions(groups, includeForces, includeEnergy);
    double sum = 0.0;
    for (auto computation : mc.getPostComputations())
        sum += computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    mc.getIntegrationUtilities().distributeForcesFromVirtualSites();
    if (includeEnergy)
        sum += mc.reduceEnergy();
    if (!mc.getForcesValid())
        valid = false;
    mc.flushQueue();
    return sum;
}

void MetalUpdateStateDataKernel::setPositions(ContextImpl& context, const std::vector<Vec3>& positions) {
    CommonUpdateStateDataKernel::setPositions(context, positions);
    MetalQueue& queue = dynamic_cast<MetalQueue&>(*cc.getCurrentQueue());
    queue.flush(true);
}

void MetalUpdateStateDataKernel::setVelocities(ContextImpl& context, const std::vector<Vec3>& velocities) {
    CommonUpdateStateDataKernel::setVelocities(context, velocities);
    MetalQueue& queue = dynamic_cast<MetalQueue&>(*cc.getCurrentQueue());
    queue.flush(true);
}

// Marks the block of velm pair words that mixed precision checkpoints start with.  The common checkpoint starts
// with its version, 3, so a checkpoint without the block (written before it existed) still loads.
static const int METAL_PAIR_BLOCK = 0x4d505231;

void MetalUpdateStateDataKernel::createCheckpoint(ContextImpl& context, ostream& stream) {
    MetalArray& velm = dynamic_cast<MetalContext&>(cc).unwrap(cc.getVelm());
    if (velm.getPairStorage()) {
        ContextSelector selector(cc);
        stream.write((char*) &METAL_PAIR_BLOCK, sizeof(int));
        vector<char> stored(velm.getSize()*velm.getElementSize());
        velm.downloadStored(stored.data());
        stream.write(stored.data(), stored.size());
        stream.write((char*) cc.getAtomIndex().data(), sizeof(int)*cc.getAtomIndex().size());
    }
    CommonUpdateStateDataKernel::createCheckpoint(context, stream);
}

void MetalUpdateStateDataKernel::loadCheckpoint(ContextImpl& context, istream& stream) {
    MetalArray& velm = dynamic_cast<MetalContext&>(cc).unwrap(cc.getVelm());
    vector<char> stored;
    vector<int> storedIndex;
    if (velm.getPairStorage()) {
        int marker;
        stream.read((char*) &marker, sizeof(int));
        if (marker == METAL_PAIR_BLOCK) {
            stored.resize(velm.getSize()*velm.getElementSize());
            stream.read(stored.data(), stored.size());
            storedIndex.resize(cc.getAtomIndex().size());
            stream.read((char*) storedIndex.data(), sizeof(int)*storedIndex.size());
        }
        else if (!stream.seekg(-(streamoff) sizeof(int), ios::cur))
            throw OpenMMException("loadCheckpoint: can't rewind the stream to read a checkpoint without velocity pairs");
    }
    CommonUpdateStateDataKernel::loadCheckpoint(context, stream);
    if (stored.empty())
        return;

    // The common load may have reset the atom order, so move each atom's pair words to where the atom is now.

    ContextSelector selector(cc);
    const vector<int>& atomIndex = cc.getAtomIndex();
    if (atomIndex != storedIndex) {
        int elementSize = velm.getElementSize();
        vector<int> storedPosition(atomIndex.size());
        for (int i = 0; i < (int) storedIndex.size(); i++)
            storedPosition[storedIndex[i]] = i;
        vector<char> moved(stored);
        for (int i = 0; i < (int) atomIndex.size(); i++)
            memcpy(&moved[i*elementSize], &stored[storedPosition[atomIndex[i]]*elementSize], elementSize);
        stored.swap(moved);
    }
    velm.uploadStored(stored.data());
}

void MetalCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
    // With LJPME, spreading the dispersion grid in the order sorted for the charge grid costs less than sorting the
    // atoms again (and the charge grid sort can then skip every other step, as it does without LJPME).
    commonInitialize(system, force, !mc.getPlatformData().useCpuPme, false, mc.getPlatformData().deterministicForces, mc.getPlatformData().useCpuPme, false);
}

void MetalCalcConstantPotentialForceKernel::initialize(const System& system, const ConstantPotentialForce& force) {
    commonInitialize(system, force, false, mc.getPlatformData().deterministicForces);
}

void MetalIntegrateLangevinMiddleStepKernel::integrate(double tol) {
    if (!mc.getIntegrationUtilities().integrateLangevinMiddle(params, tol))
        CommonIntegrateLangevinMiddleStepKernel::integrate(tol);
}

/**
 * Get whether a force leaves the velocities alone in updateContextState() and calcForcesAndEnergy().  Unknown forces,
 * thermostats, other barostats and forces that evaluate an inner Context count as touching them.
 *
 * MonteCarloBarostat counts as leaving them alone.  Its trial reads only potential energies, and a rejected trial
 * restores velm whole, so the velocities the integrator sees are the same whether the center of mass motion comes off
 * before the trial or inside the integrator.  openmm.app adds the barostat after CMMotionRemover, so without this every
 * system with one would pay the extra dispatch.  The anisotropic, flexible and membrane barostats stay out: they can
 * read the kinetic energy.
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
    ContextSelector selector(mc);
    // Forces before the remover run their calcForcesAndEnergy() after it too, so every other force has to qualify.
    otherForcesIgnoreVelocities = true;
    for (int i = 0; i < system.getNumForces(); i++)
        if (&system.getForce(i) != &force)
            otherForcesIgnoreVelocities &= ignoresVelocities(system.getForce(i));
    int numAtoms = mc.getNumAtoms();
    cmMomentum.initialize<mm_float4>(mc, mc.getPaddedNumAtoms(), "cmMomentum");
    double totalMass = 0.0;
    for (int i = 0; i < numAtoms; i++)
        totalMass += system.getParticleMass(i);
    map<string, string> defines;
    defines["INVERSE_TOTAL_MASS"] = mc.doubleToString(totalMass == 0 ? 0.0 : 1.0/totalMass);
    ComputeProgram program = mc.compileProgram(CommonKernelSources::removeCM, defines);
    kernel1 = program->createKernel("calcCenterOfMassMomentum");
    kernel1->addArg(numAtoms);
    kernel1->addArg(mc.getVelm());
    kernel1->addArg(cmMomentum);
    kernel2 = program->createKernel("removeCenterOfMassMomentum");
    kernel2->addArg(numAtoms);
    kernel2->addArg(mc.getVelm());
    kernel2->addArg(cmMomentum);
}

void MetalRemoveCMMotionKernel::execute(ContextImpl& context) {
    ContextSelector selector(mc);
    int numAtoms = mc.getNumAtoms();
    kernel1->execute(numAtoms, 64);
    MetalIntegrationUtilities& integration = mc.getIntegrationUtilities();
    if (otherForcesIgnoreVelocities && integration.canRemoveCenterOfMassMomentum() &&
            dynamic_cast<LangevinMiddleIntegrator*>(&context.getIntegrator()) != NULL) {
        // calcCenterOfMassMomentum wrote one partial momentum per thread group.
        int numGroups = min((numAtoms+63)/64, mc.getNumThreadBlocks());
        integration.removeCenterOfMassMomentum(cmMomentum, numGroups);
        return;
    }
    kernel2->execute(numAtoms, 64);
}
