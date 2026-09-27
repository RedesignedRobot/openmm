/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2025 Stanford University and the Authors.      *
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

#include "MetalIntegrationUtilities.h"
#include "MetalContext.h"
#include "MetalKernelSources.h"
#include "CommonKernelSources.h"
#include "openmm/common/ContextSelector.h"

using namespace OpenMM;
using namespace std;

MetalIntegrationUtilities::MetalIntegrationUtilities(MetalContext& context, const System& system) : IntegrationUtilities(context, system),
        ccmaConvergedMemory(NULL), totalMass(0.0), pendingCMMomentum(NULL), pendingCMGroups(0) {
        for (int i = 0; i < system.getNumParticles(); i++)
            totalMass += system.getParticleMass(i);
        ccmaEvent = context.createEvent();
        ccmaConvergedDeviceMemory = context.getDevice()->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
        ccmaConvergedMemory = (int*) ccmaConvergedDeviceMemory->contents();
}

MetalIntegrationUtilities::~MetalIntegrationUtilities() {
    ContextSelector selector(context);
    if (ccmaConvergedMemory != NULL) {
        ccmaConvergedDeviceMemory->release();
    }
}

MetalArray& MetalIntegrationUtilities::getPosDelta() {
    return dynamic_cast<MetalContext&>(context).unwrap(posDelta);
}

MetalArray& MetalIntegrationUtilities::getRandom() {
    return dynamic_cast<MetalContext&>(context).unwrap(random);
}

MetalArray& MetalIntegrationUtilities::getStepSize() {
    return dynamic_cast<MetalContext&>(context).unwrap(stepSize);
}

void MetalIntegrationUtilities::applyConstraintsImpl(bool constrainVelocities, double tol) {
    ContextSelector selector(context);
    ComputeKernel settleKernel, shakeKernel, ccmaForceKernel;
    if (constrainVelocities) {
        settleKernel = settleVelKernel;
        shakeKernel = shakeVelKernel;
        ccmaForceKernel = ccmaVelForceKernel;
    }
    else {
        settleKernel = settlePosKernel;
        shakeKernel = shakePosKernel;
        ccmaForceKernel = ccmaPosForceKernel;
    }
    if (settleAtoms.isInitialized()) {
        if (context.getUseDoublePrecision() || context.getUseMixedPrecision())
            settleKernel->setArg(1, tol);
        else
            settleKernel->setArg(1, (float) tol);
        settleKernel->execute(settleAtoms.getSize());
    }
    if (shakeAtoms.isInitialized()) {
        if (context.getUseDoublePrecision() || context.getUseMixedPrecision())
            shakeKernel->setArg(1, tol);
        else
            shakeKernel->setArg(1, (float) tol);
        shakeKernel->execute(shakeAtoms.getSize());
    }
    if (ccmaConstraintAtoms.isInitialized()) {
        if (ccmaConstraintAtoms.getSize() <= 1024) {
            // Use the version of CCMA that runs in a single kernel with one workgroup.
            ccmaFullKernel->setArg(0, (int) constrainVelocities);
            if (context.getUseDoublePrecision() || context.getUseMixedPrecision())
                ccmaFullKernel->setArg(14, tol);
            else
                ccmaFullKernel->setArg(14, (float) tol);
            ccmaFullKernel->execute(128, 128);
        }
        else {
            ccmaForceKernel->setArg(6, ccmaConvergedDeviceMemory);
            if (context.getUseDoublePrecision() || context.getUseMixedPrecision())
                ccmaForceKernel->setArg(7, tol);
            else
                ccmaForceKernel->setArg(7, (float) tol);
            ccmaDirectionsKernel->execute(ccmaConstraintAtoms.getSize());
            const int checkInterval = 4;
            ccmaConvergedMemory[0] = 0;
            ccmaUpdateKernel->setArg(4, constrainVelocities ? context.getVelm() : posDelta);
            for (int i = 0; i < 150; i++) {
                ccmaForceKernel->setArg(8, i);
                ccmaForceKernel->execute(ccmaConstraintAtoms.getSize());
                if ((i+1)%checkInterval == 0)
                    ccmaEvent->enqueue();
                ccmaMultiplyKernel->setArg(5, i);
                ccmaMultiplyKernel->execute(ccmaConstraintAtoms.getSize());
                ccmaUpdateKernel->setArg(9, i);
                ccmaUpdateKernel->execute(context.getNumAtoms());
                if ((i+1)%checkInterval == 0) {
                    ccmaEvent->wait();
                    if (ccmaConvergedMemory[0])
                        break;
                }
            }
        }
    }
}

void MetalIntegrationUtilities::distributeForcesFromVirtualSites() {
    ContextSelector selector(context);
    if (numVsites > 0) {
        Vec3 boxVectors[3];
        context.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        mm_double4 recipBoxVectorsDouble[3];
        context.computeReciprocalBoxVectors(recipBoxVectorsDouble);
        if (context.getUseDoublePrecision()) {
            mm_double4 boxVectorsDouble[3];
            for (int i = 0; i < 3; i++)
                boxVectorsDouble[i] = mm_double4(boxVectors[i][0], boxVectors[i][1], boxVectors[i][2], 0);
            vsiteForceKernel->setArg(18, boxVectorsDouble[0]);
            vsiteForceKernel->setArg(19, boxVectorsDouble[1]);
            vsiteForceKernel->setArg(20, boxVectorsDouble[2]);
            vsiteForceKernel->setArg(21, recipBoxVectorsDouble[0]);
            vsiteForceKernel->setArg(22, recipBoxVectorsDouble[1]);
            vsiteForceKernel->setArg(23, recipBoxVectorsDouble[2]);
        }
        else {
            mm_float4 boxVectorsFloat[3], recipBoxVectorsFloat[3];
            for (int i = 0; i < 3; i++) {
                boxVectorsFloat[i] = mm_float4((float) boxVectors[i][0], (float) boxVectors[i][1], (float) boxVectors[i][2], 0);
                recipBoxVectorsFloat[i] = mm_float4((float) recipBoxVectorsDouble[i].x, (float) recipBoxVectorsDouble[i].y, (float) recipBoxVectorsDouble[i].z, 0);
            }
            vsiteForceKernel->setArg(18, boxVectorsFloat[0]);
            vsiteForceKernel->setArg(19, boxVectorsFloat[1]);
            vsiteForceKernel->setArg(20, boxVectorsFloat[2]);
            vsiteForceKernel->setArg(21, recipBoxVectorsFloat[0]);
            vsiteForceKernel->setArg(22, recipBoxVectorsFloat[1]);
            vsiteForceKernel->setArg(23, recipBoxVectorsFloat[2]);
        }
        for (int i = numVsiteStages-1; i >= 0; i--) {
            vsiteForceKernel->setArg(2, context.getLongForceBuffer());
            vsiteForceKernel->setArg(25, i);
            vsiteForceKernel->execute(numVsites);
        }
    }
}

bool MetalIntegrationUtilities::canRemoveCenterOfMassMomentum() const {
    return !ccmaConstraintAtoms.isInitialized() && !context.getUseDoublePrecision() && !context.getUseMixedPrecision();
}

void MetalIntegrationUtilities::removeCenterOfMassMomentum(ComputeKernel& calcKernel, ComputeArray& cmMomentum, int numGroups) {
    pendingCMCalc = calcKernel;
    pendingCMMomentum = &cmMomentum;
    pendingCMGroups = numGroups;
}

void MetalIntegrationUtilities::calcPendingCenterOfMassMomentum() {
    if (!pendingCMCalc)
        return;
    pendingCMCalc->execute(context.getNumAtoms(), 64);
    pendingCMCalc.reset();
}

bool MetalIntegrationUtilities::integrateLangevinMiddle(ComputeArray& params, double tol) {
    if (ccmaConstraintAtoms.isInitialized() || context.getUseDoublePrecision())
        return false;
    ContextSelector selector(context);
    if (!langevinMiddleKernel) {
        // Each thread handles one unit: a SETTLE cluster, a SHAKE cluster, or an atom in neither.

        int numAtoms = context.getNumAtoms();
        vector<bool> isConstrained(numAtoms, false);
        int numSettle = 0, numShake = 0;
        if (settleAtoms.isInitialized()) {
            vector<mm_int4> atoms;
            settleAtoms.download(atoms);
            numSettle = atoms.size();
            for (mm_int4 cluster : atoms)
                isConstrained[cluster.x] = isConstrained[cluster.y] = isConstrained[cluster.z] = true;
        }
        if (shakeAtoms.isInitialized()) {
            vector<mm_int4> atoms;
            shakeAtoms.download(atoms);
            numShake = atoms.size();
            for (mm_int4 cluster : atoms)
                for (int atom : {cluster.x, cluster.y, cluster.z, cluster.w})
                    if (atom != -1)
                        isConstrained[atom] = true;
        }
        vector<int> unconstrained;
        for (int i = 0; i < numAtoms; i++)
            if (!isConstrained[i])
                unconstrained.push_back(i);
        if (unconstrained.size() > 0) {
            freeAtoms.initialize<int>(context, unconstrained.size(), "freeAtoms");
            freeAtoms.upload(unconstrained);
        }
        // Positions and velocities that overflow are lost either way, so this program drops df64's overflow guards.
        map<string, string> fusedDefines = defines;
        fusedDefines["DF64_FINITE"] = "";
        fusedDefines["INVERSE_TOTAL_MASS"] = context.doubleToString(totalMass == 0 ? 0.0 : 1.0/totalMass);
        ComputeProgram program = context.compileProgram(CommonKernelSources::integrationUtilities+MetalKernelSources::fusedLangevinMiddle, fusedDefines);
        numLangevinMiddleUnits = numSettle+numShake+unconstrained.size();
        langevinMiddleKernel = program->createKernel("integrateLangevinMiddleFused");
        langevinMiddleKernel->addArg(numSettle);
        langevinMiddleKernel->addArg(numShake);
        langevinMiddleKernel->addArg(numLangevinMiddleUnits);
        langevinMiddleKernel->addArg(context.getPaddedNumAtoms());
        langevinMiddleKernel->addArg(); // The tolerance will be set just before it is executed.
        langevinMiddleKernel->addArg(context.getPosq());
        if (context.getUseMixedPrecision())
            langevinMiddleKernel->addArg(context.getPosqCorrection());
        else
            langevinMiddleKernel->addArg(nullptr);
        langevinMiddleKernel->addArg(context.getVelm());
        langevinMiddleKernel->addArg(context.getLongForceBuffer());
        langevinMiddleKernel->addArg(); // The parameters will be set just before it is executed.
        langevinMiddleKernel->addArg(stepSize);
        langevinMiddleKernel->addArg(random);
        langevinMiddleKernel->addArg(); // The random index will be set just before it is executed.
        for (ComputeArray* array : {&settleAtoms, &settleParams, &shakeAtoms, &shakeParams, &freeAtoms}) {
            if (array->isInitialized())
                langevinMiddleKernel->addArg(*array);
            else
                langevinMiddleKernel->addArg(nullptr);
        }
        langevinMiddleKernel->addArg(0); // The center of mass removal will be set just before it is executed.
        langevinMiddleKernel->addArg(nullptr);
    }
    if (context.getUseMixedPrecision())
        langevinMiddleKernel->setArg(4, tol);
    else
        langevinMiddleKernel->setArg(4, (float) tol);
    langevinMiddleKernel->setArg(9, params);
    langevinMiddleKernel->setArg(12, prepareRandomNumbers(context.getPaddedNumAtoms()));
    calcPendingCenterOfMassMomentum();
    langevinMiddleKernel->setArg(18, pendingCMGroups);
    if (pendingCMMomentum != NULL)
        langevinMiddleKernel->setArg(19, *pendingCMMomentum);
    langevinMiddleKernel->execute(numLangevinMiddleUnits);
    pendingCMMomentum = NULL;
    pendingCMGroups = 0;
    return true;
}
