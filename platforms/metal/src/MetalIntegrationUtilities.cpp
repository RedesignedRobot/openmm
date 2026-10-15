/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2026 Stanford University and the Authors.      *
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

#include "MetalIntegrationUtilities.h"
#include "MetalContext.h"
#include "MetalKernelSources.h"
#include "CommonKernelSources.h"
#include "openmm/common/ContextSelector.h"

using namespace OpenMM;
using namespace std;

MetalIntegrationUtilities::MetalIntegrationUtilities(MetalContext& context, const System& system) : IntegrationUtilities(context, system),
        totalMass(0.0), pendingCMMomentum(NULL), pendingCMGroups(0) {
        ccmaConvergedBuffer.initialize<int>(context, 1, "ccmaConvergedBuffer");
        for (int i = 0; i < system.getNumParticles(); i++)
            totalMass += system.getParticleMass(i);
        // posDelta shares kernels and raw copies with velm, so it stores float pairs too.  The zeros the base class
        // uploaded read the same either way.
        context.unwrap(posDelta).setPairStorage(context.getUseMixedPrecision());
}

MetalIntegrationUtilities::~MetalIntegrationUtilities() {
}

void MetalIntegrationUtilities::applyConstraintsImpl(bool constrainVelocities, double tol) {
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
        if (context.getUseMixedPrecision())
            settleKernel->setArg(1, tol);
        else
            settleKernel->setArg(1, (float) tol);
        settleKernel->execute(settleAtoms.getSize());
    }
    if (shakeAtoms.isInitialized()) {
        if (context.getUseMixedPrecision())
            shakeKernel->setArg(1, tol);
        else
            shakeKernel->setArg(1, (float) tol);
        shakeKernel->execute(shakeAtoms.getSize());
    }
    if (ccmaConstraintAtoms.isInitialized()) {
        if (ccmaConstraintAtoms.getSize() <= 1024) {
            // Use the version of CCMA that runs in a single kernel with one workgroup.
            ccmaFullKernel->setArg(0, (int) constrainVelocities);
            if (context.getUseMixedPrecision())
                ccmaFullKernel->setArg(14, tol);
            else
                ccmaFullKernel->setArg(14, (float) tol);
            ccmaFullKernel->execute(128, 128);
        }
        else {
            ccmaForceKernel->setArg(6, ccmaConvergedBuffer);
            if (context.getUseMixedPrecision())
                ccmaForceKernel->setArg(7, tol);
            else
                ccmaForceKernel->setArg(7, (float) tol);
            ccmaDirectionsKernel->execute(ccmaConstraintAtoms.getSize());
            const int checkInterval = 4;
            int ccmaConvergedMemory[] = {0, 0};
            ccmaUpdateKernel->setArg(4, constrainVelocities ? context.getVelm() : posDelta);
            for (int i = 0; i < 150; i++) {
                ccmaForceKernel->setArg(8, i);
                ccmaForceKernel->execute(ccmaConstraintAtoms.getSize());
                ccmaMultiplyKernel->setArg(5, i);
                ccmaMultiplyKernel->execute(ccmaConstraintAtoms.getSize());
                ccmaUpdateKernel->setArg(9, i);
                ccmaUpdateKernel->execute(context.getNumAtoms());
                if ((i+1)%checkInterval == 0) {
                    ccmaConverged.download(ccmaConvergedMemory);
                    if (ccmaConvergedMemory[i%2])
                        break;
                }
            }
        }
    }
}

void MetalIntegrationUtilities::distributeForcesFromVirtualSites() {
    if (numVsites > 0) {
        Vec3 boxVectors[3];
        context.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        mm_double4 recipBoxVectorsDouble[3];
        context.computeReciprocalBoxVectors(recipBoxVectorsDouble);
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

bool MetalIntegrationUtilities::canRemoveCenterOfMassMomentum() const {
    return context.getUseMixedPrecision() && !ccmaConstraintAtoms.isInitialized();
}

void MetalIntegrationUtilities::removeCenterOfMassMomentum(ComputeArray& cmMomentum, int numGroups) {
    pendingCMMomentum = &cmMomentum;
    pendingCMGroups = numGroups;
}

bool MetalIntegrationUtilities::integrateLangevinMiddle(ComputeArray& params, double tol) {
    if (!context.getUseMixedPrecision() || ccmaConstraintAtoms.isInitialized())
        return false;
    ContextSelector selector(context);
    if (!langevinMiddleKernel) {
        // Each unit is a SETTLE cluster, a SHAKE cluster, or free atoms: one per thread, or four per quad of threads.

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
        int numFree = unconstrained.size();
        int numFreeQuads = (numFree+3)/4;
        if (numFree > 0) {
            unconstrained.resize(4*numFreeQuads, -1);
            freeAtoms.initialize<int>(context, unconstrained.size(), "freeAtoms");
            freeAtoms.upload(unconstrained);
        }

        // The kernel is latency bound: a thread's chain of dependent operations sets its time, not the thread count.
        // A quad of threads per unit runs the per-atom work four wide, at the cost of running the constraint code in
        // all four lanes.  That pays only while all the quads fit in one pass of the grid execute() launches.  Past
        // that, each quad runs several units in sequence and pays the redundant constraint work in full.
        int numQuadUnits = numSettle+numShake+numFreeQuads;
        bool useQuads = (4*numQuadUnits <= context.getNumThreadBlocks()*MetalContext::ThreadBlockSize);

        // Positions and velocities that overflow are lost either way, so this program drops df64's overflow guards.
        map<string, string> fusedDefines = defines;
        fusedDefines["DF64_FINITE"] = "";
        // The same constant removeCM.cc gets, so the center of mass velocity comes out the same.
        fusedDefines["INVERSE_TOTAL_MASS"] = context.doubleToString(totalMass == 0 ? 0.0 : 1.0/totalMass);
        ComputeProgram program = context.compileProgram(CommonKernelSources::integrationUtilities+MetalKernelSources::fusedLangevinMiddle, fusedDefines);
        int numUnits = (useQuads ? numQuadUnits : numSettle+numShake+numFree);
        numLangevinMiddleThreads = (useQuads ? 4*numUnits : numUnits);
        langevinMiddleKernel = program->createKernel(useQuads ? "integrateLangevinMiddleFusedQuads" : "integrateLangevinMiddleFused");
        langevinMiddleKernel->addArg(numSettle);
        langevinMiddleKernel->addArg(numShake);
        langevinMiddleKernel->addArg(numUnits);
        langevinMiddleKernel->addArg(context.getPaddedNumAtoms());
        langevinMiddleKernel->addArg(tol);
        langevinMiddleKernel->addArg(context.getPosq());
        langevinMiddleKernel->addArg(context.getPosqCorrection());
        langevinMiddleKernel->addArg(context.getVelm());
        langevinMiddleKernel->addArg(context.getLongForceBuffer());
        langevinMiddleKernel->addArg(params);
        langevinMiddleKernel->addArg(stepSize);
        langevinMiddleKernel->addArg(random);
        langevinMiddleKernel->addArg(); // The random index will be set just before it is executed.
        for (ComputeArray* array : {&settleAtoms, &settleParams, &shakeAtoms, &shakeParams}) {
            if (array->isInitialized())
                langevinMiddleKernel->addArg(*array);
            else
                langevinMiddleKernel->addArg(nullptr);
        }
        if (freeAtoms.isInitialized())
            langevinMiddleKernel->addArg(freeAtoms);
        else
            langevinMiddleKernel->addArg(nullptr);
        langevinMiddleKernel->addArg(0); // The center of mass removal will be set just before it is executed.
        langevinMiddleKernel->addArg(nullptr);
    }
    langevinMiddleKernel->setArg(4, tol);
    langevinMiddleKernel->setArg(9, params);
    langevinMiddleKernel->setArg(12, prepareRandomNumbers(context.getPaddedNumAtoms()));
    langevinMiddleKernel->setArg(18, pendingCMGroups);
    if (pendingCMMomentum != NULL)
        langevinMiddleKernel->setArg(19, *pendingCMMomentum);
    // The center of mass sum needs threadgroups of 64 threads, as in removeCenterOfMassMomentum.
    langevinMiddleKernel->execute(numLangevinMiddleThreads, 64);
    pendingCMMomentum = NULL;
    pendingCMGroups = 0;
    return true;
}
