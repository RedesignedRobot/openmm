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

// The fused LangevinMiddle kernel's arguments for the autoclear buffers other than the force buffer: MaxClearedBuffers
// buffers starting at ClearArgument, then their sizes in 32 bit words as an int4.  The flag that turns the clear on
// comes two arguments before them, followed by the atom count.
static const int MaxClearedBuffers = 4;
static const int ClearArgument = 22;

MetalIntegrationUtilities::MetalIntegrationUtilities(MetalContext& context, const System& system) : IntegrationUtilities(context, system),
        ccmaConvergedMemory(NULL), numClearThreads(0), numBoundAutoclearBuffers(-1), clearsAutoclearBuffers(false),
        pendingCMBlockSize(0), pendingCMVelocity(NULL) {
        ccmaEvent = context.createEvent();
        ccmaConvergedDeviceMemory = context.getDevice()->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
        ccmaConvergedMemory = (int*) ccmaConvergedDeviceMemory->contents();
        // posDelta shares kernels and raw copies with velm, so it stores float pairs too.  The zeros the
        // base class uploaded read the same either way.
        getPosDelta().setPairStorage(context.getUseMixedPrecision());
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
    return !ccmaConstraintAtoms.isInitialized() && !context.getUseDoublePrecision();
}

void MetalIntegrationUtilities::removeCenterOfMassMomentum(ComputeKernel& calcKernel, int blockSize, ComputeArray& cmVelocity) {
    pendingCMCalc = calcKernel;
    pendingCMBlockSize = blockSize;
    pendingCMVelocity = &cmVelocity;
}

void MetalIntegrationUtilities::calcPendingCenterOfMassMomentum() {
    if (!pendingCMCalc)
        return;
    pendingCMCalc->execute(pendingCMBlockSize, pendingCMBlockSize);
    pendingCMCalc.reset();
}

bool MetalIntegrationUtilities::integrateLangevinMiddle(ComputeArray& params, double tol) {
    if (ccmaConstraintAtoms.isInitialized() || context.getUseDoublePrecision())
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
        // all four lanes.  That pays only while all the quads fit in one pass of the grid executeKernel launches.  Past
        // that, each quad runs several units in sequence and pays the redundant constraint work in full.
        int numQuadUnits = numSettle+numShake+numFreeQuads;
        bool useQuads = (4*numQuadUnits <= context.getNumThreadBlocks()*MetalContext::ThreadBlockSize);

        // Positions and velocities that overflow are lost either way, so this program drops df64's overflow guards.
        map<string, string> fusedDefines = defines;
        fusedDefines["DF64_FINITE"] = "";
        ComputeProgram program = context.compileProgram(CommonKernelSources::integrationUtilities+MetalKernelSources::fusedLangevinMiddle, fusedDefines);
        int numUnits = (useQuads ? numQuadUnits : numSettle+numShake+numFree);
        numLangevinMiddleThreads = (useQuads ? 4*numUnits : numUnits);
        langevinMiddleKernel = program->createKernel(useQuads ? "integrateLangevinMiddleFusedQuads" : "integrateLangevinMiddleFused");
        langevinMiddleKernel->addArg(numSettle);
        langevinMiddleKernel->addArg(numShake);
        langevinMiddleKernel->addArg(numUnits);
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
        langevinMiddleKernel->addArg(0); // bindAutoclearBuffers() sets the clear arguments.
        langevinMiddleKernel->addArg(numAtoms);
        for (int i = 0; i < MaxClearedBuffers; i++)
            langevinMiddleKernel->addArg(nullptr);
        langevinMiddleKernel->addArg(mm_int4(0, 0, 0, 0));
    }
    bool clearBuffers = bindAutoclearBuffers();
    if (context.getUseMixedPrecision())
        langevinMiddleKernel->setArg(4, tol);
    else
        langevinMiddleKernel->setArg(4, (float) tol);
    langevinMiddleKernel->setArg(9, params);
    langevinMiddleKernel->setArg(12, prepareRandomNumbers(context.getPaddedNumAtoms()));
    calcPendingCenterOfMassMomentum();
    langevinMiddleKernel->setArg(18, pendingCMVelocity == NULL ? 0 : 1);
    if (pendingCMVelocity != NULL)
        langevinMiddleKernel->setArg(19, *pendingCMVelocity);
    langevinMiddleKernel->execute(max(numLangevinMiddleThreads, numClearThreads));
    if (clearBuffers)
        dynamic_cast<MetalContext&>(context).markAutoclearBuffersCleared();
    pendingCMVelocity = NULL;
    return true;
}

bool MetalIntegrationUtilities::bindAutoclearBuffers() {
    // The kernel zeroes each atom's force after reading it and clears up to MaxClearedBuffers other buffers with the
    // whole grid.  The buffer list only grows while forces initialize, so it is bound again only when its size changes.

    MetalContext& cu = dynamic_cast<MetalContext&>(context);
    const vector<ArrayInterface*>& buffers = cu.getAutoclearBuffers();
    if (buffers.size() == numBoundAutoclearBuffers)
        return clearsAutoclearBuffers;
    numBoundAutoclearBuffers = buffers.size();
    MetalArray* force = &cu.unwrap(cu.getLongForceBuffer());
    vector<ArrayInterface*> others;
    int numForce = 0;
    for (ArrayInterface* buffer : buffers) {
        if (&cu.unwrap(*buffer) == force)
            numForce++;
        else
            others.push_back(buffer);
    }
    clearsAutoclearBuffers = (numForce == 1 && others.size() <= MaxClearedBuffers);
    int words[MaxClearedBuffers] = {0, 0, 0, 0};
    numClearThreads = 0;
    for (int i = 0; i < MaxClearedBuffers; i++) {
        if (clearsAutoclearBuffers && i < others.size()) {
            words[i] = others[i]->getSize()*others[i]->getElementSize()/4;
            numClearThreads = max(numClearThreads, (words[i]+3)/4);
            langevinMiddleKernel->setArg(ClearArgument+i, *others[i]);
        }
        else
            langevinMiddleKernel->setArg(ClearArgument+i, nullptr);
    }
    langevinMiddleKernel->setArg(ClearArgument-2, clearsAutoclearBuffers ? 1 : 0);
    langevinMiddleKernel->setArg(ClearArgument+MaxClearedBuffers, mm_int4(words[0], words[1], words[2], words[3]));
    return clearsAutoclearBuffers;
}

void MetalIntegrationUtilities::refillExhaustedRandomNumbers() {
    if (!langevinMiddleKernel || !random.isInitialized() || randomPos != random.getSize())
        return;
    randomKernel->execute(random.getSize(), 64);
    randomPos = 0;
}
