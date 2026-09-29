/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 * -------------------------------------------------------------------------- */

#include "MetalAtomReorder.h"
#include "MetalContext.h"
#include "MetalKernelSources.h"
#include "MetalQueue.h"
#include <cmath>
#include <cstdlib>

using namespace OpenMM;
using namespace std;

MetalAtomReorder::MetalAtomReorder(MetalContext& context) : context(context), orderReady(false) {
    ComputeProgram program = context.compileProgram(MetalKernelSources::reorderAtoms);
    bool mixed = context.getUseMixedPrecision();
    copyKernel = program->createKernel("copyPositions");
    for (int i = 0; i < 3; i++)
        copyKernel->addArg();
    gatherKernel = program->createKernel("gatherAtoms");
    for (int i = 0; i < (mixed ? 13 : 12); i++)
        gatherKernel->addArg();
    storeKernel = program->createKernel("storeAtoms");
    for (int i = 0; i < (mixed ? 8 : 7); i++)
        storeKernel->addArg();
}

void MetalAtomReorder::prepare() {
    if (orderReady)
        return;
    if (snapshot != nullptr) {
        if (snapshotDone->load())
            computeOrder();
        return;
    }
    if (context.getStepsSinceReorder() != 250-ReorderLead)
        return;
    int numAtoms = context.getNumAtoms();
    snapshot.reset(new MetalArray(context, numAtoms, sizeof(mm_float4), "reorderSnapshot"));
    snapshotDone = make_shared<atomic<bool> >(false);
    copyKernel->setArg(0, numAtoms);
    copyKernel->setArg(1, context.getPosq());
    copyKernel->setArg(2, *snapshot);
    copyKernel->execute(numAtoms);
    shared_ptr<atomic<bool> > done = snapshotDone;
    context.getCurrentStream().addCompletedHandler([done](MTL::CommandBuffer*) {
        done->store(true);
    });
}

void MetalAtomReorder::computeOrder() {
    const mm_float4* positions = (const mm_float4*) snapshot->getDevicePointer()->contents();
    context.findAtomOrder<float, mm_float4>(positions, newToOld, cells);
    snapshot.reset();
    orderReady = true;
}

bool MetalAtomReorder::apply() {
    if (!orderReady) {
        if (snapshot == nullptr)
            return false;
        if (!snapshotDone->load())
            context.getCurrentStream().finish();
        computeOrder();
    }
    orderReady = false;
    int numAtoms = context.getNumAtoms();
    int paddedNumAtoms = context.getPaddedNumAtoms();

    // The shift is packed into three signed bytes.  A molecule 128 boxes away is left to the host path.

    for (int i = 0; i < numAtoms; i++)
        if (abs(cells[i].x) > 127 || abs(cells[i].y) > 127 || abs(cells[i].z) > 127)
            return false;
    context.updateAtomOrder(newToOld, cells);
    const vector<int>& atomIndex = context.getAtomIndex();

    // Fresh buffers are not in use by the GPU, so the host writes them directly.  The command buffer
    // retains them after they go out of scope here.

    MetalArray order(context, numAtoms, sizeof(mm_int4), "reorderOrder");
    mm_int4* entries = (mm_int4*) order.getDevicePointer()->contents();
    for (int i = 0; i < numAtoms; i++) {
        int old = newToOld[i];
        const mm_int4& c = cells[old];
        int packed = (c.x & 0xFF) | ((c.y & 0xFF) << 8) | ((c.z & 0xFF) << 16);
        entries[i] = mm_int4(old, packed, atomIndex[i], 0);
    }
    bool mixed = context.getUseMixedPrecision();
    int words = (mixed ? 4 : 2);
    MetalArray scratch(context, (size_t) paddedNumAtoms*words, sizeof(mm_float4), "reorderScratch");
    Vec3 box[3];
    context.getPeriodicBoxVectors(box[0], box[1], box[2]);
    mm_float4 hi[3], lo[3];
    for (int i = 0; i < 3; i++) {
        float h[3], l[3];
        for (int j = 0; j < 3; j++) {
            h[j] = (float) box[i][j];
            l[j] = (float) (box[i][j]-h[j]);
        }
        hi[i] = mm_float4(h[0], h[1], h[2], 0);
        lo[i] = mm_float4(l[0], l[1], l[2], 0);
    }
    int arg = 0;
    gatherKernel->setArg(arg++, numAtoms);
    gatherKernel->setArg(arg++, paddedNumAtoms);
    gatherKernel->setArg(arg++, order);
    gatherKernel->setArg(arg++, context.getPosq());
    if (mixed)
        gatherKernel->setArg(arg++, context.getPosqCorrection());
    gatherKernel->setArg(arg++, context.getVelm());
    gatherKernel->setArg(arg++, scratch);
    for (int i = 0; i < 3; i++)
        gatherKernel->setArg(arg++, hi[i]);
    for (int i = 0; i < 3; i++)
        gatherKernel->setArg(arg++, lo[i]);
    gatherKernel->execute(paddedNumAtoms);
    arg = 0;
    storeKernel->setArg(arg++, numAtoms);
    storeKernel->setArg(arg++, paddedNumAtoms);
    storeKernel->setArg(arg++, order);
    storeKernel->setArg(arg++, scratch);
    storeKernel->setArg(arg++, context.getPosq());
    if (mixed)
        storeKernel->setArg(arg++, context.getPosqCorrection());
    storeKernel->setArg(arg++, context.getVelm());
    storeKernel->setArg(arg++, context.getAtomIndexArray());
    storeKernel->execute(paddedNumAtoms);
    return true;
}

void MetalAtomReorder::cancel() {
    snapshot.reset();
    orderReady = false;
}
