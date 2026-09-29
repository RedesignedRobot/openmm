#ifndef OPENMM_METALATOMREORDER_H_
#define OPENMM_METALATOMREORDER_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 * -------------------------------------------------------------------------- */

#include "MetalArray.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeKernel.h"
#include <atomic>
#include <memory>
#include <vector>

namespace OpenMM {

class MetalContext;

/**
 * Computes each scheduled atom reorder ahead of time, so it does not drain the queue.  The synchronous path
 * downloads positions and velocities, which waits for every queued step, then sorts on the host while the GPU
 * sits idle, about 3 ms per reorder on apoa1.  Here a kernel copies positions ReorderLead steps before the
 * reorder is due, the host computes the order once that copy has completed, and two kernels apply it in stream
 * order.  MetalQueue keeps at most 9 committed buffers of 8 steps, so the host runs at most about 72 steps
 * ahead of the GPU, and the copy has finished by the time the order is needed.
 *
 * The order comes from positions ReorderLead steps old.  It is still an exchange of identical molecules and a
 * whole number of box vectors per molecule, so the physics is unchanged; molecules are sorted slightly less
 * tightly.  A NaN position is reported up to ReorderLead steps after the reorder that would have caught it.
 */
class MetalAtomReorder : public ComputeContext::ReorderPlanner {
public:
    MetalAtomReorder(MetalContext& context);
    void prepare();
    bool apply();
    void cancel();
private:
    static const int ReorderLead = 100;
    void computeOrder();
    MetalContext& context;
    ComputeKernel copyKernel, gatherKernel, storeKernel;
    std::unique_ptr<MetalArray> snapshot;
    std::shared_ptr<std::atomic<bool> > snapshotDone;
    std::vector<int> newToOld;
    std::vector<mm_int4> cells;
    bool orderReady;
};

} // namespace OpenMM

#endif /*OPENMM_METALATOMREORDER_H_*/
