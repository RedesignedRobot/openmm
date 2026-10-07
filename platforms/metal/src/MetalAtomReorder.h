#ifndef OPENMM_METALATOMREORDER_H_
#define OPENMM_METALATOMREORDER_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * Authors: Amir Ayub                                                         *
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

#include "MetalArray.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeKernel.h"
#include "openmm/common/ComputeSort.h"
#include <atomic>
#include <memory>
#include <mutex>

namespace OpenMM {

class MetalContext;

/**
 * Reorders atoms on the device, so a reorder does not drain the queue.  The host path downloads positions and
 * velocities, which waits for every queued step, then sorts while the GPU sits idle, about 3 ms per reorder on
 * apoa1.  Here the order comes from the positions at the reorder step, computed and applied by kernels in
 * stream order: the same molecule groups, curves and bins as ComputeContext::findAtomOrder(), with the box
 * shift done in float (single) or df64 (mixed).
 *
 * The device atom index and cell offsets are the current ones after apply().  The host copies catch up in
 * syncHost(), which ComputeContext calls before anything reads them; that waits for the queue, as every
 * reader of positions does anyway.  A NaN position is reported at the next reorder or host sync.
 */
class MetalAtomReorder : public ComputeContext::DeviceReorder {
public:
    MetalAtomReorder(MetalContext& context);
    bool apply();
    void syncHost();
    void hostChanged();
private:
    friend class MetalHostRobustnessAccess;
    bool buildTables();
    void uploadCellOffsets();
    void checkForNaN();
    MetalContext& context;
    ComputeKernel keysKernel, largeKeysKernel, gatherKernel, storeKernel;
    ComputeSort sort;
    std::unique_ptr<MetalArray> molecules, templateAtoms, slotMolecule, largeMolecules, keys, moleculeCells, cellOffsets;
    std::unique_ptr<MetalArray> scratch, scratchIndex, scratchOffsets, nanFlag;
    int tablesVersion;
    bool tablesUsable;
    // getPositions() reads the atom order from thread pool workers, so the host sync is guarded.
    std::atomic<bool> hostStale;
    std::atomic<bool> deviceOffsetsStale;
    std::mutex syncLock;
    int numLarge;
};

} // namespace OpenMM

#endif /*OPENMM_METALATOMREORDER_H_*/
