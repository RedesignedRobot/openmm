#ifndef OPENMM_METALEVENT_H_
#define OPENMM_METALEVENT_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2019-2025 Stanford University and the Authors.      *
 * Portions copyright (c) 2020 Advanced Micro Devices, Inc.                   *
 * Portions copyright (c) 2026 the Authors.                                   *
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
#include "openmm/common/ComputeEvent.h"
#include <cstdint>

namespace MTL {
    class CommandBuffer;
    class SharedEvent;
}

namespace OpenMM {

class MetalContext;

/**
 * This is the Metal implementation of the ComputeEventImpl interface.  Enqueueing it
 * commits the current queue's open command buffer with a signal at the end.  Waiting on the
 * host waits for that command buffer, so a failure is reported instead of the signal never
 * arriving.
 */

class MetalEvent : public ComputeEventImpl {
public:
    MetalEvent(MetalContext& context);
    ~MetalEvent();
    void enqueue();
    void wait();
    void queueWait(ComputeQueue queue);
private:
    MetalContext& context;
    MTL::SharedEvent* event;
    MTL::CommandBuffer* buffer;
    uint64_t value;
};

} // namespace OpenMM

#endif /*OPENMM_METALEVENT_H_*/
