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

#include "MetalContext.h"
#include "openmm/common/ComputeEvent.h"

namespace OpenMM {

/**
 * This is the METAL implementation of the ComputeKernelImpl interface.
 */

class MetalEvent : public ComputeEventImpl {
public:
    MetalEvent(MetalContext& context);
    ~MetalEvent();
    /**
     * Place the event into the device's execution queue.
     */
    void enqueue();
    /**
     * Place the event into the current command buffer without committing it, so the
     * operations that follow share the buffer.  Commit the queue before calling wait().
     */
    void signal();
    /**
     * Block until all operations started before the call to enqueue() have completed.
     */
    void wait();
    /**
     * Like wait(), but poll the event instead of blocking in Metal.  The host wakes sooner after
     * the signal, and spends CPU time while it waits.
     */
    void poll();
    /**
     * Enqueue a barrier that causes a specified ComputeQueue to block until all
     * operations started before the call to enqueue() have completed.
     */
    void queueWait(ComputeQueue queue);
private:
    void checkBufferStatus();
    MetalContext& context;
    MTL::SharedEvent* event;
    MTL::Event* deviceEvent;
    MTL::CommandBuffer* buffer;
    uint64_t value;
};

} // namespace OpenMM

#endif /*OPENMM_METALEVENT_H_*/
