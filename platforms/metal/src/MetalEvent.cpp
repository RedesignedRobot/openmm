/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2019-2025 Stanford University and the Authors.      *
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

#include "MetalEvent.h"
#include "openmm/OpenMMException.h"
#include <atomic>
#include <unistd.h>

using namespace OpenMM;

MetalEvent::MetalEvent(MetalContext& context) : context(context), buffer(NULL), value(0) {
    event = context.getDevice()->newSharedEvent();
    deviceEvent = context.getDevice()->newEvent();
}

MetalEvent::~MetalEvent() {
    if (buffer != NULL)
        buffer->release();
    event->release();
    deviceEvent->release();
}

void MetalEvent::enqueue() {
    MetalQueue& queue = context.getCurrentStream();
    std::lock_guard<std::recursive_mutex> guard(queue.getLock());
    signal();
    queue.commit();
}

void MetalEvent::signal() {
    MetalQueue& queue = context.getCurrentStream();
    std::lock_guard<std::recursive_mutex> guard(queue.getLock());
    if (buffer != NULL)
        buffer->release();
    buffer = queue.getCommandBuffer()->retain();
    // Both events are signaled: the shared one for host waits, the device one for queue waits.  A queue wait
    // on a MTLSharedEvent starts the waiting buffer 85 to 150 us after the signal on the M3 Ultra, a wait on a
    // MTLEvent within 8 us, and the second signal costs the signaling buffer nothing.
    buffer->encodeSignalEvent(deviceEvent, ++value);
    buffer->encodeSignalEvent(event, value);
}

void MetalEvent::wait() {
    // Waiting on the command buffer instead of the event delays the start of the next command buffer
    // by about 40 us on the M3 Ultra.  The status check returns if the buffer fails before the signal.
    if (buffer != NULL) {
        while (!event->waitUntilSignaledValue(value, 100) && buffer->status() < MTL::CommandBufferStatusCompleted)
            ;
        checkBufferStatus();
    }
}

void MetalEvent::poll() {
    // On the M3 Ultra, waitUntilSignaledValue() wakes the host 90 to 110 us after the signal, and a 10 us
    // poll within about 22 us.  The fence orders the host's reads of GPU results after the signal.
    if (buffer != NULL) {
        while (event->signaledValue() < value && buffer->status() < MTL::CommandBufferStatusCompleted)
            usleep(10);
        checkBufferStatus();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}

void MetalEvent::checkBufferStatus() {
    // A failed buffer can stop before its signal, so the host must not go on to read the results it waited for.
    if (buffer->status() == MTL::CommandBufferStatusError)
        throw OpenMMException("Error executing METAL command buffer: "+std::string(buffer->error()->localizedDescription()->utf8String()));
}

void MetalEvent::queueWait(ComputeQueue queue) {
    MetalQueue& metalQueue = *dynamic_cast<MetalQueue*>(queue.get());
    std::lock_guard<std::recursive_mutex> guard(metalQueue.getLock());
    // On the M3 Ultra a wait encoded after other work in the same command buffer costs about 22 us, and one at the
    // head of a buffer under 4 us, so work already encoded goes in its own buffer first.
    if (metalQueue.hasOpenBuffer())
        metalQueue.commit();
    metalQueue.getCommandBuffer()->encodeWait(deviceEvent, value);
}
