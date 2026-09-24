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

using namespace OpenMM;

MetalEvent::MetalEvent(MetalContext& context) : context(context), buffer(NULL), value(0) {
    event = context.getDevice()->newSharedEvent();
}

MetalEvent::~MetalEvent() {
    if (buffer != NULL)
        buffer->release();
    event->release();
}

void MetalEvent::enqueue() {
    MetalQueue& queue = context.getCurrentStream();
    std::lock_guard<std::recursive_mutex> guard(queue.getLock());
    if (buffer != NULL)
        buffer->release();
    buffer = queue.getCommandBuffer()->retain();
    buffer->encodeSignalEvent(event, ++value);
    queue.commit();
}

void MetalEvent::wait() {
    if (buffer != NULL)
        buffer->waitUntilCompleted();
}

void MetalEvent::queueWait(ComputeQueue queue) {
    MetalQueue& metalQueue = *dynamic_cast<MetalQueue*>(queue.get());
    std::lock_guard<std::recursive_mutex> guard(metalQueue.getLock());
    metalQueue.getCommandBuffer()->encodeWait(event, value);
}
