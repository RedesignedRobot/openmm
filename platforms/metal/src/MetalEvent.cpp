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
#include "MetalQueue.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;

MetalEvent::MetalEvent(MetalContext& context) : context(context), eventCreated(false) {
    metalError_t result = metalEventCreateWithFlags(&event, context.getEventFlags());
    if (result != metalSuccess)
        throw OpenMMException("Error creating METAL event:"+MetalContext::getErrorString(result));
    eventCreated = true;
}

MetalEvent::~MetalEvent() {
    if (eventCreated)
        metalEventDestroy(event);
}

void MetalEvent::enqueue() {
    metalEventRecord(event, context.getCurrentStream());
}

void MetalEvent::wait() {
    metalEventSynchronize(event);
}

void MetalEvent::queueWait(ComputeQueue queue) {
    metalStreamWaitEvent(dynamic_cast<MetalQueue*>(queue.get())->getStream(), event, 0);
}
