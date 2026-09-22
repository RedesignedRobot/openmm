/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2025 Stanford University and the Authors.           *
 * Portions copyright (c) 2026 the Authors.                                   *
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
#include "MetalQueue.h"
#include "openmm/OpenMMException.h"
#include <Metal/Metal.hpp>
#include <string>

using namespace OpenMM;
using namespace std;

MetalQueue::MetalQueue(MTL::Device* device) : openBuffer(NULL), encoder(NULL), commitCount(0) {
    queue = device->newCommandQueue();
    if (queue == NULL)
        throw OpenMMException("Error creating Metal command queue");
}

MetalQueue::~MetalQueue() {
    endEncoding();
    if (openBuffer != NULL) {
        openBuffer->commit();
        pending.push_back(openBuffer);
    }
    for (MTL::CommandBuffer* buffer : pending) {
        buffer->waitUntilCompleted();
        buffer->release();
    }
    queue->release();
}

MTL::ComputeCommandEncoder* MetalQueue::getEncoder() {
    if (encoder == NULL) {
        MTL::CommandBuffer* buffer = getCommandBuffer();
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        encoder = buffer->computeCommandEncoder()->retain();
        pool->release();
    }
    return encoder;
}

MTL::CommandBuffer* MetalQueue::getCommandBuffer() {
    endEncoding();
    if (openBuffer == NULL) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        openBuffer = queue->commandBuffer()->retain();
        pool->release();
    }
    return openBuffer;
}

void MetalQueue::endEncoding() {
    if (encoder != NULL) {
        encoder->endEncoding();
        encoder->release();
        encoder = NULL;
    }
}

void MetalQueue::commit() {
    if (openBuffer == NULL)
        return;
    endEncoding();
    openBuffer->commit();
    pending.push_back(openBuffer);
    openBuffer = NULL;
    commitCount++;
    releaseCompleted();
}

void MetalQueue::finish() {
    commit();
    for (MTL::CommandBuffer* buffer : pending)
        buffer->waitUntilCompleted();
    releaseCompleted();
}

void MetalQueue::releaseCompleted() {
    while (!pending.empty() && pending.front()->status() >= MTL::CommandBufferStatusCompleted) {
        MTL::CommandBuffer* buffer = pending.front();
        pending.pop_front();
        bool failed = (buffer->status() == MTL::CommandBufferStatusError);
        string message;
        if (failed && buffer->error() != NULL)
            message = buffer->error()->localizedDescription()->utf8String();
        buffer->release();
        if (failed)
            throw OpenMMException("Error executing Metal command buffer: "+message);
    }
}
