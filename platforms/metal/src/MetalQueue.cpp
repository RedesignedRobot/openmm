/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2025 Stanford University and the Authors.           *
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
#include "MetalContext.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;
using namespace std;

// Every committed command buffer holds memory until it completes, so the host stops this many buffers
// ahead of the GPU.
static const int MaxBuffersInFlight = 8;

MetalQueue::MetalQueue(metalStream_t stream) : stream(stream), openBuffer(NULL), encoder(NULL), concurrentDispatch(false), commitCount(0), stopCommitter(false) {
}

MetalQueue::~MetalQueue() {
    if (committer.joinable()) {
        {
            lock_guard<recursive_mutex> guard(lock);
            stopCommitter = true;
        }
        bufferOpened.notify_one();
        committer.join();
    }
    endEncoding();
    if (openBuffer != NULL)
        openBuffer->release();
    for (MTL::CommandBuffer* buffer : pending)
        buffer->release();
    stream->release();
}

MTL::ComputeCommandEncoder* MetalQueue::getEncoder() {
    if (encoder == NULL) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::DispatchType type = (concurrentDispatch ? MTL::DispatchTypeConcurrent : MTL::DispatchTypeSerial);
        encoder = getCommandBuffer()->computeCommandEncoder(type)->retain();
        pool->release();
    }
    return encoder;
}

MTL::CommandBuffer* MetalQueue::getCommandBuffer() {
    endEncoding();
    if (openBuffer == NULL) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        openBuffer = stream->commandBuffer()->retain();
        pool->release();
        if (committer.joinable())
            bufferOpened.notify_one();
    }
    return openBuffer;
}

void MetalQueue::setConcurrentDispatch(bool concurrent) {
    lock_guard<recursive_mutex> guard(lock);
    if (concurrent == concurrentDispatch)
        return;
    endEncoding();
    concurrentDispatch = concurrent;
}

void MetalQueue::endEncoding() {
    if (encoder != NULL) {
        encoder->endEncoding();
        encoder->release();
        encoder = NULL;
    }
}

void MetalQueue::commit() {
    lock_guard<recursive_mutex> guard(lock);
    submit();
    if ((int) pending.size() > MaxBuffersInFlight)
        pending.front()->waitUntilCompleted();
    while (!pending.empty() && pending.front()->status() >= MTL::CommandBufferStatusCompleted) {
        MTL::CommandBuffer* buffer = pending.front();
        pending.pop_front();
        if (buffer->status() == MTL::CommandBufferStatusError)
            throw OpenMMException("Error executing METAL command buffer: "+string(buffer->error()->localizedDescription()->utf8String()));
        buffer->release();
    }
}

void MetalQueue::submit() {
    if (openBuffer != NULL) {
        endEncoding();
        openBuffer->commit();
        pending.push_back(openBuffer);
        openBuffer = NULL;
        commitCount++;
    }
}

void MetalQueue::finish() {
    lock_guard<recursive_mutex> guard(lock);
    commit();
    for (MTL::CommandBuffer* buffer : pending)
        buffer->waitUntilCompleted();
    commit();
}

bool MetalQueue::isIdle() {
    lock_guard<recursive_mutex> guard(lock);
    return pending.empty() || pending.back()->status() >= MTL::CommandBufferStatusCompleted;
}

bool MetalQueue::isDrained() {
    lock_guard<recursive_mutex> guard(lock);
    return openBuffer == NULL && isIdle();
}

void MetalQueue::commitWhenIdle() {
    lock_guard<recursive_mutex> guard(lock);
    if (!committer.joinable())
        committer = thread(&MetalQueue::runIdleCommits, this);
}

void MetalQueue::runIdleCommits() {
    // Wait for the newest committed buffer without holding the lock, then commit the open buffer unless the
    // host committed a newer one meanwhile.  Encoders are only used under the lock, so this never commits
    // a kernel that is half encoded.  Errors are left for the host's next commit() to report.
    unique_lock<recursive_mutex> guard(lock);
    while (!stopCommitter) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        if (!pending.empty() && pending.back()->status() < MTL::CommandBufferStatusCompleted) {
            MTL::CommandBuffer* newest = pending.back()->retain();
            guard.unlock();
            newest->waitUntilCompleted();
            newest->release();
            guard.lock();
        }
        else if (openBuffer != NULL)
            submit();
        else
            bufferOpened.wait(guard);
        pool->release();
    }
}
