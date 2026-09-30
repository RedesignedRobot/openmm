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
#include <algorithm>
#include <unistd.h>

using namespace OpenMM;
using namespace std;

// Every committed command buffer holds memory until it completes, so the host stops this many buffers
// ahead of the GPU.
static const int MaxBuffersInFlight = 8;

// On the M3 Ultra a command buffer committed to a drained GPU starts 130 to 145 us after commit, against 35 to 45
// us when the GPU is busy.  While finish() waits, keep-alive slices of about this many microseconds run on a second
// command queue, at most this many in flight, so the GPU is still awake when the host commits the next real buffer.
// Every slice is a finite loop; the host stops issuing them when the wait returns.
static const double KeepAliveSliceSeconds = 100e-6;
static const int KeepAliveSlicesInFlight = 2;
static const unsigned int KeepAliveCalibrationIterations = 1<<15;
static const char* KeepAliveSource =
    "kernel void keepAlive(device unsigned int* out [[buffer(0)]], constant unsigned int& n [[buffer(1)]],\n"
    "                      unsigned int id [[thread_position_in_grid]]) {\n"
    "    unsigned int x = id+1;\n"
    "    for (unsigned int i = 0; i < n; i++)\n"
    "        x = x*1664525u+1013904223u;\n"
    "    out[id] = x;\n"
    "}\n";
static const int KeepAliveThreads = 32;

atomic<long long> MetalQueue::workCount(0);

MetalQueue::MetalQueue(metalStream_t stream) : stream(stream), openBuffer(NULL), encoder(NULL), concurrentDispatch(false), commitCount(0), stopCommitter(false), finishEvent(NULL), finishValue(0), finishedCommits(-1),
        keepAliveQueue(NULL), keepAlivePipeline(NULL), keepAliveOut(NULL), keepAliveIterations(0) {
    keepAliveEnabled = stream->device()->supportsFamily(MTL::GPUFamilyApple7);
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
    if (finishEvent != NULL)
        finishEvent->release();
    for (MTL::CommandBuffer* slice : keepAlivePending) {
        slice->waitUntilCompleted();
        slice->release();
    }
    if (keepAliveOut != NULL)
        keepAliveOut->release();
    if (keepAlivePipeline != NULL)
        keepAlivePipeline->release();
    if (keepAliveQueue != NULL)
        keepAliveQueue->release();
    stream->release();
}

MTL::ComputeCommandEncoder* MetalQueue::getEncoder() {
    workCount++;
    if (encoder == NULL) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::DispatchType type = (concurrentDispatch ? MTL::DispatchTypeConcurrent : MTL::DispatchTypeSerial);
        encoder = getCommandBuffer()->computeCommandEncoder(type)->retain();
        pool->release();
    }
    return encoder;
}

MTL::CommandBuffer* MetalQueue::getCommandBuffer() {
    workCount++;
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

void MetalQueue::addCompletedHandler(const MTL::HandlerFunction& handler) {
    lock_guard<recursive_mutex> guard(lock);
    completedHandlers.push_back(handler);
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
        for (const MTL::HandlerFunction& handler : completedHandlers)
            openBuffer->addCompletedHandler(handler);
        completedHandlers.clear();
        openBuffer->commit();
        pending.push_back(openBuffer);
        openBuffer = NULL;
        commitCount++;
    }
}

void MetalQueue::finish() {
    lock_guard<recursive_mutex> guard(lock);
    if (openBuffer == NULL && isIdle()) {
        commit();
        return;
    }
    // A command buffer's completion wakes a waiting host 90 to 110 us after the GPU finishes on the M3 Ultra, and
    // a download waits for that every time.  An event signaled at the end of the queue and polled every 10 us is
    // seen within about 22 us (MetalEvent::poll).  The queue runs its command buffers in order, so the signal
    // follows every kernel committed before it.
    if (finishEvent == NULL)
        finishEvent = stream->device()->newSharedEvent();
    MTL::CommandBuffer* last = getCommandBuffer()->retain();
    last->encodeSignalEvent(finishEvent, ++finishValue);
    commit();
    while (finishEvent->signaledValue() < finishValue && last->status() < MTL::CommandBufferStatusCompleted) {
        if (keepAliveEnabled)
            keepAlive();
        usleep(10);
    }
    MTL::CommandBufferStatus status = last->status();
    string error = (status == MTL::CommandBufferStatusError ? string(last->error()->localizedDescription()->utf8String()) : "");
    last->release();
    if (status == MTL::CommandBufferStatusError)
        throw OpenMMException("Error executing METAL command buffer: "+error);
    atomic_thread_fence(memory_order_acquire);
    commit();
    finishedCommits = commitCount;
}

void MetalQueue::keepAlive() {
    // Create the second queue, the slice kernel and its output on first use, and size the slice once from GPU
    // timestamps of a calibration run.
    if (keepAlivePipeline == NULL) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Device* device = stream->device();
        NS::Error* error = NULL;
        MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
        MTL::Library* library = device->newLibrary(NS::String::string(KeepAliveSource, NS::UTF8StringEncoding), options, &error);
        options->release();
        MTL::Function* function = (library == NULL ? NULL : library->newFunction(NS::String::string("keepAlive", NS::UTF8StringEncoding)));
        MTL::ComputePipelineState* pipeline = (function == NULL ? NULL : device->newComputePipelineState(function, &error));
        if (function != NULL)
            function->release();
        if (library != NULL)
            library->release();
        pool->release();
        if (pipeline == NULL) {
            keepAliveEnabled = false;
            return;
        }
        keepAlivePipeline = pipeline;
        keepAliveQueue = device->newCommandQueue();
        keepAliveOut = device->newBuffer(KeepAliveThreads*sizeof(unsigned int), MTL::ResourceStorageModePrivate);
        keepAliveIterations = KeepAliveCalibrationIterations;
        keepAlive();
        MTL::CommandBuffer* probe = keepAlivePending.front();
        probe->waitUntilCompleted();
        double seconds = probe->GPUEndTime()-probe->GPUStartTime();
        if (seconds > 0)
            keepAliveIterations = (unsigned int) max(1024.0, min(1.0e7, KeepAliveCalibrationIterations*KeepAliveSliceSeconds/seconds));
        reapKeepAlive();
    }
    reapKeepAlive();
    if ((int) keepAlivePending.size() >= KeepAliveSlicesInFlight)
        return;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::CommandBuffer* slice = keepAliveQueue->commandBuffer()->retain();
    MTL::ComputeCommandEncoder* sliceEncoder = slice->computeCommandEncoder();
    sliceEncoder->setComputePipelineState(keepAlivePipeline);
    sliceEncoder->setBuffer(keepAliveOut, 0, 0);
    sliceEncoder->setBytes(&keepAliveIterations, sizeof(keepAliveIterations), 1);
    sliceEncoder->dispatchThreads(MTL::Size(KeepAliveThreads, 1, 1), MTL::Size(KeepAliveThreads, 1, 1));
    sliceEncoder->endEncoding();
    slice->commit();
    keepAlivePending.push_back(slice);
    pool->release();
}

void MetalQueue::reapKeepAlive() {
    while (!keepAlivePending.empty() && keepAlivePending.front()->status() >= MTL::CommandBufferStatusCompleted) {
        keepAlivePending.front()->release();
        keepAlivePending.pop_front();
    }
}

bool MetalQueue::isIdle() {
    lock_guard<recursive_mutex> guard(lock);
    return pending.empty() || commitCount == finishedCommits || pending.back()->status() >= MTL::CommandBufferStatusCompleted;
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
