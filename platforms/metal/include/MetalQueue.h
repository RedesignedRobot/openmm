#ifndef OPENMM_METALQUEUE_H_
#define OPENMM_METALQUEUE_H_

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

#include "openmm/common/ComputeQueue.h"
#include <Metal/Metal.hpp>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

typedef MTL::CommandQueue* metalStream_t;

namespace OpenMM {

/**
 * This is the METAL implementation of the ComputeQueue interface.  It wraps a metalStream_t.
 */

class MetalQueue : public ComputeQueueImpl {
public:
    /**
     * Create a MetalQueue that wraps an existing metalStream_t.
     */
    MetalQueue(metalStream_t stream);
    ~MetalQueue();
    /**
     * Get the CUstream.
     */
    metalStream_t getStream() {
        return stream;
    }
    /**
     * Get the lock that callers must hold for as long as they use the encoder or command buffer this
     * returns, since after commitWhenIdle() another thread can commit the open buffer whenever the lock
     * is free.
     */
    std::recursive_mutex& getLock() {
        return lock;
    }
    MTL::ComputeCommandEncoder* getEncoder();
    MTL::CommandBuffer* getCommandBuffer();
    /**
     * Choose whether getEncoder() opens compute passes whose dispatches may run at the same time.  Changing
     * it ends the current pass.
     */
    void setConcurrentDispatch(bool concurrent);
    /**
     * Run a function when the next command buffer to be committed completes.  It is attached at commit time,
     * so callers never touch the open command buffer or its encoder.
     */
    void addCompletedHandler(const MTL::HandlerFunction& handler);
    void commit();
    void finish();
    /**
     * Get whether the GPU has finished every command buffer committed to this queue.
     */
    bool isIdle();
    /**
     * Get whether every kernel encoded on this queue has been committed and has finished.
     */
    bool isDrained();
    /**
     * From now on, commit the open command buffer from a background thread whenever the GPU finishes every
     * committed buffer.  Code that can leave steps open calls this, so they don't wait for the host's next
     * commit after Integrator::step() returns.
     */
    void commitWhenIdle();
    long long getCommitCount() const {
        return commitCount;
    }
private:
    void endEncoding();
    void submit();
    void runIdleCommits();
    metalStream_t stream;
    MTL::CommandBuffer* openBuffer;
    MTL::ComputeCommandEncoder* encoder;
    bool concurrentDispatch;
    std::deque<MTL::CommandBuffer*> pending;
    std::vector<MTL::HandlerFunction> completedHandlers;
    std::atomic<long long> commitCount;
    std::recursive_mutex lock;
    std::condition_variable_any bufferOpened;
    std::thread committer;
    bool stopCommitter;
};

} // namespace OpenMM

#endif /*OPENMM_METALQUEUE_H_*/
