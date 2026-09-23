#ifndef OPENMM_METALQUEUE_H_
#define OPENMM_METALQUEUE_H_

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
#include "openmm/common/ComputeQueue.h"
#include <deque>
#include <mutex>

namespace MTL {
    class CommandBuffer;
    class CommandQueue;
    class ComputeCommandEncoder;
    class Device;
}

namespace OpenMM {

/**
 * This class wraps a Metal command queue.  Kernels are not submitted one at a time.  They
 * are encoded into a single open command buffer, which is only committed when the host needs
 * to observe the results (a download, an event, or an explicit flush).  The compute encoder
 * uses serial dispatch, so every kernel sees the writes of the kernels encoded before it.
 *
 * Common code uploads arrays from its worker thread while the main thread encodes kernels, and
 * an upload commits the open command buffer.  Anyone encoding a command must therefore hold the
 * lock from getting the encoder or command buffer until the command is encoded.
 */

class MetalQueue : public ComputeQueueImpl {
public:
    MetalQueue(MTL::Device* device);
    ~MetalQueue();
    MTL::CommandQueue* getQueue() {
        return queue;
    }
    /**
     * Get the lock that serializes encoding and submission.
     */
    std::recursive_mutex& getLock() {
        return lock;
    }
    /**
     * Get the compute encoder of the open command buffer, creating both if necessary.  The
     * caller must hold the lock.
     */
    MTL::ComputeCommandEncoder* getEncoder();
    /**
     * Get the open command buffer with no encoder active, so a different kind of command
     * can be encoded into it.  The caller must hold the lock.
     */
    MTL::CommandBuffer* getCommandBuffer();
    /**
     * Submit the open command buffer, if there is one, without waiting for it.
     */
    void commit();
    /**
     * Submit the open command buffer and block until all work submitted to the queue has
     * completed.
     */
    void finish();
    /**
     * Get the number of command buffers this queue has committed.
     */
    long long getCommitCount() const {
        return commitCount;
    }
private:
    void endEncoding();
    void releaseCompleted();
    MTL::CommandQueue* queue;
    MTL::CommandBuffer* openBuffer;
    MTL::ComputeCommandEncoder* encoder;
    std::deque<MTL::CommandBuffer*> pending;
    long long commitCount;
    std::recursive_mutex lock;
};

} // namespace OpenMM

#endif /*OPENMM_METALQUEUE_H_*/
