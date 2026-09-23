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
#include <deque>
#include <mutex>
#include <string>

namespace OpenMM {

/**
 * This is the METAL implementation of the ComputeQueue interface.  It wraps an MTL::CommandQueue.
 * Kernels are encoded into one open command buffer, which is committed when the host reads or
 * writes an array, waits for an event, or finishes computing forces.
 */

class MetalQueue : public ComputeQueueImpl {
public:
    /**
     * Create a MetalQueue that create a new MTL::CommandQueue.
     */
    MetalQueue(MTL::Device* device);
    ~MetalQueue();
    /**
     * Get the MTL::CommandQueue.
     */
    MTL::CommandQueue* getStream() {
        return stream;
    }
    /**
     * Get the lock that serializes encoding and submission.  CustomCPPForce uploads from the
     * context's worker thread while the main thread encodes kernels.
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
     * Get why a completed command buffer failed, or an empty string if it succeeded.
     */
    static std::string getFailure(MTL::CommandBuffer* buffer);
    /**
     * Get the number of command buffers this queue has committed.
     */
    long long getCommitCount() const {
        return commitCount;
    }
private:
    void endEncoding();
    void releaseCompleted();
    MTL::CommandQueue* stream;
    MTL::CommandBuffer* openBuffer;
    MTL::ComputeCommandEncoder* encoder;
    std::deque<MTL::CommandBuffer*> pending;
    long long commitCount;
    std::recursive_mutex lock;
};

} // namespace OpenMM

#endif /*OPENMM_METALQUEUE_H_*/
