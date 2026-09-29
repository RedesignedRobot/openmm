/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2012-2022 Stanford University and the Authors.      *
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

#include "MetalArray.h"
#include "MetalContext.h"
#include "openmm/common/ContextSelector.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

using namespace OpenMM;

MetalArray::MetalArray() : pointer(0), ownsMemory(false), pairStorage(false) {
}

MetalArray::MetalArray(MetalContext& context, size_t size, int elementSize, const std::string& name) : pointer(0), pairStorage(false) {
    initialize(context, size, elementSize, name);
}

MetalArray::~MetalArray() {
    if (pointer != 0 && ownsMemory)
        pointer->release();
}

void MetalArray::initialize(ComputeContext& context, size_t size, int elementSize, const std::string& name) {
    if (this->pointer != 0)
        throw OpenMMException("MetalArray has already been initialized");
    MetalQueue::noteWork();
    this->context = &dynamic_cast<MetalContext&>(context);
    this->size = size;
    this->elementSize = elementSize;
    this->name = name;
    ownsMemory = true;
    pointer = this->context->getDevice()->newBuffer(std::max(size*elementSize, (size_t) 16), MTL::ResourceStorageModeShared);
    if (pointer == 0)
        throw OpenMMException("Error creating array "+name);
}

void MetalArray::resize(size_t size) {
    if (pointer == 0)
        throw OpenMMException("MetalArray has not been initialized");
    if (!ownsMemory)
        throw OpenMMException("Cannot resize an array that does not own its storage");
    pointer->release();
    pointer = 0;
    initialize(*context, size, elementSize, name);
}

ComputeContext& MetalArray::getContext() {
    return *context;
}

void MetalArray::setPairStorage(bool pairs) {
    if (pairs && elementSize != 4*sizeof(double))
        throw OpenMMException("Array "+name+" can't store float pairs: its elements aren't mixed4");
    pairStorage = pairs;
}

// hi is the double rounded to float, as df64 decodes it, and lo the exact remainder rounded to float.
static void doublesToPairs(const double* source, float* dest, size_t count) {
    for (size_t i = 0; i < count; i++) {
        float hi = (float) source[i];
        dest[2*i] = hi;
        dest[2*i+1] = (std::isfinite(hi) ? (float) (source[i]-hi) : 0.0f);
    }
}

static void pairsToDoubles(const float* source, double* dest, size_t count) {
    for (size_t i = 0; i < count; i++)
        dest[i] = (double) source[2*i]+(double) source[2*i+1];
}

void MetalArray::uploadSubArray(const void* data, int offset, int elements, bool blocking) {
    if (pointer == 0)
        throw OpenMMException("MetalArray has not been initialized");
    if (offset < 0 || offset+elements > getSize())
        throw OpenMMException("uploadSubArray: data exceeds range of array");
    MetalQueue& queue = context->getCurrentStream();
    size_t bytes = (size_t) elements*elementSize, destOffset = (size_t) offset*elementSize;
    if (bytes == 0)
        return;
    std::lock_guard<std::recursive_mutex> guard(queue.getLock());
    // While the GPU still has work, a host write could race kernels encoded before this upload, and waiting for them
    // leaves the GPU idle for a full round trip (about 250 us per upload on the M3 Ultra; CustomIntegrator steps
    // upload their globals several times per step).  So stage the data and copy it in order on the queue instead.
    if (!queue.isDrained() && bytes%4 == 0 && destOffset%4 == 0) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Buffer* staging = context->getDevice()->newBuffer(bytes, MTL::ResourceStorageModeShared);
        if (staging == 0)
            throw OpenMMException("uploadSubArray: can't allocate a staging buffer for "+name);
        if (pairStorage)
            doublesToPairs((const double*) data, (float*) staging->contents(), elements*4);
        else
            memcpy(staging->contents(), data, bytes);
        MTL::BlitCommandEncoder* blit = queue.getCommandBuffer()->blitCommandEncoder();
        blit->copyFromBuffer(staging, 0, pointer, destOffset, bytes);
        blit->endEncoding();
        MetalQueue::noteWork();
        staging->release();   // the command buffer retains it until the copy completes
        pool->release();
        return;
    }
    queue.finish();
    MetalQueue::noteWork();
    char* dest = reinterpret_cast<char*>(pointer->contents())+destOffset;
    if (pairStorage)
        doublesToPairs((const double*) data, (float*) dest, elements*4);
    else
        memcpy(dest, data, elements*elementSize);
}

void MetalArray::download(void* data, bool blocking) const {
    if (pointer == 0)
        throw OpenMMException("MetalArray has not been initialized");
    context->getCurrentStream().finish();
    if (pairStorage)
        pairsToDoubles((const float*) pointer->contents(), (double*) data, size*4);
    else
        memcpy(data, pointer->contents(), size*elementSize);
}

void MetalArray::copyTo(ArrayInterface& dest) const {
    if (pointer == 0)
        throw OpenMMException("MetalArray has not been initialized");
    if (dest.getSize() != size || dest.getElementSize() != elementSize)
        throw OpenMMException("Error copying array "+name+" to "+dest.getName()+": The destination array does not match the size of the array");
    MetalArray& cuDest = context->unwrap(dest);
    if (cuDest.pairStorage && !pairStorage)
        throw OpenMMException("Error copying array "+name+" to "+dest.getName()+": the destination stores float pairs and the source doesn't");
    cuDest.pairStorage = pairStorage;
    MetalQueue& queue = context->getCurrentStream();
    std::lock_guard<std::recursive_mutex> guard(queue.getLock());
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::BlitCommandEncoder* blit = queue.getCommandBuffer()->blitCommandEncoder();
    blit->copyFromBuffer(pointer, 0, cuDest.getDevicePointer(), 0, size*elementSize);
    blit->endEncoding();
    pool->release();
}
