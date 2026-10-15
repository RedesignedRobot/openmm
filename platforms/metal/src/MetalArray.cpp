/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2012-2026 Stanford University and the Authors.      *
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

#include "MetalArray.h"
#include "MetalContext.h"
#include "MetalQueue.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

using namespace OpenMM;

MetalArray::MetalArray() : buffer(nullptr), ownsMemory(false), pairStorage(false) {
}

MetalArray::MetalArray(MetalContext& context, size_t size, int elementSize, const std::string& name) : buffer(nullptr), pairStorage(false) {
    initialize(context, size, elementSize, name);
}

MetalArray::~MetalArray() {
    if (buffer != nullptr && ownsMemory)
        buffer->release();
}

void MetalArray::initialize(ComputeContext& context, size_t size, int elementSize, const std::string& name) {
    if (this->buffer != nullptr)
        throw OpenMMException("MetalArray has already been initialized");
    this->context = &dynamic_cast<MetalContext&>(context);
    this->size = size;
    this->elementSize = elementSize;
    this->name = name;
    ownsMemory = true;
    buffer = this->context->getDevice().newBuffer(size*elementSize, MTL::ResourceStorageModeShared);
    if (buffer == nullptr)
        throw OpenMMException("Error creating array "+name);
}

void MetalArray::resize(size_t size) {
    if (buffer == nullptr)
        throw OpenMMException("MetalArray has not been initialized");
    if (!ownsMemory)
        throw OpenMMException("Cannot resize an array that does not own its storage");
    buffer->release();
    buffer = nullptr;
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

// hi is the double rounded to float, as df64 decodes it, and lo the remainder rounded to float.
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
    if (buffer == nullptr)
        throw OpenMMException("MetalArray has not been initialized");
    if (offset < 0 || offset+elements > getSize())
        throw OpenMMException("uploadSubArray: data exceeds range of array");
    MetalQueue& queue = dynamic_cast<MetalQueue&>(*context->getCurrentQueue());
    queue.flush(true);
    char* dest = (char*) buffer->contents()+offset*elementSize;
    if (pairStorage)
        doublesToPairs((const double*) data, (float*) dest, elements*4);
    else
        memcpy(dest, data, elements*elementSize);
}

void MetalArray::download(void* data, bool blocking) const {
    if (buffer == nullptr)
        throw OpenMMException("MetalArray has not been initialized");
    MetalQueue& queue = dynamic_cast<MetalQueue&>(*context->getCurrentQueue());
    queue.flush(true);
    if (pairStorage)
        pairsToDoubles((const float*) buffer->contents(), (double*) data, size*4);
    else
        memcpy(data, buffer->contents(), size*elementSize);
}

void MetalArray::copyTo(ArrayInterface& dest) const {
    if (buffer == nullptr)
        throw OpenMMException("MetalArray has not been initialized");
    if (dest.getSize() != size || dest.getElementSize() != elementSize)
        throw OpenMMException("Error copying array "+name+" to "+dest.getName()+": The destination array does not match the size of the array");
    MetalArray& metalDest = context->unwrap(dest);
    if (metalDest.pairStorage && !pairStorage)
        throw OpenMMException("Error copying array "+name+" to "+dest.getName()+": the destination stores float pairs and the source doesn't");
    metalDest.pairStorage = pairStorage;
    MetalQueue& queue = dynamic_cast<MetalQueue&>(*context->getCurrentQueue());
    queue.flush();
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::BlitCommandEncoder* blit = queue.getCommandBuffer().blitCommandEncoder();
    blit->copyFromBuffer(buffer, 0, metalDest.getBuffer(), 0, size*elementSize);
    blit->endEncoding();
    pool->release();
}
