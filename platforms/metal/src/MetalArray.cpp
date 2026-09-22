/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2012-2022 Stanford University and the Authors.      *
 * Portions copyright (c) 2020-2023 Advanced Micro Devices, Inc.              *
 * Portions copyright (c) 2026 the Authors.                                   *
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
#include "MetalQueue.h"
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstring>

using namespace OpenMM;

MetalArray::MetalArray() : buffer(NULL), ownsMemory(false) {
}

MetalArray::MetalArray(MetalContext& context, size_t size, int elementSize, const std::string& name) : buffer(NULL) {
    initialize(context, size, elementSize, name);
}

MetalArray::~MetalArray() {
    if (buffer != NULL && ownsMemory)
        buffer->release();
}

void MetalArray::initialize(ComputeContext& context, size_t size, int elementSize, const std::string& name) {
    if (buffer != NULL)
        throw OpenMMException("MetalArray has already been initialized");
    this->context = &dynamic_cast<MetalContext&>(context);
    this->size = size;
    this->elementSize = elementSize;
    this->name = name;
    ownsMemory = true;
    // Metal cannot create empty buffers.
    size_t bytes = std::max(size*elementSize, (size_t) 16);
    buffer = this->context->getDevice()->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (buffer == NULL)
        throw OpenMMException("Error creating array "+name+": failed to allocate "+this->context->intToString(bytes)+" bytes");
}

void MetalArray::resize(size_t size) {
    if (buffer == NULL)
        throw OpenMMException("MetalArray has not been initialized");
    if (!ownsMemory)
        throw OpenMMException("Cannot resize an array that does not own its storage");
    buffer->release();
    buffer = NULL;
    initialize(*context, size, elementSize, name);
}

ComputeContext& MetalArray::getContext() {
    return *context;
}

void MetalArray::uploadSubArray(const void* data, int offset, int elements, bool blocking) {
    if (buffer == NULL)
        throw OpenMMException("MetalArray has not been initialized");
    if (offset < 0 || offset+elements > getSize())
        throw OpenMMException("uploadSubArray: data exceeds range of array");
    context->getCurrentMetalQueue().finish();
    memcpy((char*) buffer->contents()+offset*elementSize, data, elements*elementSize);
}

void MetalArray::download(void* data, bool blocking) const {
    if (buffer == NULL)
        throw OpenMMException("MetalArray has not been initialized");
    context->getCurrentMetalQueue().finish();
    memcpy(data, buffer->contents(), size*elementSize);
}

void MetalArray::copyTo(ArrayInterface& dest) const {
    if (buffer == NULL)
        throw OpenMMException("MetalArray has not been initialized");
    if (dest.getSize() != size || dest.getElementSize() != elementSize)
        throw OpenMMException("Error copying array "+name+" to "+dest.getName()+": The destination array does not match the size of the array");
    if (size == 0)
        return;
    MTL::CommandBuffer* commandBuffer = context->getCurrentMetalQueue().getCommandBuffer();
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::BlitCommandEncoder* blit = commandBuffer->blitCommandEncoder();
    blit->copyFromBuffer(buffer, 0, context->unwrap(dest).getBuffer(), 0, size*elementSize);
    blit->endEncoding();
    pool->release();
}
