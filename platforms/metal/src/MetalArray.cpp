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
#include <iostream>
#include <sstream>
#include <vector>

using namespace OpenMM;

MetalArray::MetalArray() : pointer(0), ownsMemory(false) {
}

MetalArray::MetalArray(MetalContext& context, size_t size, int elementSize, const std::string& name) : pointer(0) {
    initialize(context, size, elementSize, name);
}

MetalArray::~MetalArray() {
    if (pointer != 0 && ownsMemory) {
        ContextSelector selector(*context);
        metalError_t result = metalFree(pointer);
        if (result != metalSuccess) {
            std::stringstream str;
            str<<"Error deleting array "<<name<<": "<<MetalContext::getErrorString(result)<<" ("<<result<<")";
            throw OpenMMException(str.str());
        }
    }
}

void MetalArray::initialize(ComputeContext& context, size_t size, int elementSize, const std::string& name) {
    if (this->pointer != 0)
        throw OpenMMException("MetalArray has already been initialized");
    this->context = &dynamic_cast<MetalContext&>(context);
    this->size = size;
    this->elementSize = elementSize;
    this->name = name;
    ownsMemory = true;
    ContextSelector selector(*this->context);
    metalError_t result = metalMalloc(&pointer, size*elementSize);
    if (result != metalSuccess) {
        std::stringstream str;
        str<<"Error creating array "<<name<<": "<<MetalContext::getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(str.str());
    }
}

void MetalArray::resize(size_t size) {
    if (pointer == 0)
        throw OpenMMException("MetalArray has not been initialized");
    if (!ownsMemory)
        throw OpenMMException("Cannot resize an array that does not own its storage");
    ContextSelector selector(*context);
    metalError_t result = metalFree(pointer);
    if (result != metalSuccess) {
        std::stringstream str;
        str<<"Error deleting array "<<name<<": "<<MetalContext::getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(str.str());
    }
    pointer = 0;
    initialize(*context, size, elementSize, name);
}

ComputeContext& MetalArray::getContext() {
    return *context;
}

void MetalArray::uploadSubArray(const void* data, int offset, int elements, bool blocking) {
    if (pointer == 0)
        throw OpenMMException("MetalArray has not been initialized");
    if (offset < 0 || offset+elements > getSize())
        throw OpenMMException("uploadSubArray: data exceeds range of array");
    metalError_t result;
    result = metalMemcpyAsync(reinterpret_cast<char*>(pointer)+offset*elementSize, const_cast<void*>(data), elements*elementSize, metalMemcpyHostToDevice, context->getCurrentStream());
    if (blocking && result == metalSuccess)
        result = metalStreamSynchronize(context->getCurrentStream());
    if (result != metalSuccess) {
        std::stringstream str;
        str<<"Error uploading array "<<name<<": "<<MetalContext::getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(str.str());
    }
}

void MetalArray::download(void* data, bool blocking) const {
    if (pointer == 0)
        throw OpenMMException("MetalArray has not been initialized");
    metalError_t result;
    result = metalMemcpyAsync(data, pointer, size*elementSize, metalMemcpyDeviceToHost, context->getCurrentStream());
    if (blocking && result == metalSuccess)
        result = metalStreamSynchronize(context->getCurrentStream());
    if (result != metalSuccess) {
        std::stringstream str;
        str<<"Error downloading array "<<name<<": "<<MetalContext::getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(str.str());
    }
}

void MetalArray::copyTo(ArrayInterface& dest) const {
    if (pointer == 0)
        throw OpenMMException("MetalArray has not been initialized");
    if (dest.getSize() != size || dest.getElementSize() != elementSize)
        throw OpenMMException("Error copying array "+name+" to "+dest.getName()+": The destination array does not match the size of the array");
    MetalArray& cuDest = context->unwrap(dest);
    metalError_t result = metalMemcpyAsync(cuDest.getDevicePointer(), pointer, size*elementSize, metalMemcpyDeviceToDevice, context->getCurrentStream());
    if (result != metalSuccess) {
        std::stringstream str;
        str<<"Error copying array "<<name<<" to "<<dest.getName()<<": "<<MetalContext::getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(str.str());
    }
}
