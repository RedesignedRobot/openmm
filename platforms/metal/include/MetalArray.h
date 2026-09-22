#ifndef OPENMM_METALARRAY_H_
#define OPENMM_METALARRAY_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2022 Stanford University and the Authors.      *
 * Portions copyright (c) 2020-2022 Advanced Micro Devices, Inc.              *
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
#include "openmm/OpenMMException.h"
#include "openmm/common/windowsExportCommon.h"
#include "openmm/common/ArrayInterface.h"
#include <iostream>
#include <sstream>
#include <vector>

namespace MTL {
    class Buffer;
}

namespace OpenMM {

class MetalContext;

/**
 * This class encapsulates a Metal buffer.  It provides a simplified API for working with it
 * and for copying data to and from it.  Buffers use shared storage, so transfers are memory
 * copies, but they must first wait for queued kernels that might use the buffer.
 */

class OPENMM_EXPORT_COMMON MetalArray : public ArrayInterface {
public:
    /**
     * Create a MetalArray object.  The object is allocated on the heap with the "new" operator.
     * The template argument is the data type of each array element.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param name              the name of the array
     */
    template <class T>
    static MetalArray* create(MetalContext& context, size_t size, const std::string& name) {
        return new MetalArray(context, size, sizeof(T), name);
    }
    /**
     * Create an uninitialized MetalArray object.  It does not point to any device memory,
     * and cannot be used until initialize() is called on it.
     */
    MetalArray();
    /**
     * Create a MetalArray object.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param elementSize       the size of each element in bytes
     * @param name              the name of the array
     */
    MetalArray(MetalContext& context, size_t size, int elementSize, const std::string& name);
    ~MetalArray();
    /**
     * Initialize this object.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param elementSize       the size of each element in bytes
     * @param name              the name of the array
     */
    void initialize(ComputeContext& context, size_t size, int elementSize, const std::string& name);
    /**
     * Initialize this object.  The template argument is the data type of each array element.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param name              the name of the array
     */
    template <class T>
    void initialize(ComputeContext& context, size_t size, const std::string& name) {
        initialize(context, size, sizeof(T), name);
    }
    /**
     * Recreate the internal storage to have a different size.
     */
    void resize(size_t size);
    /**
     * Get whether this array has been initialized.
     */
    bool isInitialized() const {
        return (buffer != NULL);
    }
    /**
     * Get the number of elements in the array.
     */
    size_t getSize() const {
        return size;
    }
    /**
     * Get the size of each element in bytes.
     */
    int getElementSize() const {
        return elementSize;
    }
    /**
     * Get the name of the array.
     */
    const std::string& getName() const {
        return name;
    }
    /**
     * Get the context this array belongs to.
     */
    ComputeContext& getContext();
    /**
     * Get the Metal buffer.
     */
    MTL::Buffer* getBuffer() {
        return buffer;
    }
    /**
     * Copy the values in a vector to the device memory.
     */
    template <class T>
    void upload(const std::vector<T>& data, bool convert=false) {
        ArrayInterface::upload(data, convert);
    }
    /**
     * Copy the values in the Buffer to a vector.
     */
    template <class T>
    void download(std::vector<T>& data) const {
        ArrayInterface::download(data);
    }
    /**
     * Copy the values from host memory to the array.
     *
     * @param data     the data to copy
     * @param blocking ignored: Metal transfers always complete before this returns
     */
    void upload(const void* data, bool blocking=true) {
        uploadSubArray(data, 0, getSize(), blocking);
    }
    /**
     * Copy values from host memory to a subset of the array.
     *
     * @param data     the data to copy
     * @param offset   the index of the element within the array at which the copy should begin
     * @param elements the number of elements to copy
     * @param blocking ignored: Metal transfers always complete before this returns
     */
    void uploadSubArray(const void* data, int offset, int elements, bool blocking=true);
    /**
     * Copy the values in the device memory to an array.
     *
     * @param data     the array to copy the memory to
     * @param blocking ignored: Metal transfers always complete before this returns
     */
    void download(void* data, bool blocking=true) const;
    /**
     * Copy the values in the device memory to a second array.
     *
     * @param dest     the destination array to copy to
     */
    void copyTo(ArrayInterface& dest) const;
private:
    MetalContext* context;
    MTL::Buffer* buffer;
    size_t size;
    int elementSize;
    bool ownsMemory;
    std::string name;
};

} // namespace OpenMM

#endif /*OPENMM_METALARRAY_H_*/
