/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2019 Stanford University and the Authors.           *
 * Portions copyright (c) 2020 Advanced Micro Devices, Inc.                   *
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

#include "MetalKernel.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/internal/AssertionUtilities.h"
#include <cstring>
#include <vector>

using namespace OpenMM;
using namespace std;

MetalKernel::MetalKernel(MetalContext& context, metalFunction_t kernel, const string& name) : context(context), kernel(kernel), name(name) {
}

string MetalKernel::getName() const {
    return name;
}

int MetalKernel::getMaxBlockSize() const {
    return kernel->maxTotalThreadsPerThreadgroup();
}

void MetalKernel::execute(int threads, int blockSize) {
    int numArgs = arrayArgs.size();
    argPointers.resize(numArgs);
    for (int i = 0; i < numArgs; i++) {
        if (arrayArgs[i] != NULL)
            argPointers[i] = &arrayArgs[i]->getDevicePointer();
        else
            argPointers[i] = &primitiveArgs[i];
    }
    // The float charge spread keeps one slab per SIMD group, and a 64 thread launch leaves too few
    // SIMD groups in flight to hide the slab latency.
    if (blockSize == -1 && name == "gridSpreadCharge" && !context.getUseFixedPointChargeSpreading())
        blockSize = MetalContext::SpreadBlockSize;
    context.executeKernel(kernel, argPointers.data(), threads, blockSize);
}

void MetalKernel::addArrayArg(ArrayInterface& value) {
    int index = arrayArgs.size();
    addEmptyArg();
    setArrayArg(index, value);
}

void MetalKernel::addPrimitiveArg(const void* value, int size) {
    int index = arrayArgs.size();
    addEmptyArg();
    setPrimitiveArg(index, value, size);
}

void MetalKernel::addEmptyArg() {
    primitiveArgs.push_back(make_double4(0, 0, 0, 0));
    arrayArgs.push_back(NULL);
}

void MetalKernel::setArrayArg(int index, ArrayInterface& value) {
    ASSERT_VALID_INDEX(index, arrayArgs);
    arrayArgs[index] = &context.unwrap(value);
}

void MetalKernel::setPrimitiveArg(int index, const void* value, int size) {
    ASSERT_VALID_INDEX(index, primitiveArgs);
    if (size > sizeof(double4))
        throw OpenMMException("Unsupported value type for kernel argument");
    memcpy(&primitiveArgs[index], value, size);
    arrayArgs[index] = NULL;
}
