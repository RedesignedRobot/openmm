/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2019 Stanford University and the Authors.           *
 * Portions copyright (c) 2020 Advanced Micro Devices, Inc.                   *
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
#include "MetalKernel.h"
#include "MetalQueue.h"
#include "openmm/internal/AssertionUtilities.h"
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstring>

using namespace OpenMM;
using namespace std;

// Primitive arguments are padded so a host value is never shorter than the Metal type it
// binds to (a float3 occupies 16 bytes), and threadgroup memory lengths must be multiples of 16.
static int roundUp16(int size) {
    return (size+15) & ~15;
}

MetalKernel::MetalKernel(MetalContext& context, MTL::ComputePipelineState* pipeline, const string& name) : context(context), pipeline(pipeline), name(name) {
}

MetalKernel::~MetalKernel() {
    pipeline->release();
}

string MetalKernel::getName() const {
    return name;
}

int MetalKernel::getMaxBlockSize() const {
    return pipeline->maxTotalThreadsPerThreadgroup();
}

void MetalKernel::execute(int threads, int blockSize) {
    if (blockSize == -1)
        blockSize = MetalContext::ThreadBlockSize;
    if (blockSize > getMaxBlockSize())
        throw OpenMMException("Error invoking kernel "+name+": block size "+context.intToString(blockSize)+" exceeds the maximum of "+context.intToString(getMaxBlockSize()));
    int gridSize = min((threads+blockSize-1)/blockSize, context.getNumThreadBlocks());
    if (gridSize <= 0)
        return;
    MetalQueue& queue = context.getCurrentMetalQueue();
    std::lock_guard<std::recursive_mutex> guard(queue.getLock());
    MTL::ComputeCommandEncoder* encoder = queue.getEncoder();
    encoder->setComputePipelineState(pipeline);
    for (int i = 0; i < args.size(); i++) {
        if (args[i].array != NULL)
            encoder->setBuffer(args[i].array->getBuffer(), 0, i);
        else
            encoder->setBytes(args[i].value, args[i].size, i);
    }
    for (int i = 0; i < threadgroupMemory.size(); i++)
        encoder->setThreadgroupMemoryLength(roundUp16(threadgroupMemory[i]), i);
    encoder->dispatchThreadgroups(MTL::Size(gridSize, 1, 1), MTL::Size(blockSize, 1, 1));
}

void MetalKernel::setThreadgroupMemoryArg(int index, int bytes) {
    if (index >= threadgroupMemory.size())
        threadgroupMemory.resize(index+1, 0);
    threadgroupMemory[index] = bytes;
}

void MetalKernel::addArrayArg(ArrayInterface& value) {
    int index = args.size();
    addEmptyArg();
    setArrayArg(index, value);
}

void MetalKernel::addPrimitiveArg(const void* value, int size) {
    int index = args.size();
    addEmptyArg();
    setPrimitiveArg(index, value, size);
}

void MetalKernel::addEmptyArg() {
    Arg arg = {NULL, {0}, sizeof(arg.value)};
    args.push_back(arg);
}

void MetalKernel::setArrayArg(int index, ArrayInterface& value) {
    ASSERT_VALID_INDEX(index, args);
    args[index].array = &context.unwrap(value);
}

void MetalKernel::setPrimitiveArg(int index, const void* value, int size) {
    ASSERT_VALID_INDEX(index, args);
    Arg& arg = args[index];
    if (size > sizeof(arg.value))
        throw OpenMMException("Unsupported value type for kernel argument");
    memset(arg.value, 0, sizeof(arg.value));
    memcpy(arg.value, value, size);
    arg.array = NULL;
    arg.size = roundUp16(size);
}
