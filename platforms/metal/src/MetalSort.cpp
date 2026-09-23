/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2010-2025 Stanford University and the Authors.      *
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
#include "MetalSort.h"
#include "MetalKernel.h"
#include "MetalKernelSources.h"
#include <Metal/Metal.hpp>
#include <algorithm>
#include <map>
#include <string>

using namespace OpenMM;
using namespace std;

MetalSort::MetalSort(MetalContext& context, ComputeSortImpl::SortTrait* trait, unsigned int length, bool uniform) :
        trait(trait), dataLength(length) {
    // Create kernels.

    map<string, string> replacements;
    replacements["DATA_TYPE"] = trait->getDataType();
    replacements["KEY_TYPE"] =  trait->getKeyType();
    replacements["SORT_KEY"] = trait->getSortKey();
    replacements["MIN_KEY"] = trait->getMinKey();
    replacements["MAX_KEY"] = trait->getMaxKey();
    replacements["MAX_VALUE"] = trait->getMaxValue();
    replacements["UNIFORM"] = (uniform ? "1" : "0");
    ComputeProgram program = context.compileProgram(context.replaceStrings(MetalKernelSources::sort, replacements));
    shortListKernel = program->createKernel("sortShortList");
    computeRangeKernel = program->createKernel("computeRange");
    assignElementsKernel = program->createKernel(uniform ? "assignElementsToBuckets" : "assignElementsToBuckets2");
    computeBucketPositionsKernel = program->createKernel("computeBucketPositions");
    copyToBucketsKernel = program->createKernel("copyDataToBuckets");
    sortBucketsKernel = program->createKernel("sortBuckets");

    // Work out the work group sizes for various kernels.  Metal reports execution errors
    // asynchronously, so every size must be known to be legal before a kernel is launched.

    unsigned int maxGroupSize = min(256, computeRangeKernel->getMaxBlockSize());
    unsigned int maxPositionsSize = min(256, computeBucketPositionsKernel->getMaxBlockSize());
    int maxSharedMem = context.getDevice()->maxThreadgroupMemoryLength();
    int maxLocalBuffer = (maxSharedMem/trait->getDataSize())/2;
    int maxShortList = min(1024, max(maxLocalBuffer, MetalContext::ThreadBlockSize*context.getNumThreadBlocks()));
    for (rangeKernelSize = 1; rangeKernelSize*2 <= maxGroupSize; rangeKernelSize *= 2)
        ;
    positionsKernelSize = min(rangeKernelSize, maxPositionsSize);
    isShortList = (length <= maxShortList && length*trait->getDataSize() <= maxSharedMem && rangeKernelSize <= shortListKernel->getMaxBlockSize());
    sortKernelSize = (isShortList ? rangeKernelSize : rangeKernelSize/2);
    if (rangeKernelSize > length)
        rangeKernelSize = length;
    if (sortKernelSize > maxLocalBuffer)
        sortKernelSize = maxLocalBuffer;
    if (!isShortList && sortKernelSize > sortBucketsKernel->getMaxBlockSize())
        throw OpenMMException("MetalSort: sortBuckets kernel cannot use a block size of "+context.intToString(sortKernelSize));
    unsigned int targetBucketSize = sortKernelSize/2;
    unsigned int numBuckets = length/targetBucketSize;
    if (numBuckets < 1)
        numBuckets = 1;
    if (positionsKernelSize > numBuckets)
        positionsKernelSize = numBuckets;

    // Create workspace arrays.

    dataRange.initialize(context, 2, trait->getKeySize(), "sortDataRange");
    bucketOffset.initialize<unsigned int>(context, numBuckets, "bucketOffset");
    bucketOfElement.initialize<unsigned int>(context, length, "bucketOfElement");
    offsetInBucket.initialize<unsigned int>(context, length, "offsetInBucket");
    buckets.initialize(context, length, trait->getDataSize(), "buckets");

    // Set the arguments.  Only the array being sorted changes from one call to the next.

    shortListKernel->addArg();
    shortListKernel->addArg(dataLength);
    dynamic_cast<MetalKernel&>(*shortListKernel).setThreadgroupMemoryArg(0, dataLength*trait->getDataSize());
    computeRangeKernel->addArg();
    computeRangeKernel->addArg(dataLength);
    computeRangeKernel->addArg(dataRange);
    computeRangeKernel->addArg(numBuckets);
    computeRangeKernel->addArg(bucketOffset);
    dynamic_cast<MetalKernel&>(*computeRangeKernel).setThreadgroupMemoryArg(0, rangeKernelSize*trait->getKeySize());
    dynamic_cast<MetalKernel&>(*computeRangeKernel).setThreadgroupMemoryArg(1, rangeKernelSize*trait->getKeySize());
    assignElementsKernel->addArg();
    assignElementsKernel->addArg(dataLength);
    assignElementsKernel->addArg(numBuckets);
    assignElementsKernel->addArg(dataRange);
    assignElementsKernel->addArg(bucketOffset);
    assignElementsKernel->addArg(bucketOfElement);
    assignElementsKernel->addArg(offsetInBucket);
    computeBucketPositionsKernel->addArg(numBuckets);
    computeBucketPositionsKernel->addArg(bucketOffset);
    dynamic_cast<MetalKernel&>(*computeBucketPositionsKernel).setThreadgroupMemoryArg(0, positionsKernelSize*sizeof(unsigned int));
    copyToBucketsKernel->addArg();
    copyToBucketsKernel->addArg(buckets);
    copyToBucketsKernel->addArg(dataLength);
    copyToBucketsKernel->addArg(bucketOffset);
    copyToBucketsKernel->addArg(bucketOfElement);
    copyToBucketsKernel->addArg(offsetInBucket);
    sortBucketsKernel->addArg();
    sortBucketsKernel->addArg(buckets);
    sortBucketsKernel->addArg(numBuckets);
    sortBucketsKernel->addArg(bucketOffset);
    dynamic_cast<MetalKernel&>(*sortBucketsKernel).setThreadgroupMemoryArg(0, sortKernelSize*trait->getDataSize());
}

MetalSort::~MetalSort() {
    delete trait;
}

void MetalSort::sort(ArrayInterface& data) {
    if (data.getSize() != dataLength || data.getElementSize() != trait->getDataSize())
        throw OpenMMException("MetalSort called with different data size");
    if (data.getSize() == 0)
        return;
    if (isShortList) {
        // We can use a simpler sort kernel that does the entire operation in one kernel.

        shortListKernel->setArg(0, data);
        shortListKernel->execute(sortKernelSize, sortKernelSize);
        return;
    }

    // Compute the range of data values.

    computeRangeKernel->setArg(0, data);
    computeRangeKernel->execute(rangeKernelSize, rangeKernelSize);

    // Assign array elements to buckets.

    assignElementsKernel->setArg(0, data);
    assignElementsKernel->execute(dataLength);

    // Compute the position of each bucket.

    computeBucketPositionsKernel->execute(positionsKernelSize, positionsKernelSize);

    // Copy the data into the buckets.

    copyToBucketsKernel->setArg(0, data);
    copyToBucketsKernel->execute(dataLength);

    // Sort each bucket.

    sortBucketsKernel->setArg(0, data);
    sortBucketsKernel->execute(((dataLength+sortKernelSize-1)/sortKernelSize)*sortKernelSize, sortKernelSize);
}
