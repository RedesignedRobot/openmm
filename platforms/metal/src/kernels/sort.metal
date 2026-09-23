KEY_TYPE getValue(DATA_TYPE value) {
    return SORT_KEY;
}

/**
 * Sort a list that is short enough to entirely fit in local memory.  This is executed as
 * a single thread block.
 */
KERNEL void sortShortList(GLOBAL DATA_TYPE* RESTRICT data [[buffer(0)]], constant uint& length [[buffer(1)]], LOCAL DATA_TYPE* dataBuffer [[threadgroup(0)]]) {
    // Load the data into local memory.

    for (uint index = LOCAL_ID; index < length; index += LOCAL_SIZE)
        dataBuffer[index] = data[index];
    SYNC_THREADS;

    // Perform a bitonic sort in local memory.

    for (uint k = 2; k < 2*length; k *= 2) {
        for (uint j = k/2; j > 0; j /= 2) {
            for (uint i = LOCAL_ID; i < length; i += LOCAL_SIZE) {
                uint ixj = i^j;
                if (ixj > i && ixj < length) {
                    DATA_TYPE value1 = dataBuffer[i];
                    DATA_TYPE value2 = dataBuffer[ixj];
                    bool ascending = ((i&k) == 0);
                    for (uint mask = k*2; mask < 2*length; mask *= 2)
                        ascending = ((i&mask) == 0 ? !ascending : ascending);
                    KEY_TYPE lowKey  = (ascending ? getValue(value1) : getValue(value2));
                    KEY_TYPE highKey = (ascending ? getValue(value2) : getValue(value1));
                    if (lowKey > highKey) {
                        dataBuffer[i] = value2;
                        dataBuffer[ixj] = value1;
                    }
                }
            }
            SYNC_THREADS;
        }
    }

    // Write the data back to global memory.

    for (uint index = LOCAL_ID; index < length; index += LOCAL_SIZE)
        data[index] = dataBuffer[index];
}

/**
 * Calculate the minimum and maximum value in the array to be sorted.  This kernel
 * is executed as a single work group.
 */
KERNEL void computeRange(GLOBAL const DATA_TYPE* RESTRICT data [[buffer(0)]], constant uint& length [[buffer(1)]], GLOBAL KEY_TYPE* RESTRICT range [[buffer(2)]],
        constant uint& numBuckets [[buffer(3)]], GLOBAL uint* RESTRICT bucketOffset [[buffer(4)]],
        LOCAL KEY_TYPE* minBuffer [[threadgroup(0)]], LOCAL KEY_TYPE* maxBuffer [[threadgroup(1)]]) {
#if UNIFORM
    KEY_TYPE minimum = MAX_KEY;
    KEY_TYPE maximum = MIN_KEY;

    // Each thread calculates the range of a subset of values.

    for (uint index = LOCAL_ID; index < length; index += LOCAL_SIZE) {
        KEY_TYPE value = getValue(data[index]);
        minimum = min(minimum, value);
        maximum = max(maximum, value);
    }

    // Now reduce them.

    minBuffer[LOCAL_ID] = minimum;
    maxBuffer[LOCAL_ID] = maximum;
    SYNC_THREADS;
    for (uint step = 1; step < LOCAL_SIZE; step *= 2) {
        if (LOCAL_ID+step < LOCAL_SIZE && LOCAL_ID%(2*step) == 0) {
            minBuffer[LOCAL_ID] = min(minBuffer[LOCAL_ID], minBuffer[LOCAL_ID+step]);
            maxBuffer[LOCAL_ID] = max(maxBuffer[LOCAL_ID], maxBuffer[LOCAL_ID+step]);
        }
        SYNC_THREADS;
    }
    minimum = minBuffer[0];
    maximum = maxBuffer[0];
    if (LOCAL_ID == 0) {
        range[0] = minimum;
        range[1] = maximum;
    }
#endif

    // Clear the bucket counters in preparation for the next kernel.

    for (uint index = LOCAL_ID; index < numBuckets; index += LOCAL_SIZE)
        bucketOffset[index] = 0;
}

/**
 * Assign elements to buckets.  This version is optimized for uniformly distributed data.
 */
KERNEL void assignElementsToBuckets(GLOBAL const DATA_TYPE* RESTRICT data, uint length, uint numBuckets, GLOBAL const KEY_TYPE* RESTRICT range,
        GLOBAL uint* RESTRICT bucketOffset, GLOBAL uint* RESTRICT bucketOfElement, GLOBAL uint* RESTRICT offsetInBucket) {
    float minValue = (float) (range[0]);
    float maxValue = (float) (range[1]);
    float bucketWidth = (maxValue-minValue)/numBuckets;
    for (uint index = GLOBAL_ID; index < length; index += GLOBAL_SIZE) {
        float key = (float) getValue(data[index]);
        uint bucketIndex = min((uint) ((key-minValue)/bucketWidth), numBuckets-1);
        offsetInBucket[index] = ATOMIC_ADD(&bucketOffset[bucketIndex], 1u);
        bucketOfElement[index] = bucketIndex;
    }
}

/**
 * Assign elements to buckets.  This version is optimized for non-uniformly distributed data.
 */
KERNEL void assignElementsToBuckets2(GLOBAL const DATA_TYPE* RESTRICT data, uint length, uint numBuckets, GLOBAL const KEY_TYPE* RESTRICT range,
        GLOBAL uint* RESTRICT bucketOffset, GLOBAL uint* RESTRICT bucketOfElement, GLOBAL uint* RESTRICT offsetInBucket) {
    // Load 64 datapoints and sort them to get an estimate of the data distribution.

    LOCAL KEY_TYPE elements[64];
    if (LOCAL_ID < 64) {
        uint index = (uint) (LOCAL_ID*(float) length/64.0f);
        elements[LOCAL_ID] = getValue(data[index]);
    }
    SYNC_THREADS;
    for (uint k = 2; k <= 64; k *= 2) {
        for (uint j = k/2; j > 0; j /= 2) {
            if (LOCAL_ID < 64) {
                uint ixj = LOCAL_ID^j;
                if (ixj > LOCAL_ID) {
                    KEY_TYPE value1 = elements[LOCAL_ID];
                    KEY_TYPE value2 = elements[ixj];
                    bool ascending = (LOCAL_ID&k) == 0;
                    KEY_TYPE lowKey = (ascending ? value1 : value2);
                    KEY_TYPE highKey = (ascending ? value2 : value1);
                    if (lowKey > highKey) {
                        elements[LOCAL_ID] = value2;
                        elements[ixj] = value1;
                    }
                }
            }
            SYNC_THREADS;
        }
    }

    // Create a function composed of linear segments mapping data values to bucket indices.

    LOCAL float segmentLowerBound[9];
    LOCAL float segmentBaseIndex[9];
    LOCAL float segmentIndexScale[9];
    if (LOCAL_ID == 0) {
        segmentLowerBound[0] = elements[0]-0.2f*(elements[5]-elements[0]);
        segmentLowerBound[1] = elements[5];
        segmentLowerBound[2] = elements[10];
        segmentLowerBound[3] = elements[20];
        segmentLowerBound[4] = elements[30];
        segmentLowerBound[5] = elements[40];
        segmentLowerBound[6] = elements[50];
        segmentLowerBound[7] = elements[60];
        segmentLowerBound[8] = elements[63]+0.2f*(elements[63]-elements[58]);
        segmentBaseIndex[0] = numBuckets/16;
        segmentBaseIndex[1] = 3*numBuckets/16;
        segmentBaseIndex[2] = 5*numBuckets/16;
        segmentBaseIndex[3] = 7*numBuckets/16;
        segmentBaseIndex[4] = 9*numBuckets/16;
        segmentBaseIndex[5] = 11*numBuckets/16;
        segmentBaseIndex[6] = 13*numBuckets/16;
        segmentBaseIndex[7] = 15*numBuckets/16;
        segmentBaseIndex[8] = numBuckets;
        for (int i = 0; i < 8; i++)
            if (segmentLowerBound[i+1] == segmentLowerBound[i])
                segmentIndexScale[i] = 0;
            else
                segmentIndexScale[i] = (segmentBaseIndex[i+1]-segmentBaseIndex[i])/(segmentLowerBound[i+1]-segmentLowerBound[i]);
    }
    SYNC_THREADS;

    // Assign elements to buckets.

    for (uint index = GLOBAL_ID; index < length; index += GLOBAL_SIZE) {
        float key = (float) getValue(data[index]);
        int segment;
        for (segment = 0; segment < 7 && key > segmentLowerBound[segment+1]; segment++)
            ;
        // Converting a negative float to uint is undefined, so clamp before converting.
        uint bucketIndex = (uint) max(0.0f, segmentBaseIndex[segment]+(key-segmentLowerBound[segment])*segmentIndexScale[segment]);
        bucketIndex = min(bucketIndex, numBuckets-1);
        offsetInBucket[index] = ATOMIC_ADD(&bucketOffset[bucketIndex], 1u);
        bucketOfElement[index] = bucketIndex;
    }
}

/**
 * Sum the bucket sizes to compute the start position of each bucket.  This kernel
 * is executed as a single work group.
 */
KERNEL void computeBucketPositions(constant uint& numBuckets [[buffer(0)]], GLOBAL uint* RESTRICT bucketOffset [[buffer(1)]], LOCAL uint* buffer [[threadgroup(0)]]) {
    uint globalOffset = 0;
    for (uint startBucket = 0; startBucket < numBuckets; startBucket += LOCAL_SIZE) {
        // Load the bucket sizes into local memory.

        uint globalIndex = startBucket+LOCAL_ID;
        SYNC_THREADS;
        buffer[LOCAL_ID] = (globalIndex < numBuckets ? bucketOffset[globalIndex] : 0);
        SYNC_THREADS;

        // Perform a parallel prefix sum.

        for (uint step = 1; step < LOCAL_SIZE; step *= 2) {
            uint add = (LOCAL_ID >= step ? buffer[LOCAL_ID-step] : 0);
            SYNC_THREADS;
            buffer[LOCAL_ID] += add;
            SYNC_THREADS;
        }

        // Write the results back to global memory.

        if (globalIndex < numBuckets)
            bucketOffset[globalIndex] = buffer[LOCAL_ID]+globalOffset;
        globalOffset += buffer[LOCAL_SIZE-1];
    }
}

/**
 * Copy the input data into the buckets for sorting.
 */
KERNEL void copyDataToBuckets(GLOBAL const DATA_TYPE* RESTRICT data, GLOBAL DATA_TYPE* RESTRICT buckets, uint length, GLOBAL const uint* RESTRICT bucketOffset,
        GLOBAL const uint* RESTRICT bucketOfElement, GLOBAL const uint* RESTRICT offsetInBucket) {
    for (uint index = GLOBAL_ID; index < length; index += GLOBAL_SIZE) {
        DATA_TYPE element = data[index];
        uint bucketIndex = bucketOfElement[index];
        uint offset = (bucketIndex == 0 ? 0 : bucketOffset[bucketIndex-1]);
        buckets[offset+offsetInBucket[index]] = element;
    }
}

/**
 * Sort the data in each bucket.
 */
KERNEL void sortBuckets(GLOBAL DATA_TYPE* RESTRICT data [[buffer(0)]], GLOBAL const DATA_TYPE* RESTRICT buckets [[buffer(1)]], constant uint& numBuckets [[buffer(2)]],
        GLOBAL const uint* RESTRICT bucketOffset [[buffer(3)]], LOCAL DATA_TYPE* buffer [[threadgroup(0)]]) {
    for (uint index = GROUP_ID; index < numBuckets; index += NUM_GROUPS) {
        uint startIndex = (index == 0 ? 0 : bucketOffset[index-1]);
        uint endIndex = bucketOffset[index];
        uint length = endIndex-startIndex;
        if (length <= LOCAL_SIZE) {
            // Load the data into local memory.

            if (LOCAL_ID < length)
                buffer[LOCAL_ID] = buckets[startIndex+LOCAL_ID];
            else
                buffer[LOCAL_ID] = MAX_VALUE;
            SYNC_THREADS;

            // Perform a bitonic sort in local memory.

            for (uint k = 2; k <= LOCAL_SIZE; k *= 2) {
                for (uint j = k/2; j > 0; j /= 2) {
                    uint ixj = LOCAL_ID^j;
                    if (ixj > LOCAL_ID) {
                        DATA_TYPE value1 = buffer[LOCAL_ID];
                        DATA_TYPE value2 = buffer[ixj];
                        bool ascending = (LOCAL_ID&k) == 0;
                        KEY_TYPE lowKey = (ascending ? getValue(value1) : getValue(value2));
                        KEY_TYPE highKey = (ascending ? getValue(value2) : getValue(value1));
                        if (lowKey > highKey) {
                            buffer[LOCAL_ID] = value2;
                            buffer[ixj] = value1;
                        }
                    }
                    SYNC_THREADS;
                }
            }

            // Write the data to the sorted array.

            if (LOCAL_ID < length)
                data[startIndex+LOCAL_ID] = buffer[LOCAL_ID];
        }
        else {
            // Copy the bucket data over to the output array.

            for (uint i = LOCAL_ID; i < length; i += LOCAL_SIZE)
                data[startIndex+i] = buckets[startIndex+i];
            SYNC_THREADS;

            // Perform a bitonic sort in global memory.

            for (uint k = 2; k < 2*length; k *= 2) {
                for (uint j = k/2; j > 0; j /= 2) {
                    for (uint i = LOCAL_ID; i < length; i += LOCAL_SIZE) {
                        uint ixj = i^j;
                        if (ixj > i && ixj < length) {
                            DATA_TYPE value1 = data[startIndex+i];
                            DATA_TYPE value2 = data[startIndex+ixj];
                            bool ascending = ((i&k) == 0);
                            for (uint mask = k*2; mask < 2*length; mask *= 2)
                                ascending = ((i&mask) == 0 ? !ascending : ascending);
                            KEY_TYPE lowKey  = (ascending ? getValue(value1) : getValue(value2));
                            KEY_TYPE highKey = (ascending ? getValue(value2) : getValue(value1));
                            if (lowKey > highKey) {
                                data[startIndex+i] = value2;
                                data[startIndex+ixj] = value1;
                            }
                        }
                    }
                    SYNC_THREADS;
                }
            }
        }
    }
}
