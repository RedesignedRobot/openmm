/**
 * Find a bounding box for the atoms in each block.
 */
KERNEL void findBlockBounds(int numAtoms, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const real4* RESTRICT posq, GLOBAL real4* RESTRICT blockCenter, GLOBAL real4* RESTRICT blockBoundingBox, GLOBAL int* RESTRICT rebuildNeighborList,
        GLOBAL real2* RESTRICT blockSizeRange) {
    int index = GLOBAL_ID;
    int base = index*TILE_SIZE;
    real minSize = 1e38f, maxSize = 0;
    while (base < numAtoms) {
        real4 pos = posq[base];
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_POS(pos)
#endif
        real4 minPos = pos;
        real4 maxPos = pos;
        int last = min(base+TILE_SIZE, numAtoms);
        for (int i = base+1; i < last; i++) {
            pos = posq[i];
#ifdef USE_PERIODIC
            real4 center = 0.5f*(maxPos+minPos);
            APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)
#endif
            minPos = min(minPos, pos);
            maxPos = max(maxPos, pos);
        }
        real4 blockSize = 0.5f*(maxPos-minPos);
        real4 center = 0.5f*(maxPos+minPos);
        center.w = 0;
        for (int i = base; i < last; i++) {
            real4 delta = posq[i]-center;
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(delta)
#endif
            center.w = max(center.w, delta.x*delta.x+delta.y*delta.y+delta.z*delta.z);
        }
        center.w = sqrt(center.w);
        blockBoundingBox[index] = blockSize;
        blockCenter[index] = center;
        real totalSize = blockSize.x+blockSize.y+blockSize.z;
        minSize = min(minSize, totalSize);
        maxSize = max(maxSize, totalSize);
        index += GLOBAL_SIZE;
        base = index*TILE_SIZE;
    }

    // Record the range of sizes seen by threads in this block.

    LOCAL real minBuffer[64], maxBuffer[64];
    minBuffer[LOCAL_ID] = minSize;
    maxBuffer[LOCAL_ID] = maxSize;
    SYNC_THREADS;
    for (int step = 1; step < 64; step *= 2) {
        if (LOCAL_ID+step < 64 && LOCAL_ID%(2*step) == 0) {
            minBuffer[LOCAL_ID] = min(minBuffer[LOCAL_ID], minBuffer[LOCAL_ID+step]);
            maxBuffer[LOCAL_ID] = max(maxBuffer[LOCAL_ID], maxBuffer[LOCAL_ID+step]);
        }
        SYNC_THREADS;
    }
    if (LOCAL_ID == 0)
        blockSizeRange[GROUP_ID] = make_real2(minBuffer[0], maxBuffer[0]);
    if (GLOBAL_ID == 0)
        rebuildNeighborList[0] = 0;
}

KERNEL void computeSortKeys(GLOBAL const real4* RESTRICT blockBoundingBox, GLOBAL unsigned int* RESTRICT sortedBlocks, GLOBAL real2* RESTRICT blockSizeRange, int numSizes) {
    // Find the total range of sizes recorded by all blocks.

    LOCAL real2 sizeRange;
    if (LOCAL_ID == 0) {
        sizeRange = blockSizeRange[0];
        for (int i = 1; i < numSizes; i++) {
            real2 size = blockSizeRange[i];
            if (size.x > 0)
                sizeRange.x = min(sizeRange.x, size.x);
            sizeRange.y = max(sizeRange.y, size.y);
        }
        sizeRange.x = LOG(sizeRange.x);
        sizeRange.y = LOG(sizeRange.y);
    }
    SYNC_THREADS;

    // Sort keys store the bin in the high order part and the block in the low
    // order part.

    int numSizeBins = 20;
    real scale = numSizeBins/(sizeRange.y-sizeRange.x);
    for (unsigned int i = GLOBAL_ID; i < NUM_BLOCKS; i += GLOBAL_SIZE) {
        real4 box = blockBoundingBox[i];
        real size = LOG(box.x+box.y+box.z);
        int bin = (int) ((size-sizeRange.x)*scale);
        bin = max(0, min(bin, numSizeBins-1));
        sortedBlocks[i] = (((unsigned int) bin)<<BIN_SHIFT) + i;
    }
}

/**
 * Sort the data about bounding boxes so it can be accessed more efficiently in the next kernel.
 */
KERNEL void sortBoxData(GLOBAL const unsigned int* RESTRICT sortedBlocks, GLOBAL const real4* RESTRICT blockCenter,
        GLOBAL const real4* RESTRICT blockBoundingBox, GLOBAL real4* RESTRICT sortedBlockCenter,
        GLOBAL real4* RESTRICT sortedBlockBoundingBox, GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT oldPositions,
        GLOBAL unsigned int* RESTRICT interactionCount, GLOBAL int* RESTRICT rebuildNeighborList, int forceRebuild
#ifdef USE_LARGE_BLOCKS
        , GLOBAL real4* RESTRICT largeBlockCenter, GLOBAL real4* RESTRICT largeBlockBoundingBox, real4 periodicBoxSize,
        real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
#endif
        ) {
    for (int i = GLOBAL_ID; i < NUM_BLOCKS; i += GLOBAL_SIZE) {
        unsigned int index = sortedBlocks[i] & BLOCK_INDEX_MASK;
        sortedBlockCenter[i] = blockCenter[index];
        sortedBlockBoundingBox[i] = blockBoundingBox[index];

#ifdef USE_LARGE_BLOCKS
        // Compute the sizes of large blocks (composed of 32 regular blocks) starting from each block.

        real4 minPos = blockCenter[index]-blockBoundingBox[index];
        real4 maxPos = blockCenter[index]+blockBoundingBox[index];
        int last = min(i+32, NUM_BLOCKS);
        for (int j = i+1; j < last; j++) {
            unsigned int index2 = sortedBlocks[j] & BLOCK_INDEX_MASK;
            real4 blockPos = blockCenter[index2];
            real4 width = blockBoundingBox[index2];
#ifdef USE_PERIODIC
            real4 center = 0.5f*(maxPos+minPos);
            APPLY_PERIODIC_TO_POS_WITH_CENTER(blockPos, center)
#endif
            minPos = min(minPos, blockPos-width);
            maxPos = max(maxPos, blockPos+width);
        }
        largeBlockCenter[i] = 0.5f*(maxPos+minPos);
        largeBlockBoundingBox[i] = 0.5f*(maxPos-minPos);
#endif
    }

    // Also check whether any atom has moved enough so that we really need to rebuild the neighbor list.

    bool rebuild = forceRebuild;
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        real4 delta = oldPositions[i]-posq[i];
        if (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z > 0.25f*PADDING*PADDING)
            rebuild = true;
    }
    if (rebuild) {
        rebuildNeighborList[0] = 1;
        interactionCount[0] = 0;
    }
}

#define BUFFER_SIZE 256

/**
 * Build the neighbor list.  Each simdgroup processes one block at a time, and all the
 * threadgroup memory a simdgroup touches is private to it, so simdgroup barriers are
 * sufficient for synchronization.
 */
KERNEL void findBlocksWithInteractions(real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL unsigned int* RESTRICT interactionCount, GLOBAL int* RESTRICT interactingTiles, GLOBAL unsigned int* RESTRICT interactingAtoms,
        GLOBAL const real4* RESTRICT posq, unsigned int maxTiles, unsigned int startBlockIndex, unsigned int numBlocks, GLOBAL unsigned int* RESTRICT sortedBlocks,
        GLOBAL const real4* RESTRICT sortedBlockCenter, GLOBAL const real4* RESTRICT sortedBlockBoundingBox,
        GLOBAL const unsigned int* RESTRICT exclusionIndices, GLOBAL const unsigned int* RESTRICT exclusionRowIndices, GLOBAL real4* RESTRICT oldPositions,
        GLOBAL const int* RESTRICT rebuildNeighborList
#ifdef USE_LARGE_BLOCKS
        , GLOBAL real4* RESTRICT largeBlockCenter, GLOBAL real4* RESTRICT largeBlockBoundingBox
#endif
        ) {

    if (rebuildNeighborList[0] == 0)
        return; // The neighbor list doesn't need to be rebuilt.

    const int indexInWarp = LOCAL_ID%32;
    const int warpStart = LOCAL_ID-indexInWarp;
    const int totalWarps = GLOBAL_SIZE/32;
    const int warpIndex = GLOBAL_ID/32;
    LOCAL int workgroupBuffer[BUFFER_SIZE*(GROUP_SIZE/32)];
    LOCAL int warpExclusions[MAX_EXCLUSIONS*(GROUP_SIZE/32)];
    LOCAL real3 posBuffer[GROUP_SIZE];
    LOCAL unsigned int workgroupTileIndex[GROUP_SIZE/32];
    LOCAL bool includeBlockFlags[GROUP_SIZE];
    LOCAL short2 atomCountBuffer[GROUP_SIZE];
    LOCAL int* buffer = workgroupBuffer+BUFFER_SIZE*(warpStart/32);
    LOCAL int* exclusionsForX = warpExclusions+MAX_EXCLUSIONS*(warpStart/32);
    LOCAL unsigned int* tileStartIndex = workgroupTileIndex+(warpStart/32);
#ifdef USE_LARGE_BLOCKS
    LOCAL bool largeBlockFlags[GROUP_SIZE];
#endif

    // Loop over blocks.

    for (int block1 = startBlockIndex+warpIndex; block1 < startBlockIndex+numBlocks; block1 += totalWarps) {
        // Load data for this block.  Note that all threads in a warp are processing the same block.

        int x = sortedBlocks[block1] & BLOCK_INDEX_MASK;
        real4 blockCenterX = sortedBlockCenter[block1];
        real4 blockSizeX = sortedBlockBoundingBox[block1];
        int neighborsInBuffer = 0;
        real3 pos1 = posq[x*TILE_SIZE+indexInWarp].xyz;
#ifdef USE_PERIODIC
        const bool singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= PADDED_CUTOFF &&
                                         0.5f*periodicBoxSize.y-blockSizeX.y >= PADDED_CUTOFF &&
                                         0.5f*periodicBoxSize.z-blockSizeX.z >= PADDED_CUTOFF);
        if (singlePeriodicCopy) {
            // The box is small enough that we can just translate all the atoms into a single periodic
            // box, then skip having to apply periodic boundary conditions later.

            APPLY_PERIODIC_TO_POS_WITH_CENTER(pos1, blockCenterX)
        }
#endif
        posBuffer[LOCAL_ID] = pos1;

        // Load exclusion data for block x.

        const int exclusionStart = exclusionRowIndices[x];
        const int exclusionEnd = exclusionRowIndices[x+1];
        const int numExclusions = exclusionEnd-exclusionStart;
        for (int j = indexInWarp; j < numExclusions; j += 32)
            exclusionsForX[j] = exclusionIndices[exclusionStart+j];
        SYNC_WARPS;

        // Loop over atom blocks to search for neighbors.  The threads in a warp compare block1 against 32
        // other blocks in parallel.

#ifdef USE_LARGE_BLOCKS
        int loadedLargeBlocks = 0;
#endif
        for (int block2Base = block1+1; block2Base < NUM_BLOCKS; block2Base += 32) {
#ifdef USE_LARGE_BLOCKS
            if (loadedLargeBlocks == 0) {
                // Check the next set of large blocks.

                int largeBlockIndex = block2Base + 32*indexInWarp;
                bool includeLargeBlock = false;
                if (largeBlockIndex < NUM_BLOCKS) {
                    real4 largeCenter = largeBlockCenter[largeBlockIndex];
                    real4 largeSize = largeBlockBoundingBox[largeBlockIndex];
                    real4 blockDelta = blockCenterX-largeCenter;
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
                    blockDelta.x = max((real) 0, fabs(blockDelta.x)-blockSizeX.x-largeSize.x);
                    blockDelta.y = max((real) 0, fabs(blockDelta.y)-blockSizeX.y-largeSize.y);
                    blockDelta.z = max((real) 0, fabs(blockDelta.z)-blockSizeX.z-largeSize.z);
                    includeLargeBlock = (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < PADDED_CUTOFF_SQUARED);
#ifdef TRICLINIC
                    // The calculation to find the nearest periodic copy is only guaranteed to work if the nearest copy is less than half a box width away.
                    // If there's any possibility we might have missed it, do a detailed check.

                    if (periodicBoxSize.z/2-blockSizeX.z-largeSize.z < PADDED_CUTOFF || periodicBoxSize.y/2-blockSizeX.y-largeSize.y < PADDED_CUTOFF)
                        includeLargeBlock = true;
#endif
                }
                largeBlockFlags[LOCAL_ID] = includeLargeBlock;
                loadedLargeBlocks = 32;
                SYNC_WARPS;
            }
            if (!largeBlockFlags[warpStart+32-(loadedLargeBlocks--)]) {
                // None of the next 32 blocks interact with block 1.

                continue;
            }
#endif
            int block2 = block2Base+indexInWarp;
            bool includeBlock2 = (block2 < NUM_BLOCKS);
            if (includeBlock2) {
                real4 blockCenterY = sortedBlockCenter[block2];
                real4 blockSizeY = sortedBlockBoundingBox[block2];
                real4 blockDelta = blockCenterX-blockCenterY;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
                includeBlock2 &= (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < (PADDED_CUTOFF+blockCenterX.w+blockCenterY.w)*(PADDED_CUTOFF+blockCenterX.w+blockCenterY.w));
                blockDelta.x = max((real) 0, fabs(blockDelta.x)-blockSizeX.x-blockSizeY.x);
                blockDelta.y = max((real) 0, fabs(blockDelta.y)-blockSizeX.y-blockSizeY.y);
                blockDelta.z = max((real) 0, fabs(blockDelta.z)-blockSizeX.z-blockSizeY.z);
                includeBlock2 &= (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < PADDED_CUTOFF_SQUARED);
#ifdef TRICLINIC
                // The calculation to find the nearest periodic copy is only guaranteed to work if the nearest copy is less than half a box width away.
                // If there's any possibility we might have missed it, do a detailed check.

                if (periodicBoxSize.z/2-blockSizeX.z-blockSizeY.z < PADDED_CUTOFF || periodicBoxSize.y/2-blockSizeX.y-blockSizeY.y < PADDED_CUTOFF)
                    includeBlock2 = true;
#endif
                if (includeBlock2) {
                    int y = sortedBlocks[block2] & BLOCK_INDEX_MASK;
                    for (int k = 0; k < numExclusions; k++)
                        includeBlock2 &= (exclusionsForX[k] != y);
                }
            }

            // Loop over any blocks we identified as potentially containing neighbors.

            includeBlockFlags[LOCAL_ID] = includeBlock2;
            SYNC_WARPS;
            for (int i = 0; i < TILE_SIZE; i++) {
                while (i < TILE_SIZE && !includeBlockFlags[warpStart+i])
                    i++;
                if (i < TILE_SIZE) {
                    int y = sortedBlocks[block2Base+i] & BLOCK_INDEX_MASK;

                    // Check each atom in block Y for interactions.

                    int atom2 = y*TILE_SIZE+indexInWarp;
                    real3 pos2 = posq[atom2].xyz;
#ifdef USE_PERIODIC
                    if (singlePeriodicCopy)
                        APPLY_PERIODIC_TO_POS_WITH_CENTER(pos2, blockCenterX)
#endif
                    bool interacts = false;
                    if (atom2 < NUM_ATOMS) {
#ifdef USE_PERIODIC
                        if (!singlePeriodicCopy) {
                            for (int j = 0; j < TILE_SIZE; j++) {
                                real3 delta = pos2-posBuffer[warpStart+j];
                                APPLY_PERIODIC_TO_DELTA(delta)
                                interacts |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED);
                            }
                        }
                        else {
#endif
                            for (int j = 0; j < TILE_SIZE; j++) {
                                real3 delta = pos2-posBuffer[warpStart+j];
                                interacts |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED);
                            }
#ifdef USE_PERIODIC
                        }
#endif
                    }

                    // Do a prefix sum to compact the list of atoms.

                    atomCountBuffer[LOCAL_ID].x = (interacts ? 1 : 0);
                    SYNC_WARPS;
                    int whichBuffer = 0;
                    for (int offset = 1; offset < TILE_SIZE; offset *= 2) {
                        if (whichBuffer == 0)
                            atomCountBuffer[LOCAL_ID].y = (indexInWarp < offset ? atomCountBuffer[LOCAL_ID].x : atomCountBuffer[LOCAL_ID].x+atomCountBuffer[LOCAL_ID-offset].x);
                        else
                            atomCountBuffer[LOCAL_ID].x = (indexInWarp < offset ? atomCountBuffer[LOCAL_ID].y : atomCountBuffer[LOCAL_ID].y+atomCountBuffer[LOCAL_ID-offset].y);
                        whichBuffer = 1-whichBuffer;
                        SYNC_WARPS;
                    }

                    // Add any interacting atoms to the buffer.

                    if (interacts)
                        buffer[neighborsInBuffer+atomCountBuffer[LOCAL_ID].y-1] = atom2;
                    neighborsInBuffer += atomCountBuffer[warpStart+TILE_SIZE-1].y;
                    if (neighborsInBuffer > BUFFER_SIZE-TILE_SIZE) {
                        // Store the new tiles to memory.

                        unsigned int tilesToStore = neighborsInBuffer/TILE_SIZE;
                        if (indexInWarp == 0)
                            *tileStartIndex = ATOMIC_ADD(interactionCount, tilesToStore);
                        SYNC_WARPS;
                        unsigned int newTileStartIndex = *tileStartIndex;
                        if (newTileStartIndex+tilesToStore <= maxTiles) {
                            if (indexInWarp < tilesToStore)
                                interactingTiles[newTileStartIndex+indexInWarp] = x;
                            for (int j = 0; j < tilesToStore; j++)
                                interactingAtoms[(newTileStartIndex+j)*TILE_SIZE+indexInWarp] = buffer[indexInWarp+j*TILE_SIZE];
                        }
                        if (indexInWarp+TILE_SIZE*tilesToStore < BUFFER_SIZE)
                            buffer[indexInWarp] = buffer[indexInWarp+TILE_SIZE*tilesToStore];
                        neighborsInBuffer -= TILE_SIZE*tilesToStore;
                   }
                }
                else {
                    SYNC_WARPS;
                }
            }
        }

        // If we have a partially filled buffer,  store it to memory.

        if (neighborsInBuffer > 0) {
            unsigned int tilesToStore = (neighborsInBuffer+TILE_SIZE-1)/TILE_SIZE;
            if (indexInWarp == 0)
                *tileStartIndex = ATOMIC_ADD(interactionCount, tilesToStore);
            SYNC_WARPS;
            unsigned int newTileStartIndex = *tileStartIndex;
            if (newTileStartIndex+tilesToStore <= maxTiles) {
                if (indexInWarp < tilesToStore)
                    interactingTiles[newTileStartIndex+indexInWarp] = x;
                for (int j = 0; j < tilesToStore; j++)
                    interactingAtoms[(newTileStartIndex+j)*TILE_SIZE+indexInWarp] = (indexInWarp+j*TILE_SIZE < neighborsInBuffer ? buffer[indexInWarp+j*TILE_SIZE] : NUM_ATOMS);
            }
        }
    }

    // Record the positions the neighbor list is based on.

    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE)
        oldPositions[i] = posq[i];
}
