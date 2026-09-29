#define BUFFER_SIZE 256
// Fewest atoms of block X passing the sphere test for which findBlocksWithInteractions tests all 32 unrolled.
#define MIN_ATOMS_TO_UNROLL 8

#if defined(AMD_RDNA)

typedef unsigned int warpflags;

__device__ inline int warpPopc(warpflags x) {
    return __popc(x);
}

#else

typedef unsigned long long warpflags;

__device__ inline int warpPopc(warpflags x) {
    return __popcll(x);
}

#endif

struct alignas(sizeof(__half) * 4) BoundingBox {
    __device__ BoundingBox(real3 f) {
        // Round up so we'll err on the side of making the box a little too large.
        // This ensures interactions will never be missed.
        v[0] = __float2half_ru((float) f.x);
        v[1] = __float2half_ru((float) f.y);
        v[2] = __float2half_ru((float) f.z);
    }
    __device__ real3 toReal3() const device {
        return make_real3(__half2float(v[0]), __half2float(v[1]), __half2float(v[2]));
    }
private:
    __half v[3];
};

/**
 * Find a bounding box for the atoms in each block.  Each SIMD group handles one block, and each threadgroup of
 * FIND_BLOCK_BOUNDS_GROUP threads records the range of block sizes with one atomic pair, not one per block.
 */
extern "C" __global__ void findBlockBounds(int numAtoms, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        const real4* __restrict__ posq, real4* __restrict__ blockCenter, real4* __restrict__ blockBoundingBox, int* __restrict__ rebuildNeighborList,
        real2* __restrict__ blockSizeRange
#ifdef INDIRECT_REBUILD
        , const real4* __restrict__ oldPositions, unsigned int* __restrict__ displacementFlags, bool forceRebuild
#endif
        ) {
    const int indexInTile = threadIdx.x%TILE_SIZE;
    const int tileInGroup = threadIdx.x/TILE_SIZE;
    const int index = blockIdx.x*(FIND_BLOCK_BOUNDS_GROUP/TILE_SIZE) + tileInGroup;
    const int base = index * TILE_SIZE;
    const bool validBlock = base < numAtoms;

    real4 tPos = posq[base + indexInTile < numAtoms ? base + indexInTile : 0];
#ifdef INDIRECT_REBUILD
    bool moved = forceRebuild;
    if (!forceRebuild && base+indexInTile < numAtoms) {
        real4 displacement = oldPositions[base+indexInTile]-tPos;
        moved = displacement.x*displacement.x + displacement.y*displacement.y + displacement.z*displacement.z > 0.25f*PADDING*PADDING;
    }
    bool tileMoved = simd_any(moved);
    if (indexInTile == 0 && validBlock)
        displacementFlags[index] = tileMoved;
#endif
    real4 pos = make_real4(tPos.x, tPos.y, tPos.z, 0);
#ifdef USE_PERIODIC
    // Wrap every atom to its image nearest the first atom.  For a block narrower than half the box this picks the
    // same images as wrapping one atom at a time around the running center, without 31 dependent steps.

    real4 first = make_real4(SHFL(tPos.x, 0), SHFL(tPos.y, 0), SHFL(tPos.z, 0), 0);
    APPLY_PERIODIC_TO_POS(first)
    APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, first)
#endif
    real4 minPos = make_real4(simd_min(pos.x), simd_min(pos.y), simd_min(pos.z), 0);
    real4 maxPos = make_real4(simd_max(pos.x), simd_max(pos.y), simd_max(pos.z), 0);

    real4 blockSize = 0.5f*(maxPos-minPos);
    real4 center = 0.5f*(maxPos+minPos);
    real4 delta = tPos - center;
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(delta)
#endif
    real tcenter = simd_max(delta.x*delta.x+delta.y*delta.y+delta.z*delta.z);

    // Record the range of sizes.

    __shared__ real groupMinSize[FIND_BLOCK_BOUNDS_GROUP/TILE_SIZE];
    __shared__ real groupMaxSize[FIND_BLOCK_BOUNDS_GROUP/TILE_SIZE];
    if (indexInTile == 0) {
        real totalSize = blockSize.x+blockSize.y+blockSize.z;
        if (validBlock) {
            center.w = SQRT(tcenter);
            blockBoundingBox[index] = blockSize;
            blockCenter[index] = center;
        }
        groupMinSize[tileInGroup] = (validBlock && totalSize > 0 ? totalSize : 1e38f);
        groupMaxSize[tileInGroup] = (validBlock ? totalSize : 0.0f);
    }
    SYNC_THREADS;
    if (threadIdx.x == 0) {
        real minSize = groupMinSize[0];
        real maxSize = groupMaxSize[0];
        for (int i = 1; i < FIND_BLOCK_BOUNDS_GROUP/TILE_SIZE; i++) {
            minSize = min(minSize, groupMinSize[i]);
            maxSize = max(maxSize, groupMaxSize[i]);
        }
        if (minSize < 1e38f)
            atomicMin((device real*) blockSizeRange, minSize);
        atomicMax((device real*) blockSizeRange+1, maxSize);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
        rebuildNeighborList[0] = 0;
}

extern "C" __global__ void computeSortKeys(const real4* __restrict__ blockBoundingBox, unsigned int* __restrict__ sortedBlocks, const real2* __restrict__ blockSizeRange/*, int numSizes*/) {
    // Sort keys store the bin in the high order part and the block in the low
    // order part.

    real2 sizeRange = make_real2(LOG(blockSizeRange->x), LOG(blockSizeRange->y));
    int numSizeBins = 20;
    real scale = numSizeBins/(sizeRange.y-sizeRange.x);
    unsigned int i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i < NUM_BLOCKS) {
        real4 box = blockBoundingBox[i];
        real size = LOG(box.x+box.y+box.z);
        int bin = (size-sizeRange.x)*scale;
        bin = max(0, min(bin, numSizeBins-1));
        sortedBlocks[i] = (((unsigned int) bin)<<BIN_SHIFT) + i;
    }
}

/**
 * Sort the data about bounding boxes so it can be accessed more efficiently in the next kernel.
 */
extern "C" __global__ void sortBoxData(const unsigned int* __restrict__ sortedBlocks, const real4* __restrict__ blockCenter,
        const real4* __restrict__ blockBoundingBox, real4* __restrict__ sortedBlockCenter, BoundingBox* __restrict__ sortedBlockBoundingBox,
#ifdef USE_LARGE_BLOCKS
        real4* __restrict__ largeBlockCenter, BoundingBox* __restrict__ largeBlockBoundingBox, real4 periodicBoxSize,
        real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
#endif
        const real4* __restrict__ posq, const real4* __restrict__ oldPositions,
        unsigned int* __restrict__ interactionCount, int* __restrict__ rebuildNeighborList, bool forceRebuild,
        real2* __restrict__ blockSizeRange) {
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i < NUM_BLOCKS) {
        unsigned int index = sortedBlocks[i] & BLOCK_INDEX_MASK;
        sortedBlockCenter[i] = blockCenter[index];
        sortedBlockBoundingBox[i] = BoundingBox(trimTo3(blockBoundingBox[index]));

#ifdef USE_LARGE_BLOCKS
        // Compute the sizes of large blocks (composed of warpSize regular blocks) starting from each block.

        real4 minPos = blockCenter[index]-blockBoundingBox[index];
        real4 maxPos = blockCenter[index]+blockBoundingBox[index];
        int last = min(i+warpSize, NUM_BLOCKS);
        for (int j = i+1; j < last; j++) {
            unsigned int index2 = sortedBlocks[j] & BLOCK_INDEX_MASK;
            real4 blockPos = blockCenter[index2];
            real4 width = blockBoundingBox[index2];
#ifdef USE_PERIODIC
            real4 center = 0.5f*(maxPos+minPos);
            APPLY_PERIODIC_TO_POS_WITH_CENTER(blockPos, center)
#endif
            minPos = make_real4(min(minPos.x, blockPos.x-width.x), min(minPos.y, blockPos.y-width.y), min(minPos.z, blockPos.z-width.z), 0);
            maxPos = make_real4(max(maxPos.x, blockPos.x+width.x), max(maxPos.y, blockPos.y+width.y), max(maxPos.z, blockPos.z+width.z), 0);
        }
        largeBlockCenter[i] = 0.5f*(maxPos+minPos);
        largeBlockBoundingBox[i] = BoundingBox(trimTo3(0.5f*(maxPos-minPos)));
#endif
    }

    // Initialize min and max for the atomic reduction in findBlockBounds (for the next step)
    if (i == 0) {
        blockSizeRange->x = 1e38;
        blockSizeRange->y = 0;
    }

#ifndef INDIRECT_REBUILD
    // Also check whether any atom has moved enough so that we really need to rebuild the neighbor list.

    bool rebuild = forceRebuild;
    if (i < NUM_ATOMS) {
        real4 delta = oldPositions[i]-posq[i];
        if (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z > 0.25f*PADDING*PADDING)
            rebuild = true;
    }

    if (rebuild) {
        rebuildNeighborList[0] = 1;
        interactionCount[0] = 0;
        interactionCount[1] = 0;
    }
#endif
}

#if MAX_BITS_FOR_PAIRS > 0

__device__ inline
void collectInteractions(PRIVATE unsigned int& interacts, float d, int bit) {
    interacts |= (__float_as_uint(d) >> 31) << bit;
}

#else

// Simplified version that does not collect individual bits (they are not needed without single pairs),
// only sets the flag that there are any interactions.

__device__ inline
void collectInteractions(PRIVATE unsigned int& interacts, float d, int) {
    interacts |= __float_as_uint(d) & (1 << 31);
}

#endif

#if !defined(USE_DOUBLE_PRECISION) && __has_builtin(__builtin_amdgcn_mfma_f32_4x4x1f32)

#define USE_MFMA

using vfloat = __attribute__((__vector_size__(4 * sizeof(float)))) float;

template<int BlockId>
inline __device__
void mfma4x4(const float4& pos1, const float4& pos2, const vfloat& c, unsigned int& interacts) {
    vfloat d;
    d = __builtin_amdgcn_mfma_f32_4x4x1f32(pos1.x, -pos2.x, c, 4, BlockId, 0);
    d = __builtin_amdgcn_mfma_f32_4x4x1f32(pos1.y, -pos2.y, d, 4, BlockId, 0);
    d = __builtin_amdgcn_mfma_f32_4x4x1f32(pos1.z, -pos2.z, d, 4, BlockId, 0);
    d = __builtin_amdgcn_mfma_f32_4x4x1f32(pos1.w,    1.0f, d, 4, BlockId, 0);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        collectInteractions(interacts, d[i], BlockId * 4 + i);
    }
}

#endif

/**
 * Compare the bounding boxes for each pair of atom blocks (comprised of TILE_SIZE atoms each), forming a tile. If the two
 * atom blocks are sufficiently far apart, mark them as non-interacting. There are two stages in the algorithm.
 *
 * STAGE 1:
 *
 * A coarse grained atom block against interacting atom block neighbour list is constructed.
 *
 * Each warp first loads in some block X of interest. Each thread within the warp then loads
 * in a different atom block Y. If Y has exclusions with X, then Y is not processed.  If the bounding boxes
 * of the two atom blocks are within the cutoff distance, then the two atom blocks are considered to be
 * interacting and Y is added to the buffer for X.
 *
 * STAGE 2:
 *
 * A fine grained atom block against interacting atoms neighbour list is constructed.
 *
 * The warp loops over atom blocks Y that were found to (possibly) interact with atom block X.  Each thread
 * in the warp loops over the TILE_SIZE atoms in X and compares their positions to one particular atom from block Y.
 * If it finds one closer than the cutoff distance, the atom is added to the list of atoms interacting with block X.
 * This continues until the buffer fills up, at which point the results are written to global memory.
 *
 * [in] periodicBoxSize        - size of the rectangular periodic box
 * [in] invPeriodicBoxSize     - inverse of the periodic box
 * [in] blockCenter            - the center of each bounding box
 * [in] blockBoundingBox       - bounding box of each atom block
 * [out] interactionCount      - total number of tiles that have interactions
 * [out] interactingTiles      - set of blocks that have interactions
 * [out] interactingAtoms      - a list of atoms that interact with each atom block
 * [in] posq                   - x,y,z coordinates of each atom and charge q
 * [in] maxTiles               - maximum number of tiles to process, used for multi-GPUs
 * [in] startBlockIndex        - first block to process, used for multi-GPUs,
 * [in] numBlocks              - total number of atom blocks
 * [in] sortedBlocks           - a sorted list of atom blocks based on volume
 * [in] sortedBlockCenter      - sorted centers, duplicated for fast access to avoid indexing
 * [in] sortedBlockBoundingBox - sorted bounding boxes, duplicated for fast access
 * [in] exclusionIndices       - maps into exclusionRowIndices with the starting position for a given atom
 * [in] exclusionRowIndices    - stores the a continuous list of exclusions
 *           eg: block 0 is excluded from atom 3,5,6
 *               block 1 is excluded from atom 3,4
 *               block 2 is excluded from atom 1,3,5,6
 *              exclusionIndices[0][3][5][8]
 *           exclusionRowIndices[3][5][6][3][4][1][3][5][6]
 *                         index 0  1  2  3  4  5  6  7  8
 * [out] oldPos                - stores the positions of the atoms in which this neighbourlist was built on
 *                             - this is used to decide when to rebuild a neighbourlist
 * [in] rebuildNeighbourList   - whether or not to execute this kernel
 *
 */
extern "C" __global__ __launch_bounds__(GROUP_SIZE) void findBlocksWithInteractions(real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        unsigned int* __restrict__ interactionCount, int* __restrict__ interactingTiles, unsigned int* __restrict__ interactingAtoms,
        int2* __restrict__ singlePairs, const real4* __restrict__ posq, unsigned int maxTiles, unsigned int maxSinglePairs,
        unsigned int startBlockIndex, unsigned int numBlocks, const unsigned int* __restrict__ sortedBlocks, const real4* __restrict__ sortedBlockCenter,
        const BoundingBox* __restrict__ sortedBlockBoundingBox,
#ifdef USE_LARGE_BLOCKS
        const real4* __restrict__ largeBlockCenter, const BoundingBox* __restrict__ largeBlockBoundingBox,
#endif
        const unsigned int* __restrict__ exclusionIndices, const unsigned int* __restrict__ exclusionRowIndices,
        real4* __restrict__ oldPositions, const int* __restrict__ rebuildNeighborList
#ifdef ROLE_ORDER
        , const int* __restrict__ atomRoles
#endif
#ifdef FB_PROBE_COUNT
        , unsigned int* __restrict__ fbCounters
#endif
        ) {

    if (rebuildNeighborList[0] == 0)
        return; // The neighbor list doesn't need to be rebuilt.
#ifdef FB_PROBE_COUNT
    // Scratch probe: per-warp counts, added once at the end.  All increments sit at warp-uniform points.
    unsigned int cChunks = 0, cLargeSkip = 0, cCenterSkip = 0, cCand = 0, cRoleSkip = 0, cAtomTests = 0;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        atomicAdd(&fbCounters[0], 1u);
#endif
#if defined(FB_PROBE_BBOX2) || defined(FB_PROBE_ATOM2)
    unsigned int fbDummy = 0;
#endif

    // Starting from ROCm 7 warpSize is no longer constexpr. Redefine as a local value for using it
    // in sizes of __shared__ arrays:
#if defined(AMD_RDNA)
    constexpr int warpSize = 32;
#else
    constexpr int warpSize = 64;
#endif
    constexpr int tilesPerWarp = warpSize/TILE_SIZE;
    constexpr int warpsPerBlock = GROUP_SIZE/warpSize;
    const int indexInWarp = threadIdx.x%warpSize;
    const int indexInTile = threadIdx.x%TILE_SIZE;
    const int tileInWarp = tilesPerWarp == 1 ? 0 : (threadIdx.x/TILE_SIZE)%tilesPerWarp;
    const int warpInBlock = warpsPerBlock == 1 ? 0 : threadIdx.x/warpSize;
    const int warpIndex = blockIdx.x*warpsPerBlock + (warpsPerBlock == 1 ? 0 : warpInBlock);
    const warpflags warpMask = (static_cast<warpflags>(1)<<indexInWarp)-1;

    __shared__ int workgroupBuffer[BUFFER_SIZE*warpsPerBlock];
    __shared__ real4 workgroupPosBuffer[TILE_SIZE*warpsPerBlock];
    __shared__ int workgroupExclusions[MAX_EXCLUSIONS*warpsPerBlock];
    __shared__ int workgroupBlock2Buffer[(warpSize+1)*warpsPerBlock];
#ifdef FB_PROBE_ATOM2_INDEP
    __shared__ real4 workgroupPosBufferDup[TILE_SIZE*warpsPerBlock];
    threadgroup real4* posBufferDup = workgroupPosBufferDup+TILE_SIZE*warpInBlock;
#define FB_DUP_BUF posBufferDup
#else
#define FB_DUP_BUF posBuffer
#endif
#ifdef FB_PROBE_HALF
    // Scratch probe: block X positions relative to its center as half4 (w = half the squared length), so the
    // 32x32 test reads 8 bytes per atom instead of 16.  Rounding is under 2.5e-4 nm per coordinate at |p| < 1 nm,
    // covered by widening the padded cutoff 0.001 nm: the list is a superset and the force kernel applies the cutoff.
    __shared__ half4 workgroupPosBufferH[TILE_SIZE*warpsPerBlock];
    threadgroup half4* posBufferH = workgroupPosBufferH+TILE_SIZE*warpInBlock;
#endif

    threadgroup int* buffer = workgroupBuffer+BUFFER_SIZE*warpInBlock;
    threadgroup real4* posBuffer = workgroupPosBuffer+TILE_SIZE*warpInBlock;
    threadgroup int* exclusionsForX = workgroupExclusions+MAX_EXCLUSIONS*warpInBlock;
    threadgroup int* block2Buffer = workgroupBlock2Buffer+(warpSize+1)*warpInBlock;

    // Loop over blocks.

    int block1 = startBlockIndex+warpIndex/NUM_TILES_IN_BATCH;
    if (block1 < startBlockIndex+numBlocks) {
        // Load data for this block.  Note that all threads in a warp are processing the same block.

        int x = sortedBlocks[block1] & BLOCK_INDEX_MASK;
        real4 blockCenterX = sortedBlockCenter[block1];
        real3 blockSizeX = sortedBlockBoundingBox[block1].toReal3();
        int neighborsInBuffer = 0;
        real4 pos1 = posq[x*TILE_SIZE+indexInTile];
#ifdef ROLE_ORDER
        // In role order an atom is listed only for the x atoms it shares a role with (1 LJ, 2 charge): any other
        // pair contributes nothing.
        const int role1 = atomRoles[x*TILE_SIZE+indexInTile];
        const tileflags ljAtomsX = BALLOT((role1 & 1) != 0);
        const tileflags chargedAtomsX = BALLOT((role1 & 2) != 0);
#endif
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
        pos1.w = 0.5f * (pos1.x * pos1.x + pos1.y * pos1.y + pos1.z * pos1.z);
        if (tileInWarp == 0) {
            posBuffer[indexInTile] = pos1;
#ifdef FB_PROBE_ATOM2_INDEP
            posBufferDup[indexInTile] = pos1;
#endif
#ifdef FB_PROBE_HALF
            {
                real3 p = trimTo3(pos1)-trimTo3(blockCenterX);
                posBufferH[indexInTile] = half4((half) p.x, (half) p.y, (half) p.z, (half) (0.5f*(p.x*p.x+p.y*p.y+p.z*p.z)));
            }
#endif
        }

        // Load exclusion data for block x.

        const int exclusionStart = exclusionRowIndices[x];
        const int exclusionEnd = exclusionRowIndices[x+1];
        const int numExclusions = exclusionEnd-exclusionStart;
        #pragma unroll 1
        for (int j = indexInWarp; j < numExclusions; j += warpSize)
            exclusionsForX[j] = exclusionIndices[exclusionStart+j];
        SYNC_WARPS;

        // Loop over atom blocks to search for neighbors.  The threads in a warp compare block1 against warpSize
        // other blocks in parallel.
        // For small systems multiple warps (NUM_TILES_IN_BATCH = 4, 2...) process one block1 reducing the overall
        // duration of the kernel because first blocks block1 have to process more block2 blocks so most of compute
        // units are idle at the end of the kernel (the kernel works on the upper triangle of
        // the NUM_BLOCKS x NUM_BLOCKS matrix).

#ifdef FB_PROBE_BBOX2
        // Scratch probe: the bounding-box scan a second time, results discarded.  The delta per call is the scan's cost.
        {
#ifdef USE_LARGE_BLOCKS
            warpflags largeBlockFlags = 0;
            int loadedLargeBlocks = 0;
#endif
            for (int block2Base = ((block1+1)/warpSize + warpIndex%NUM_TILES_IN_BATCH)*warpSize; block2Base < NUM_BLOCKS; block2Base += warpSize*NUM_TILES_IN_BATCH) {
                const bool lastIteration = block2Base + warpSize*NUM_TILES_IN_BATCH >= NUM_BLOCKS;
#ifdef USE_LARGE_BLOCKS
                if (loadedLargeBlocks == 0) {
                    int largeBlockIndex = block2Base + warpSize*NUM_TILES_IN_BATCH*indexInWarp;
                    bool includeLargeBlock = false;
                    if (largeBlockIndex < NUM_BLOCKS) {
                        real4 largeCenter = largeBlockCenter[largeBlockIndex];
                        real3 largeSize = largeBlockBoundingBox[largeBlockIndex].toReal3();
                        real4 blockDelta = blockCenterX-largeCenter;
#ifdef USE_PERIODIC
                        APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
                        blockDelta.x = max(0.0f, fabs(blockDelta.x)-blockSizeX.x-largeSize.x);
                        blockDelta.y = max(0.0f, fabs(blockDelta.y)-blockSizeX.y-largeSize.y);
                        blockDelta.z = max(0.0f, fabs(blockDelta.z)-blockSizeX.z-largeSize.z);
                        includeLargeBlock = (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < PADDED_CUTOFF_SQUARED*1.0001f);
                    }
                    largeBlockFlags = __ballot(includeLargeBlock);
                    loadedLargeBlocks = warpSize;
                }
                loadedLargeBlocks--;
                if ((largeBlockFlags&1) == 0 && !lastIteration) {
                    largeBlockFlags >>= 1;
                    continue;
                }
                largeBlockFlags >>= 1;
#endif
                int block2 = block2Base+indexInWarp;
                bool includeBlock2 = (block1 < block2 && block2 < NUM_BLOCKS);
                block2 = includeBlock2 ? block2 : block1;
                real4 blockCenterY = sortedBlockCenter[block2];
                real4 blockDelta = blockCenterX-blockCenterY;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
                includeBlock2 &= (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < (PADDED_CUTOFF*1.0001f+blockCenterX.w+blockCenterY.w)*(PADDED_CUTOFF*1.0001f+blockCenterX.w+blockCenterY.w));
                if (!lastIteration && __ballot(includeBlock2) == 0)
                    continue;
                real3 blockSizeY = sortedBlockBoundingBox[block2].toReal3();
                blockDelta.x = max(0.0f, fabs(blockDelta.x)-blockSizeX.x-blockSizeY.x);
                blockDelta.y = max(0.0f, fabs(blockDelta.y)-blockSizeX.y-blockSizeY.y);
                blockDelta.z = max(0.0f, fabs(blockDelta.z)-blockSizeX.z-blockSizeY.z);
                includeBlock2 &= (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < PADDED_CUTOFF_SQUARED*1.0001f);
                fbDummy += warpPopc(__ballot(includeBlock2));
            }
        }
#endif
#ifdef USE_LARGE_BLOCKS
        warpflags largeBlockFlags = 0;
        int loadedLargeBlocks = 0;
#endif
        int block2Count = 0;
        // Load blocks from addresses aligned by warpSize for faster loading from sortedBlockCenter and sortedBlockBoundingBox.
        for (int block2Base = ((block1+1)/warpSize + warpIndex%NUM_TILES_IN_BATCH)*warpSize; block2Base < NUM_BLOCKS; block2Base += warpSize*NUM_TILES_IN_BATCH) {
            SYNC_WARPS;
            // Last iteration cannot be skipped (on CDNA where tilesPerWarp == 2)
            const bool lastIteration = block2Base + warpSize*NUM_TILES_IN_BATCH >= NUM_BLOCKS;
#ifdef USE_LARGE_BLOCKS
            if (loadedLargeBlocks == 0) {
                // Check the next set of large blocks.

                int largeBlockIndex = block2Base + warpSize*NUM_TILES_IN_BATCH*indexInWarp;
                bool includeLargeBlock = false;
                if (largeBlockIndex < NUM_BLOCKS) {
                    real4 largeCenter = largeBlockCenter[largeBlockIndex];
                    real3 largeSize = largeBlockBoundingBox[largeBlockIndex].toReal3();
                    real4 blockDelta = blockCenterX-largeCenter;
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
                    blockDelta.x = max(0.0f, fabs(blockDelta.x)-blockSizeX.x-largeSize.x);
                    blockDelta.y = max(0.0f, fabs(blockDelta.y)-blockSizeX.y-largeSize.y);
                    blockDelta.z = max(0.0f, fabs(blockDelta.z)-blockSizeX.z-largeSize.z);
                    includeLargeBlock = (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < PADDED_CUTOFF_SQUARED);
#ifdef TRICLINIC
                    // The calculation to find the nearest periodic copy is only guaranteed to work if the nearest copy is less than half a box width away.
                    // If there's any possibility we might have missed it, do a detailed check.

                    if (periodicBoxSize.z/2-blockSizeX.z-largeSize.z < PADDED_CUTOFF || periodicBoxSize.y/2-blockSizeX.y-largeSize.y < PADDED_CUTOFF)
                        includeLargeBlock = true;
#endif
                }
                largeBlockFlags = __ballot(includeLargeBlock);
                loadedLargeBlocks = warpSize;
            }
            loadedLargeBlocks--;
            if ((largeBlockFlags&1) == 0 && !lastIteration) {
                // None of the next warpSize blocks interact with block 1.

                largeBlockFlags >>= 1;
#ifdef FB_PROBE_COUNT
                cLargeSkip++;
#endif
                continue;
            }
            largeBlockFlags >>= 1;
#endif
#ifdef FB_PROBE_COUNT
            cChunks++;
#endif
            int block2 = block2Base+indexInWarp;
            bool includeBlock2 = (block1 < block2 && block2 < NUM_BLOCKS);
            block2 = includeBlock2 ? block2 : block1;
            bool forceInclude = false;
            real4 blockCenterY = sortedBlockCenter[block2];
            real4 blockDelta = blockCenterX-blockCenterY;
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
            includeBlock2 &= (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < (PADDED_CUTOFF+blockCenterX.w+blockCenterY.w)*(PADDED_CUTOFF+blockCenterX.w+blockCenterY.w));
#ifndef TRICLINIC
            if (!lastIteration && __ballot(includeBlock2) == 0) {
#ifdef FB_PROBE_COUNT
                cCenterSkip++;
#endif
                continue;
            }
#endif
            real3 blockSizeY = sortedBlockBoundingBox[block2].toReal3();
            blockDelta.x = max(0.0f, fabs(blockDelta.x)-blockSizeX.x-blockSizeY.x);
            blockDelta.y = max(0.0f, fabs(blockDelta.y)-blockSizeX.y-blockSizeY.y);
            blockDelta.z = max(0.0f, fabs(blockDelta.z)-blockSizeX.z-blockSizeY.z);
            includeBlock2 &= (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < PADDED_CUTOFF_SQUARED);
#ifdef TRICLINIC
            // The calculation to find the nearest periodic copy is only guaranteed to work if the nearest copy is less than half a box width away.
            // If there's any possibility we might have missed it, do a detailed check.

            if (periodicBoxSize.z/2-blockSizeX.z-blockSizeY.z < PADDED_CUTOFF || periodicBoxSize.y/2-blockSizeX.y-blockSizeY.y < PADDED_CUTOFF)
                includeBlock2 = forceInclude = true;
#endif

            // Collect any blocks we identified as potentially containing neighbors.

            warpflags includeBlockFlags = __ballot(includeBlock2);
            if (includeBlock2) {
                int index = block2Count + warpPopc(includeBlockFlags&warpMask);
                block2Buffer[index] = (block2 << 1) | (forceInclude ? 1 : 0);
            }
            block2Count += warpPopc(includeBlockFlags);
#ifdef FB_PROBE_COUNT
            cCand += warpPopc(includeBlockFlags);
#endif
            SYNC_WARPS;

            // Loop over the collected candidates (each warp processes 2 blocks on CDNA or 1 block on RDNA).
            // Process even number of blocks on CDNA so both half-warps have work to do (except for
            // the last iteration of the for-block2Base loop when the left-over must be processed).

            const int block2ToProcess = lastIteration ? block2Count : block2Count/tilesPerWarp*tilesPerWarp;
            for (int block2Index = 0; block2Index < block2ToProcess; block2Index += tilesPerWarp) {
                bool includeBlock2 = block2Index + tileInWarp < block2Count;
                const int b = block2Buffer[min(block2Index + tileInWarp, block2Count - 1)];
                const bool forceInclude = b & 1;
                const int block2 = b >> 1;
                int y = sortedBlocks[block2] & BLOCK_INDEX_MASK;
#ifdef ROLE_ORDER
                // An atom is listed only for the x atoms it shares a role with.  The role regions are contiguous, so
                // a block of LJ-only atoms against a block of charge-only atoms (water O against H/H/M) has no pair to
                // test, and one ballot skips its atom test.  On tip4pew that was a third of the candidate blocks.
                const int role2 = atomRoles[y*TILE_SIZE+indexInTile];
                const tileflags roleFlags = ((role2 & 1) ? ljAtomsX : 0) | ((role2 & 2) ? chargedAtomsX : 0);
                if (BALLOT(roleFlags != 0) == 0) {
#ifdef FB_PROBE_COUNT
                    cRoleSkip++;
#endif
                    continue;
                }
#endif
#ifdef FB_PROBE_COUNT
                cAtomTests++;
#endif

                #pragma unroll 1
                for (int k = indexInTile; k < numExclusions; k += TILE_SIZE)
                    includeBlock2 &= (exclusionsForX[k] != y);
                includeBlock2 = BALLOT(!includeBlock2) == 0;

                // Check each atom in block Y for interactions.

                int atom2 = y*TILE_SIZE+indexInTile;
                real4 pos2 = posq[atom2];
#ifdef USE_PERIODIC
                if (singlePeriodicCopy) {
                    APPLY_PERIODIC_TO_POS_WITH_CENTER(pos2, blockCenterX)
                }
#endif
                pos2.w = 0.5f * (pos2.x * pos2.x + pos2.y * pos2.y + pos2.z * pos2.z);
                real4 blockCenterY = sortedBlockCenter[block2];
                real3 atomDelta = trimTo3(pos1)-trimTo3(blockCenterY);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(atomDelta)
#endif
                tileflags atomFlags = BALLOT(forceInclude || atomDelta.x*atomDelta.x+atomDelta.y*atomDelta.y+atomDelta.z*atomDelta.z < (PADDED_CUTOFF+blockCenterY.w)*(PADDED_CUTOFF+blockCenterY.w));
                tileflags interacts = 0;
                // The condition `posj.w + pos2.w - posj.x*pos2.x - posj.y*pos2.y - posj.z*pos2.z < 0.5f * PADDED_CUTOFF_SQUARED` is expressed as
                // `posj.x*pos2.x - posj.y*pos2.y - posj.z*pos2.z - posj.w - 0.5f * PADDED_CUTOFF_SQUARED - pos2.w` and computed using fma
                // (it saves 1 instruction).
                // Sign bit is used directly instead of `halfDist2 < 0.5f * PADDED_CUTOFF_SQUARED ? 1<<j : 0`.
#ifdef USE_PERIODIC
                if (!singlePeriodicCopy) {
                    while (atomFlags) {
                        int j = __ffs(atomFlags)-1;
                        atomFlags = atomFlags ^ (static_cast<tileflags>(1) << j);
                        real3 delta = trimTo3(pos2)-trimTo3(posBuffer[j]);
                        APPLY_PERIODIC_TO_DELTA(delta)
                        real d = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z - PADDED_CUTOFF_SQUARED;
#ifdef ROLE_ORDER
                        d = ((roleFlags >> j) & 1) ? d : 1.0f;
#endif
                        collectInteractions(interacts, d, j);
                    }
                }
                else {
#endif
                    const real lim = 0.5f * PADDED_CUTOFF_SQUARED - pos2.w;
#ifdef FB_PROBE_NOTEST
                    // Scratch probe: no 32x32 test, every atom passing the sphere test is listed (a superset).
#ifdef ROLE_ORDER
                    interacts = atomFlags & roleFlags;
#else
                    interacts = atomFlags;
#endif
                    atomFlags = 0;
#endif
#ifdef FB_PROBE_HALF
                    if (__popc(atomFlags) >= MIN_ATOMS_TO_UNROLL) {
                        const real3 p2r = trimTo3(pos2)-trimTo3(blockCenterX);
                        const real limH = 0.5f*(PADDED_CUTOFF+0.001f)*(PADDED_CUTOFF+0.001f) - 0.5f*(p2r.x*p2r.x+p2r.y*p2r.y+p2r.z*p2r.z);
                        #pragma unroll
                        for (int j = 0; j < TILE_SIZE; j++) {
                            float4 pj = float4(posBufferH[j]);
                            real d = fma(-pj.x, p2r.x, fma(-pj.y, p2r.y, fma(-pj.z, p2r.z, pj.w - limH)));
                            interacts |= (__float_as_uint(d) >> 31) << j;
                        }
#ifdef ROLE_ORDER
                        interacts &= atomFlags & roleFlags;
#else
                        interacts &= atomFlags;
#endif
                        atomFlags = 0;
                    }
#endif
#if defined(USE_MFMA)
                    const vfloat c = { -lim, -lim, -lim, -lim };
                    mfma4x4<0>(pos1, pos2, c, interacts);
                    mfma4x4<1>(pos1, pos2, c, interacts);
                    mfma4x4<2>(pos1, pos2, c, interacts);
                    mfma4x4<3>(pos1, pos2, c, interacts);
                    mfma4x4<4>(pos1, pos2, c, interacts);
                    mfma4x4<5>(pos1, pos2, c, interacts);
                    mfma4x4<6>(pos1, pos2, c, interacts);
                    mfma4x4<7>(pos1, pos2, c, interacts);
#else
                    // atomFlags is uniform across the SIMD group. When enough atoms of block X pass the sphere test,
                    // an unrolled test of all 32 is cheaper than the serial loop over the passing ones; masking with
                    // atomFlags gives the same bits. The bits are set directly, because collectInteractions keeps
                    // only bit 31 when there are no single pairs.
                    if (__popc(atomFlags) >= MIN_ATOMS_TO_UNROLL) {
                        #pragma unroll
                        for (int j = 0; j < TILE_SIZE; j++) {
                            real4 posj = posBuffer[j];
                            real d = fma(-posj.x, pos2.x, fma(-posj.y, pos2.y, fma(-posj.z, pos2.z, posj.w - lim)));
                            interacts |= (__float_as_uint(d) >> 31) << j;
                        }
#ifdef ROLE_ORDER
                        interacts &= atomFlags & roleFlags;
#else
                        interacts &= atomFlags;
#endif
                        atomFlags = 0;
                    }
                    while (atomFlags) {
                        int j = __ffs(atomFlags)-1;
                        atomFlags = atomFlags ^ (static_cast<tileflags>(1) << j);
                        real4 posj = posBuffer[j];
                        real d = fma(-posj.x, pos2.x, fma(-posj.y, pos2.y, fma(-posj.z, pos2.z, posj.w - lim)));
#ifdef ROLE_ORDER
                        d = ((roleFlags >> j) & 1) ? d : 1.0f;
#endif
                        collectInteractions(interacts, d, j);
                    }
#endif
#ifdef USE_PERIODIC
                }
#endif
#ifdef FB_PROBE_ATOM2
                // Scratch probe: the 32x32 atom test a second time with a slightly wider cutoff, result discarded.
#ifdef USE_PERIODIC
                if (singlePeriodicCopy) {
#else
                {
#endif
                    tileflags atomFlags2 = BALLOT(forceInclude || atomDelta.x*atomDelta.x+atomDelta.y*atomDelta.y+atomDelta.z*atomDelta.z < (PADDED_CUTOFF*1.0001f+blockCenterY.w)*(PADDED_CUTOFF*1.0001f+blockCenterY.w));
                    tileflags interacts2 = 0;
                    const real lim2 = 0.5f * PADDED_CUTOFF_SQUARED * 1.0001f - pos2.w;
                    if (__popc(atomFlags2) >= MIN_ATOMS_TO_UNROLL) {
                        #pragma unroll
                        for (int j = 0; j < TILE_SIZE; j++) {
                            real4 posj = FB_DUP_BUF[j];
                            real d = fma(-posj.x, pos2.x, fma(-posj.y, pos2.y, fma(-posj.z, pos2.z, posj.w - lim2)));
                            interacts2 |= (__float_as_uint(d) >> 31) << j;
                        }
#ifdef ROLE_ORDER
                        interacts2 &= atomFlags2 & roleFlags;
#else
                        interacts2 &= atomFlags2;
#endif
                        atomFlags2 = 0;
                    }
                    while (atomFlags2) {
                        int j = __ffs(atomFlags2)-1;
                        atomFlags2 = atomFlags2 ^ (static_cast<tileflags>(1) << j);
                        real4 posj = FB_DUP_BUF[j];
                        real d = fma(-posj.x, pos2.x, fma(-posj.y, pos2.y, fma(-posj.z, pos2.z, posj.w - lim2)));
#ifdef ROLE_ORDER
                        d = ((roleFlags >> j) & 1) ? d : 1.0f;
#endif
                        collectInteractions(interacts2, d, j);
                    }
                    fbDummy += interacts2;
                }
#endif
                if (atom2 >= NUM_ATOMS || !includeBlock2) {
                    interacts = 0;
                }

#if MAX_BITS_FOR_PAIRS > 0
                const unsigned int interactCount = __popc(interacts);

                // Record interactions that should be computed as single pairs rather than in blocks.
                const bool storeAsSinglePair = interactCount > 0 && interactCount <= MAX_BITS_FOR_PAIRS;
                if (__ballot(storeAsSinglePair)) {
                    unsigned int sum = 0;
                    unsigned int prevSum = 0;
                    for (int i = 1; i <= MAX_BITS_FOR_PAIRS; i++) {
                        warpflags b = __ballot(interactCount == i);
                        sum += warpPopc(b) * i;
                        prevSum += warpPopc(b&warpMask) * i;
                    }
                    unsigned int pairStartIndex = 0;
                    if (indexInWarp == 0)
                        pairStartIndex = atomicAdd(&interactionCount[1], sum);
                    pairStartIndex = __shfl(pairStartIndex, 0);
                    unsigned int pairIndex = pairStartIndex + prevSum;
                    if (storeAsSinglePair && pairIndex+interactCount <= maxSinglePairs) {
                        while (interacts != 0) {
                            int j = __ffs(interacts)-1;
                            singlePairs[pairIndex] = make_int2(atom2, x*TILE_SIZE+j);
                            interacts = interacts ^ (static_cast<tileflags>(1) << j);
                            pairIndex++;
                        }
                    }
                }
#else
                const unsigned int interactCount = interacts;
#endif

                // Add any interacting atoms to the buffer.

                warpflags includeAtomFlags = __ballot(interactCount > MAX_BITS_FOR_PAIRS);
                if (interactCount > MAX_BITS_FOR_PAIRS) {
                    int index = neighborsInBuffer+warpPopc(includeAtomFlags&warpMask);
                    buffer[index] = atom2;
                }
                neighborsInBuffer += warpPopc(includeAtomFlags);
                SYNC_WARPS;
                if (neighborsInBuffer > BUFFER_SIZE-warpSize) {
                    // Store the new tiles to memory.

                    unsigned int tilesToStore = neighborsInBuffer/warpSize*tilesPerWarp;
                    unsigned int tileStartIndex = 0;
                    if (indexInWarp == 0)
                        tileStartIndex = atomicAdd(&interactionCount[0], tilesToStore);
                    unsigned int newTileStartIndex = __shfl(tileStartIndex, 0);
                    if (newTileStartIndex+tilesToStore <= maxTiles) {
                        if (indexInWarp < tilesToStore)
                            interactingTiles[newTileStartIndex+indexInWarp] = x;
                        for (int j = 0; j < tilesToStore/tilesPerWarp; j++)
                            interactingAtoms[newTileStartIndex*TILE_SIZE+j*warpSize+indexInWarp] = buffer[j*warpSize+indexInWarp];
                    }
                    if (indexInWarp+TILE_SIZE*tilesToStore < BUFFER_SIZE)
                        buffer[indexInWarp] = buffer[indexInWarp+TILE_SIZE*tilesToStore];
                    neighborsInBuffer -= TILE_SIZE*tilesToStore;
                    SYNC_WARPS;
                }
            }

            // Move not processed blocks to the head of block2Buffer.
            if (indexInWarp < block2Count - block2ToProcess)
                block2Buffer[indexInWarp] = block2Buffer[block2ToProcess + indexInWarp];
            block2Count = block2Count - block2ToProcess;
        }

        // If we have a partially filled buffer,  store it to memory.

        if (neighborsInBuffer > 0) {
            unsigned int tilesToStore = (neighborsInBuffer+TILE_SIZE-1)/TILE_SIZE;
            unsigned int tileStartIndex = 0;
            if (indexInWarp == 0)
                tileStartIndex = atomicAdd(&interactionCount[0], tilesToStore);
            unsigned int newTileStartIndex = __shfl(tileStartIndex, 0);
            if (newTileStartIndex+tilesToStore <= maxTiles) {
                if (indexInWarp < tilesToStore)
                    interactingTiles[newTileStartIndex+indexInWarp] = x;
                for (int j = 0; j <= tilesToStore/tilesPerWarp; j++) {
                    if (j*warpSize+indexInWarp < tilesToStore*TILE_SIZE)
                        interactingAtoms[newTileStartIndex*TILE_SIZE+j*warpSize+indexInWarp] = (j*warpSize+indexInWarp < neighborsInBuffer ? buffer[j*warpSize+indexInWarp] : PADDED_NUM_ATOMS);
                }
            }
        }
#ifdef FB_PROBE_COUNT
        if (indexInWarp == 0) {
            atomicAdd(&fbCounters[1], cChunks);
            atomicAdd(&fbCounters[2], cLargeSkip);
            atomicAdd(&fbCounters[3], cCenterSkip);
            atomicAdd(&fbCounters[4], cCand);
            atomicAdd(&fbCounters[5], cRoleSkip);
            atomicAdd(&fbCounters[6], cAtomTests);
        }
#endif
#if defined(FB_PROBE_BBOX2) || defined(FB_PROBE_ATOM2)
        // Sink the probe's results so the compiler keeps the work.  numBlocks never reaches this value.
        if (fbDummy == 0xffffffffu && numBlocks == 0xffffffffu)
            interactionCount[1] = fbDummy;
#endif
    }

    // Record the positions the neighbor list is based on.

    int i = threadIdx.x+blockIdx.x*GROUP_SIZE;
    if (i < NUM_ATOMS) {
        oldPositions[i] = posq[i];
    }
}

extern "C" __global__ void copyInteractionCounts(const unsigned int* __restrict__ interactionCount,
        unsigned int* __restrict__ pinnedInteractionCount) {
    // One 16 byte store, so the host never sees the new copy counter with old counts.  The second copy goes to a
    // ring the next 3 copies leave alone, for the host's re-read once the command buffer completes.
    uint4 counts = uint4(interactionCount[0], interactionCount[1], pinnedInteractionCount[2]+1, 0);
    *((device uint4*) pinnedInteractionCount) = counts;
    *((device uint4*) (pinnedInteractionCount+4+4*(counts.z%4))) = counts;
}

#ifdef INDIRECT_REBUILD
// A separate dispatch supplies the device-wide ordering between the per-tile
// displacement test and the decision.  It runs as one threadgroup, so on rebuild
// steps it also computes the sort keys and sorts them, with only threadgroup
// barriers, in sortShortList2's order.  Above MaxBlocksSortedInOneGroup blocks
// (SORT_BLOCKS_BY_BINS) it sizes the bin sort's three dispatches instead.
// It also does copyInteractionCounts' work for steps where the host doesn't wait
// for the count, so those steps skip that dispatch.
extern "C" __global__ void prepareRebuildDispatch(const unsigned int* __restrict__ displacementFlags,
        int* __restrict__ rebuildNeighborList, unsigned int* __restrict__ interactionCount,
        unsigned int* __restrict__ pinnedInteractionCount, real2* __restrict__ blockSizeRange,
        const real4* __restrict__ blockBoundingBox, unsigned int* __restrict__ sortedBlocks,
        unsigned int* __restrict__ dispatch) {
#ifndef SORT_BLOCKS_BY_BINS
    __shared__ unsigned char blockBin[NUM_BLOCKS];
    __shared__ unsigned int simdCounts[32*NUM_SIZE_BINS];
    __shared__ unsigned int binStart[NUM_SIZE_BINS];
#endif
    __shared__ bool anyMoved;
    if (threadIdx.x == 0)
        anyMoved = false;
    __syncthreads();
    bool moved = false;
    for (int i = threadIdx.x; i < NUM_BLOCKS; i += blockDim.x)
        moved = moved || displacementFlags[i] != 0;
    if (simd_any(moved))
        anyMoved = true;
    __syncthreads();
    bool rebuild = anyMoved;
    if (threadIdx.x == 0) {
        pinnedInteractionCount[0] = interactionCount[0];
        pinnedInteractionCount[1] = interactionCount[1];
        rebuildNeighborList[0] = rebuild;
        if (rebuild) {
            interactionCount[0] = 0;
            interactionCount[1] = 0;
        }
        else {
            // Nothing reads this step's range.  sortBoxData resets it on rebuild steps.
            blockSizeRange->x = 1e38;
            blockSizeRange->y = 0;
        }
        dispatch[0] = rebuild ? (NUM_ATOMS+63)/64 : 0;
        dispatch[3] = rebuild ? (NUM_BLOCKS*32*NUM_TILES_IN_BATCH+GROUP_SIZE-1)/GROUP_SIZE : 0;
#ifdef SORT_BLOCKS_BY_BINS
        // countBlockBins and scatterBlocks, then sumBlockBins.
        dispatch[6] = rebuild ? (NUM_BLOCKS+BIN_SORT_GROUP-1)/BIN_SORT_GROUP : 0;
        dispatch[9] = rebuild ? 1 : 0;
        for (int i = 0; i < 4; i++) {
#else
        for (int i = 0; i < 2; i++) {
#endif
            dispatch[3*i+1] = 1;
            dispatch[3*i+2] = 1;
        }
    }
#ifndef SORT_BLOCKS_BY_BINS
    if (!rebuild)
        return;

    // A stable counting sort over computeSortKeys' size bins, so sortedBlocks holds its keys in sorted order.
    // Each SIMD group owns one contiguous run of blocks.  It counts its run per bin, one pass over the per-SIMD
    // counts gives every run its offset inside each bin, and each SIMD group then places its run in order.
    // Two barriers in all: one threadgroup runs on one core, so barriers and global loads set the time.
    // The rank sort this replaced compared every pair of keys, 23 us per rebuild at 921 blocks on an M3 Ultra.

    const int lane = threadIdx.x%32;
    const int simd = threadIdx.x/32;
    const int numSimds = (blockDim.x+31)/32;
    const unsigned int perSimd = (NUM_BLOCKS+numSimds-1)/numSimds;
    const unsigned int first = min(simd*perSimd, (unsigned int) NUM_BLOCKS);
    const unsigned int last = min(first+perSimd, (unsigned int) NUM_BLOCKS);
    real2 sizeRange = make_real2(LOG(blockSizeRange->x), LOG(blockSizeRange->y));
    real scale = NUM_SIZE_BINS/(sizeRange.y-sizeRange.x);

    // Lane b keeps bin b's count for this SIMD group's run.
    unsigned int count = 0;
    for (unsigned int base = first; base < last; base += 32) {
        const unsigned int i = base+lane;
        int bin = -1;
        if (i < last) {
            real4 box = blockBoundingBox[i];
            real size = LOG(box.x+box.y+box.z);
            bin = (size-sizeRange.x)*scale;
            bin = max(0, min(bin, NUM_SIZE_BINS-1));
            blockBin[i] = bin;
        }
        for (int b = 0; b < NUM_SIZE_BINS; b++) {
            unsigned int n = __popc(BALLOT(bin == b));
            if (lane == b)
                count += n;
        }
    }
    if (lane < NUM_SIZE_BINS)
        simdCounts[simd*NUM_SIZE_BINS+lane] = count;
    __syncthreads();

    // SIMD group 0, lane b: bin b's per-run counts become offsets inside the bin, and the bin totals become
    // bin starts.  All 32 lanes take part in the prefix sum.
    if (simd == 0) {
        unsigned int total = 0;
        if (lane < NUM_SIZE_BINS) {
            for (int w = 0; w < numSimds; w++) {
                unsigned int n = simdCounts[w*NUM_SIZE_BINS+lane];
                simdCounts[w*NUM_SIZE_BINS+lane] = total;
                total += n;
            }
        }
        unsigned int start = simd_prefix_exclusive_sum(total);
        if (lane < NUM_SIZE_BINS)
            binStart[lane] = start;
    }
    __syncthreads();

    // Lane b keeps the next free slot of bin b for this run.
    unsigned int next = 0;
    if (lane < NUM_SIZE_BINS)
        next = binStart[lane]+simdCounts[simd*NUM_SIZE_BINS+lane];
    for (unsigned int base = first; base < last; base += 32) {
        const unsigned int i = base+lane;
        const int bin = (i < last ? (int) blockBin[i] : -1);
        unsigned int slot = 0;
        for (int b = 0; b < NUM_SIZE_BINS; b++) {
            unsigned int bits = BALLOT(bin == b);
            unsigned int binNext = SHFL(next, (ushort) b);
            if (bin == b)
                slot = binNext+__popc(bits & ((1u<<lane)-1));
            if (lane == b)
                next += __popc(bits);
        }
        if (bin >= 0)
            sortedBlocks[slot] = (((unsigned int) bin)<<BIN_SHIFT) + i;
    }
#endif
}

#ifdef SORT_BLOCKS_BY_BINS
/**
 * A stable counting sort over computeSortKeys' 20 size bins, for block counts too large for
 * prepareRebuildDispatch.  Inside a bin, blocks stay in index order, which is the order of sorting
 * computeSortKeys' keys, so the neighbor list and its tile orientation are unchanged.  countBlockBins
 * ranks each block inside its threadgroup's share of its bin, sumBlockBins turns the per-threadgroup
 * counts into offsets, and scatterBlocks writes the keys.  Adapted from the dual neighbor list's
 * dualSortLocal, dualSortOffsets and dualSortScatter (r3/duallist 5ec392b34).
 */
extern "C" __global__ void countBlockBins(const real4* __restrict__ blockBoundingBox, const real2* __restrict__ blockSizeRange,
        unsigned int* __restrict__ binRanks, unsigned int* __restrict__ binCounts) {
    __shared__ unsigned int warpCounts[(BIN_SORT_GROUP/32)*NUM_SIZE_BINS];
    const int lane = threadIdx.x%32;
    const int warp = threadIdx.x/32;
    const unsigned int index = blockIdx.x*BIN_SORT_GROUP+threadIdx.x;
    int bin = -1;
    if (index < NUM_BLOCKS) {
        real2 sizeRange = make_real2(LOG(blockSizeRange->x), LOG(blockSizeRange->y));
        real scale = NUM_SIZE_BINS/(sizeRange.y-sizeRange.x);
        real4 box = blockBoundingBox[index];
        real size = LOG(box.x+box.y+box.z);
        bin = (size-sizeRange.x)*scale;
        bin = max(0, min(bin, NUM_SIZE_BINS-1));
    }
    unsigned int rank = 0;
    for (int b = 0; b < NUM_SIZE_BINS; b++) {
        unsigned int bits = BALLOT(bin == b);
        if (bin == b)
            rank = __popc(bits & ((1u<<lane)-1));
        if (lane == 0)
            warpCounts[warp*NUM_SIZE_BINS+b] = __popc(bits);
    }
    SYNC_THREADS;
    if (bin >= 0) {
        for (int w = 0; w < warp; w++)
            rank += warpCounts[w*NUM_SIZE_BINS+bin];
        binRanks[index] = (bin<<8) | rank;
    }
    if (threadIdx.x < NUM_SIZE_BINS) {
        unsigned int count = 0;
        for (int w = 0; w < BIN_SORT_GROUP/32; w++)
            count += warpCounts[w*NUM_SIZE_BINS+threadIdx.x];
        binCounts[blockIdx.x*NUM_SIZE_BINS+threadIdx.x] = count;
    }
}

extern "C" __global__ void sumBlockBins(const unsigned int* __restrict__ binCounts, unsigned int* __restrict__ binOffsets) {
    __shared__ unsigned int totals[NUM_SIZE_BINS];
    const int bin = threadIdx.x;
    const int groups = (NUM_BLOCKS+BIN_SORT_GROUP-1)/BIN_SORT_GROUP;
    if (bin < NUM_SIZE_BINS) {
        unsigned int total = 0;
        for (int g = 0; g < groups; g++)
            total += binCounts[g*NUM_SIZE_BINS+bin];
        totals[bin] = total;
    }
    SYNC_THREADS;
    if (bin < NUM_SIZE_BINS) {
        unsigned int offset = 0;
        for (int b = 0; b < bin; b++)
            offset += totals[b];
        for (int g = 0; g < groups; g++) {
            binOffsets[g*NUM_SIZE_BINS+bin] = offset;
            offset += binCounts[g*NUM_SIZE_BINS+bin];
        }
    }
}

extern "C" __global__ void scatterBlocks(const unsigned int* __restrict__ binRanks, const unsigned int* __restrict__ binOffsets,
        unsigned int* __restrict__ sortedBlocks) {
    const unsigned int index = blockIdx.x*BIN_SORT_GROUP+threadIdx.x;
    if (index < NUM_BLOCKS) {
        unsigned int bin = binRanks[index]>>8;
        unsigned int rank = binRanks[index]&255;
        sortedBlocks[binOffsets[blockIdx.x*NUM_SIZE_BINS+bin]+rank] = (bin<<BIN_SHIFT) + index;
    }
}
#endif
#endif

#ifdef ROLE_ORDER
/**
 * Copy the positions into role order for the neighbor list kernels.  The nonbonded kernel reads posq through atomOrder
 * itself, so it sees charges written after this runs.
 */
extern "C" __global__ void gatherOrderedPositions(const real4* __restrict__ posq, const int* __restrict__ atomOrder,
        real4* __restrict__ orderedPosq) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < PADDED_NUM_ATOMS; i += blockDim.x*gridDim.x)
        orderedPosq[i] = posq[atomOrder[i]];
}

/**
 * Add the forces the nonbonded kernel accumulated in nonbonded order into the context force buffer, and clear them
 * for the next step.  Runs after the force pass, so a plain add in a fixed order is exact.
 */
extern "C" __global__ void scatterOrderedForces(unsigned long long* __restrict__ orderedForces, const int* __restrict__ atomOrder,
        unsigned long long* __restrict__ forceBuffers) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < PADDED_NUM_ATOMS; i += blockDim.x*gridDim.x) {
        const int atom = atomOrder[i];
        forceBuffers[atom] += orderedForces[i];
        forceBuffers[atom+PADDED_NUM_ATOMS] += orderedForces[i+PADDED_NUM_ATOMS];
        forceBuffers[atom+2*PADDED_NUM_ATOMS] += orderedForces[i+2*PADDED_NUM_ATOMS];
        orderedForces[i] = 0;
        orderedForces[i+PADDED_NUM_ATOMS] = 0;
        orderedForces[i+2*PADDED_NUM_ATOMS] = 0;
    }
}
#endif
