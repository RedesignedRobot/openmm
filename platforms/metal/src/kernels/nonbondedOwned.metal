/**
 * Direct interactions for small GB systems. Each SIMD group owns an unordered
 * block pair, including pairs with exclusions, and writes both endpoint slots.
 * The composed interaction body retains Coulomb, LJ and the OBC chain rule.
 */
DEVICE bool ownedBlocksInRange(real4 a, real4 sa, real4 b, real4 sb,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX,
        real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    real4 delta = b-a;
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(delta)
#endif
    delta.x = max((real) 0, fabs(delta.x)-sa.x-sb.x);
    delta.y = max((real) 0, fabs(delta.y)-sa.y-sb.y);
    delta.z = max((real) 0, fabs(delta.z)-sa.z-sb.z);
    return delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < MAX_CUTOFF_SQUARED*1.0001f;
}

extern "C" __launch_bounds__(THREAD_BLOCK_SIZE) __global__ void computeNonbonded(
        unsigned long long* __restrict__ forceBuffers, mixed* __restrict__ energyBuffer, const real4* __restrict__ posq, const tileflags* __restrict__ exclusions,
        const int2* __restrict__ exclusionTiles, unsigned int startTileIndex, unsigned long long numTileIndices
#ifdef USE_CUTOFF
        , const int* __restrict__ tiles, const unsigned int* __restrict__ interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, const real4* __restrict__ blockCenter,
        const real4* __restrict__ blockSize, const unsigned int* __restrict__ interactingAtoms, unsigned int maxSinglePairs,
        const int2* __restrict__ singlePairs
#endif
        PARAMETER_ARGUMENTS, real4* __restrict__ partialForce, const int* __restrict__ pairExclusions) {

    const unsigned int lane = LOCAL_ID%TILE_SIZE;
    mixed energy = 0;
    INIT_DERIVATIVES
    for (unsigned int owner = GLOBAL_ID/TILE_SIZE; owner < NUM_BLOCKS*(NUM_BLOCKS+1)/2;
            owner += GLOBAL_SIZE/TILE_SIZE) {
        unsigned int x = (unsigned int) ((SQRT(8.0f*owner+1.0f)-1.0f)*0.5f);
        if (x*(x+1)/2 > owner)
            x--;
        else if ((x+1)*(x+2)/2 <= owner)
            x++;
        const unsigned int y = owner-x*(x+1)/2;
        const unsigned int atom1 = x*TILE_SIZE+lane;
        const unsigned int atomY = y*TILE_SIZE+lane;
        real3 force = make_real3(0);
        real3 shflForce = make_real3(0);
        if (ownedBlocksInRange(blockCenter[x], blockSize[x], blockCenter[y], blockSize[y], periodicBoxSize,
                invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ)) {
            const real4 posq1 = posq[atom1];
            LOAD_ATOM1_PARAMETERS
            unsigned int j = atomY;
            real4 shflPosq = posq[j];
            DECLARE_LOCAL_PARAMETERS
            LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
            // Timing-only ceiling probe: intentionally omit exclusion-mask reads.
            // Diagonal pairs are evaluated from both sides. Off-diagonal pairs
            // rotate each y atom and its force back to its original lane.
            for (unsigned int k = 0; k < TILE_SIZE; k++) {
                const unsigned int column = (lane+k)%TILE_SIZE;
                const unsigned int atom2 = y*TILE_SIZE+column;
                const real4 posq2 = shflPosq;
                real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 != atom2 && r2 < MAX_CUTOFF_SQUARED) {
                    real invR = RSQRT(r2);
                    real r = r2*invR;
                    LOAD_ATOM2_PARAMETERS
                    bool isExcluded = false; // Incorrect forces, diagnostic only.
                    real dEdR = 0;
                    real tempEnergy = 0;
                    const real interactionScale = (x == y ? 0.5f : 1.0f);
                    COMPUTE_INTERACTION
#ifdef INCLUDE_ENERGY
                    energy += interactionScale*tempEnergy;
#endif
#ifdef INCLUDE_FORCES
                    delta *= dEdR;
                    force -= delta;
                    if (x != y)
                        shflForce += delta;
#endif
                }
                SHUFFLE_WARP_DATA
            }
        }
#ifdef INCLUDE_FORCES
        partialForce[y*PADDED_NUM_ATOMS+atom1] = make_real4(force.x, force.y, force.z, 0);
        if (x != y)
            partialForce[x*PADDED_NUM_ATOMS+atomY] = make_real4(shflForce.x, shflForce.y, shflForce.z, 0);
#endif
    }
#ifdef INCLUDE_ENERGY
    energyBuffer[GLOBAL_ID] += energy;
#endif
    SAVE_DERIVATIVES
}

KERNEL void reduceNonbondedOwned(GLOBAL const real4* RESTRICT partialForce, GLOBAL mm_ulong* RESTRICT forceBuffers) {
    const unsigned int part = GLOBAL_ID%8;
    for (unsigned int atom = GLOBAL_ID/8; atom < NUM_ATOMS; atom += GLOBAL_SIZE/8) {
        mm_long fx = 0, fy = 0, fz = 0;
        for (unsigned int block = part; block < NUM_BLOCKS; block += 8) {
            real4 f = partialForce[block*PADDED_NUM_ATOMS+atom];
            fx += realToFixedPoint(f.x);
            fy += realToFixedPoint(f.y);
            fz += realToFixedPoint(f.z);
        }
        for (ushort mask = 1; mask < 8; mask *= 2) {
            fx += as_type<mm_long>(simd_shuffle_xor(as_type<uint2>(fx), mask));
            fy += as_type<mm_long>(simd_shuffle_xor(as_type<uint2>(fy), mask));
            fz += as_type<mm_long>(simd_shuffle_xor(as_type<uint2>(fz), mask));
        }
        if (part == 0) {
            ATOMIC_ADD(&forceBuffers[atom], (mm_ulong) fx);
            ATOMIC_ADD(&forceBuffers[atom+PADDED_NUM_ATOMS], (mm_ulong) fy);
            ATOMIC_ADD(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (mm_ulong) fz);
        }
    }
}
