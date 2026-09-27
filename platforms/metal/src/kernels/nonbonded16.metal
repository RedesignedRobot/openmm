// Ordinary 16x32 packet loop. Exclusions and single pairs remain in nonbonded.metal.
// Both SIMD halves hold the same sixteen x atoms and rotate independent y sets.
// The emitter stores xHalf=2*xBlock+half and 32 y indices per packet.
#define warpRotateLeft(value) SHFL(value, nextLane16)
    const unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return;
    const unsigned int nextLane16 = (tgx & 16) | ((tgx+1) & 15);
    mm_long xForceX = 0, xForceY = 0, xForceZ = 0;
    const unsigned int end = firstNeighborListTile(numTiles, warp+1, totalWarps);
    for (unsigned int pos = firstNeighborListTile(numTiles, warp, totalWarps); pos < end; pos++) {
        const int xHalf = tiles[pos];
        const int x = xHalf >> 1;
        const unsigned int atom1 = xHalf*16+(tgx & 15);
        real3 force = make_real3(0);
        real4 posq1 = posq[atom1];
        LOAD_ATOM1_PARAMETERS
        unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];
        int atomIndex = j;
        DECLARE_LOCAL_PARAMETERS
        real4 shflPosq;
        real3 shflForce = make_real3(0);
        if (j < PADDED_NUM_ATOMS) {
            shflPosq = posq[j];
            LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
        }
        else {
            shflPosq = make_real4(0);
            CLEAR_LOCAL_PARAMETERS
        }
        real4 blockSizeX = blockSize[x];
        bool singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= MAX_CUTOFF &&
                                  0.5f*periodicBoxSize.y-blockSizeX.y >= MAX_CUTOFF &&
                                  0.5f*periodicBoxSize.z-blockSizeX.z >= MAX_CUTOFF);
#ifdef USE_PERIODIC
            if (singlePeriodicCopy) {
                // The box is small enough that we can just translate all the atoms into a single periodic
                // box, then skip having to apply periodic boundary conditions later.
                real4 blockCenterX = blockCenter[x];
                APPLY_PERIODIC_TO_POS_WITH_CENTER(posq1, blockCenterX)
                APPLY_PERIODIC_TO_POS_WITH_CENTER(shflPosq, blockCenterX)
                #pragma unroll 2
                for (j = 0; j < 16; j++) {
                    real4 posq2 = shflPosq;
                    real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                    if (r2 < MAX_CUTOFF_SQUARED) {
#endif
                        real invR = RSQRT(r2);
                        real r = r2*invR;
                        LOAD_ATOM2_PARAMETERS
                        int atom2 = atomIndex;
#ifdef USE_SYMMETRIC
                        real dEdR = 0.0f;
#else
                        real3 dEdR1 = make_real3(0);
                        real3 dEdR2 = make_real3(0);
#endif
#ifdef USE_EXCLUSIONS
                        bool isExcluded = !(atom1 < NUM_ATOMS && atom2 < NUM_ATOMS);
#endif
                        real tempEnergy = 0.0f;
                        const real interactionScale = 1.0f;
                        COMPUTE_INTERACTION
#ifdef INCLUDE_ENERGY
                        energy += tempEnergy;
#endif
#ifdef INCLUDE_FORCES
#ifdef USE_SYMMETRIC
                        delta *= dEdR;
                        force.x -= delta.x;
                        force.y -= delta.y;
                        force.z -= delta.z;
                        shflForce.x += delta.x;
                        shflForce.y += delta.y;
                        shflForce.z += delta.z;
#else // !USE_SYMMETRIC
                        force.x -= dEdR1.x;
                        force.y -= dEdR1.y;
                        force.z -= dEdR1.z;
                        shflForce.x += dEdR2.x;
                        shflForce.y += dEdR2.y;
                        shflForce.z += dEdR2.z;
#endif // end USE_SYMMETRIC
#endif
#ifdef USE_CUTOFF
                    }
#endif
                    SHUFFLE_WARP_DATA
                    atomIndex = warpRotateLeft(atomIndex);
                }
            }
            else
#endif
            {
                // We need to apply periodic boundary conditions separately for each interaction.
                #pragma unroll 2
                for (j = 0; j < 16; j++) {
                    real4 posq2 = shflPosq;
                    real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(delta)
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                    if (r2 < MAX_CUTOFF_SQUARED) {
#endif
                        real invR = RSQRT(r2);
                        real r = r2*invR;
                        LOAD_ATOM2_PARAMETERS
                        int atom2 = atomIndex;
#ifdef USE_SYMMETRIC
                        real dEdR = 0.0f;
#else
                        real3 dEdR1 = make_real3(0);
                        real3 dEdR2 = make_real3(0);
#endif
#ifdef USE_EXCLUSIONS
                        bool isExcluded = !(atom1 < NUM_ATOMS && atom2 < NUM_ATOMS);
#endif
                        real tempEnergy = 0.0f;
                        const real interactionScale = 1.0f;
                        COMPUTE_INTERACTION
#ifdef INCLUDE_ENERGY
                        energy += tempEnergy;
#endif
#ifdef INCLUDE_FORCES
#ifdef USE_SYMMETRIC
                        delta *= dEdR;
                        force.x -= delta.x;
                        force.y -= delta.y;
                        force.z -= delta.z;
                        shflForce.x += delta.x;
                        shflForce.y += delta.y;
                        shflForce.z += delta.z;
#else // !USE_SYMMETRIC
                        force.x -= dEdR1.x;
                        force.y -= dEdR1.y;
                        force.z -= dEdR1.z;
                        shflForce.x += dEdR2.x;
                        shflForce.y += dEdR2.y;
                        shflForce.z += dEdR2.z;
#endif // end USE_SYMMETRIC
#endif
#ifdef USE_CUTOFF
                    }
#endif
                    SHUFFLE_WARP_DATA
                    atomIndex = warpRotateLeft(atomIndex);
                }
            }
#ifdef INCLUDE_FORCES
        // Every lane executes the shuffle. Only the lower half writes x forces.
        force += SHFL(force, tgx ^ 16);
        xForceX += realToFixedPoint(force.x);
        xForceY += realToFixedPoint(force.y);
        xForceZ += realToFixedPoint(force.z);
        if (pos+1 == end || tiles[pos+1] != xHalf) {
            if (tgx < 16) {
                atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>(xForceX));
                atomicAdd(&forceBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>(xForceY));
                atomicAdd(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>(xForceZ));
            }
            xForceX = xForceY = xForceZ = 0;
        }
        if (atomIndex < PADDED_NUM_ATOMS) {
            atomicAdd(&forceBuffers[atomIndex], static_cast<unsigned long long>(realToFixedPoint(shflForce.x)));
            atomicAdd(&forceBuffers[atomIndex+PADDED_NUM_ATOMS], static_cast<unsigned long long>(realToFixedPoint(shflForce.y)));
            atomicAdd(&forceBuffers[atomIndex+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>(realToFixedPoint(shflForce.z)));
        }
#endif
    }
#undef warpRotateLeft
