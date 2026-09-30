/**
 * GBSA-OBC with block-pair ownership.
 *
 * GB has no exclusions, so the pair kernels loop over all block pairs and cull them with the current block bounds.
 * Each SIMD group owns one block pair (x, y) at a time.  Both pair kernels evaluate each unordered pair once: each
 * lane owns one atom of block x, and the atoms of block y with their sums rotate round the SIMD group.  The Born sum
 * takes both directed terms from one distance.
 *
 * The Born sum adds each atom's sum over a block pair to the fixed point bornSum buffer, one atomic per atom per block
 * pair.  Integer addition is exact, so the total doesn't depend on scheduling, and Force1 turns the totals of its
 * two blocks into Born radii itself instead of waiting for a separate kernel.  Force1 adds its sums the same way, to
 * the force buffer and to bornForce, finished for the chain rule term in the nonbonded kernel, so no reduction kernel
 * runs between Force1 and the nonbonded kernel.
 */

#define CULL_SQUARED (CUTOFF_SQUARED*1.0001f)

/**
 * Whether any pair of atoms in blocks x and y can be within the cutoff, from the current block bounds.
 */
DEVICE bool blocksInRange(real4 centerX, real4 sizeX, real4 centerY, real4 sizeY, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    real4 blockDelta = centerY-centerX;
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
    blockDelta.x = max((real) 0, fabs(blockDelta.x)-sizeX.x-sizeY.x);
    blockDelta.y = max((real) 0, fabs(blockDelta.y)-sizeX.y-sizeY.y);
    blockDelta.z = max((real) 0, fabs(blockDelta.z)-sizeX.z-sizeY.z);
    return (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < CULL_SQUARED);
}

/**
 * Decode the index of an unordered block pair (x, y) with x <= y, where pairs are numbered row by row in y.
 */
DEVICE uint2 decodeBlockPair(unsigned int index) {
    unsigned int y = (unsigned int) ((SQRT(8.0f*index+1.0f)-1.0f)*0.5f);
    if (y*(y+1)/2 > index)
        y--;
    else if ((y+1)*(y+2)/2 <= index)
        y++;
    return uint2(index-y*(y+1)/2, y);
}

/**
 * The Born sum term of an atom with offset radius radius1 from an atom with scaled radius scaledRadius2 at distance r.
 */
DEVICE real bornSumTerm(real r, real invR, real radius1, real invRadius1, real scaledRadius2) {
    real rScaledRadiusJ = r+scaledRadius2;
    if (radius1 >= rScaledRadiusJ)
        return 0;
    real lowerLimit = max(radius1, fabs(r-scaledRadius2));
    real invProduct = RECIP(lowerLimit*rScaledRadiusJ);
    real l_ij = rScaledRadiusJ*invProduct;
    real u_ij = lowerLimit*invProduct;
    real l_ij2 = l_ij*l_ij;
    real u_ij2 = u_ij*u_ij;
    real ratio = LOG(u_ij*lowerLimit);
    real term = l_ij - u_ij + (0.50f*invR*ratio) + 0.25f*(r*(u_ij2-l_ij2) + (scaledRadius2*scaledRadius2*invR)*(l_ij2-u_ij2));
    return term + (radius1 < scaledRadius2-r ? 2.0f*(invRadius1-l_ij) : 0);
}

/**
 * The Born sum of this lane's atom in block x over the atoms of the same block.
 */
DEVICE real bornSumOverOwnBlock(unsigned int x, unsigned int tgx, GLOBAL const real4* RESTRICT posq,
        GLOBAL const float2* RESTRICT global_params, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    const unsigned int atom1 = x*TILE_SIZE + tgx;
    const real4 posq1 = posq[atom1];
    const float2 params1 = global_params[atom1];
    const real invRadius1 = RECIP(params1.x);
    real bornSum = 0;
    for (unsigned int j = 0; j < TILE_SIZE; j++) {
        const unsigned int atom2 = x*TILE_SIZE + j;
        const real4 posq2 = simd_broadcast(posq1, (ushort) j);
        const float2 params2 = simd_broadcast(params1, (ushort) j);
        real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_DELTA(delta)
#endif
        real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 != atom2 && r2 < CUTOFF_SQUARED) {
            real invR = RSQRT(r2);
            bornSum += bornSumTerm(r2*invR, invR, params1.x, invRadius1, params2.y);
        }
    }
    return bornSum;
}

/**
 * Compute each atom's Born sum over each block.  A SIMD group owns an unordered block pair (x, y).  Off the diagonal
 * it evaluates the distance of each pair once and both directed terms from it: each lane keeps its x atom's sum in
 * registers, and the y atoms with their sums rotate one lane per step, as in computeGBSAForce1Owned.  The x sums go
 * to slot y and the y sums to slot x of partialBornSum.
 */
KERNEL void computeBornSumOwned(GLOBAL mm_ulong* RESTRICT bornSum, GLOBAL const real4* RESTRICT posq,
        GLOBAL const float2* RESTRICT global_params, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, GLOBAL const real4* RESTRICT blockCenter,
        GLOBAL const real4* RESTRICT blockSize) {
    const unsigned int totalWarps = GLOBAL_SIZE/TILE_SIZE;
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE-1);
    for (unsigned int owner = GLOBAL_ID/TILE_SIZE; owner < NUM_BLOCK_PAIRS; owner += totalWarps) {
        const uint2 pair = decodeBlockPair(owner);
        const unsigned int x = pair.x;
        const unsigned int y = pair.y;
        real sumX = 0, sumY = 0;
        if (blocksInRange(blockCenter[x], blockSize[x], blockCenter[y], blockSize[y], periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ)) {
            if (x == y)
                sumX = bornSumOverOwnBlock(x, tgx, posq, global_params, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
            else {
                const unsigned int atom1 = x*TILE_SIZE + tgx;
                const real4 posq1 = posq[atom1];
                const float2 params1 = global_params[atom1];
                const real invRadius1 = RECIP(params1.x);
                const real4 posqAtomY = posq[y*TILE_SIZE + tgx];
                const float2 paramsAtomY = global_params[y*TILE_SIZE + tgx];
                real3 posY = make_real3(posqAtomY.x, posqAtomY.y, posqAtomY.z);
                real3 paramsY = make_real3(paramsAtomY.x, paramsAtomY.y, RECIP(paramsAtomY.x));
                for (unsigned int j = 0; j < TILE_SIZE; j++) {
                    const unsigned int atom2 = y*TILE_SIZE + ((tgx+j) & (TILE_SIZE-1));
                    real3 delta = posY-make_real3(posq1.x, posq1.y, posq1.z);
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(delta)
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && r2 < CUTOFF_SQUARED) {
                        real invR = RSQRT(r2);
                        real r = r2*invR;
                        sumX += bornSumTerm(r, invR, params1.x, invRadius1, paramsY.y);
                        sumY += bornSumTerm(r, invR, paramsY.x, paramsY.z, params1.y);
                    }
                    posY = simd_shuffle_rotate_down(posY, (ushort) 1);
                    paramsY = simd_shuffle_rotate_down(paramsY, (ushort) 1);
                    sumY = simd_shuffle_rotate_down(sumY, (ushort) 1);
                }
            }
        }
        if (sumX != 0)
            ATOMIC_ADD(&bornSum[x*TILE_SIZE + tgx], (mm_ulong) realToFixedPoint(sumX));
        if (sumY != 0)
            ATOMIC_ADD(&bornSum[y*TILE_SIZE + tgx], (mm_ulong) realToFixedPoint(sumY));
    }
}

/**
 * An atom's charge, Born radius and inverse Born radius from its Born sum, with the OBC chain rule factor in w.
 */
DEVICE real4 bornAtomData(unsigned int index, GLOBAL const mm_long* RESTRICT bornSum, GLOBAL const float2* RESTRICT params,
        GLOBAL const real* RESTRICT charge) {
    float offsetRadius = params[index].x;
    real sum = RECIP((real) 0x100000000)*bornSum[index];
    sum *= 0.5f*offsetRadius;
    real sum2 = sum*sum;
    real sum3 = sum*sum2;
    real tanhSum = tanh(OBC_ALPHA*sum - OBC_BETA*sum2 + OBC_GAMMA*sum3);
    real nonOffsetRadius = offsetRadius + DIELECTRIC_OFFSET;
    real radius = RECIP(RECIP(offsetRadius) - tanhSum/nonOffsetRadius);
    real chain = offsetRadius*(OBC_ALPHA - 2*OBC_BETA*sum + 3*OBC_GAMMA*sum2);
    chain = (1-tanhSum*tanhSum)*chain / nonOffsetRadius;
    return make_real4(charge[index], radius, RECIP(radius), chain);
}

/**
 * The GBSA pair terms of atoms 1 and 2: dE/dr over r in x, dGpol/dalpha2 in y, and the energy in z (with the cutoff
 * shift unless the atoms are the same).
 */
DEVICE real4 gbsaPair(real r2, real4 atomData1, real quarterInvRadius1, real chargeProduct1, real3 atomData2, bool sameAtom) {
    real alpha2_ij = atomData1.y*atomData2.y;
    real D_ij = r2*(quarterInvRadius1*atomData2.z);
    real expTerm = EXP(-D_ij);
    real denominator2 = r2 + alpha2_ij*expTerm;
    real invDenominator = RSQRT(denominator2);
    real scaledChargeProduct = chargeProduct1*atomData2.x;
    real tempEnergy = scaledChargeProduct*invDenominator;
    real Gpol = tempEnergy*invDenominator*invDenominator;
    real dGpol_dalpha2_ij = -0.5f*Gpol*expTerm*(1.0f+D_ij);
    real dEdR = Gpol*(1.0f - 0.25f*expTerm);
    if (!sameAtom)
        tempEnergy -= scaledChargeProduct/CUTOFF;
    return make_real4(dEdR, dGpol_dalpha2_ij, tempEnergy, 0);
}

/**
 * Add one atom's sums over a block pair to the force buffer and bornForce.  The finished Born force is
 * (sum + saTerm/B)*B^2*chain, linear in the sums, so each sum takes the factor B^2*chain here.  Fixed point addition
 * is exact, so the totals don't depend on scheduling.
 */
DEVICE void addAtomForce(GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL mm_ulong* RESTRICT bornForce, unsigned int atom,
        real4 force, real4 atomData) {
    if (atom >= NUM_ATOMS)
        return;
    ATOMIC_ADD(&forceBuffers[atom], (mm_ulong) realToFixedPoint(force.x));
    ATOMIC_ADD(&forceBuffers[atom+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.y));
    ATOMIC_ADD(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.z));
    ATOMIC_ADD(&bornForce[atom], (mm_ulong) realToFixedPoint(force.w*atomData.y*atomData.y*atomData.w));
}

/**
 * Compute the first part of the GBSA interaction, the force in xyz and the Born force in w, over each block pair
 * (x, y) with x <= y.  An off-diagonal pair is evaluated once: each lane keeps its x atom's sum in registers, and the
 * y atoms with their sums rotate one lane per step, so each lane ends holding one y atom's sum.  Each lane then adds
 * both sums with one atomic per value.  A diagonal pair is evaluated from both sides, like the Born sum, and adds each
 * atom's surface area term once.
 */
KERNEL void computeGBSAForce1Owned(GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL mm_ulong* RESTRICT bornForce,
        GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real4* RESTRICT posq, GLOBAL const mm_long* RESTRICT bornSum,
        GLOBAL const float2* RESTRICT params, GLOBAL const real* RESTRICT charge, int needEnergy, real4 periodicBoxSize,
        real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const real4* RESTRICT blockCenter, GLOBAL const real4* RESTRICT blockSize) {
    const unsigned int totalWarps = GLOBAL_SIZE/TILE_SIZE;
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE-1);
    mixed energy = 0;
    for (unsigned int owner = GLOBAL_ID/TILE_SIZE; owner < NUM_BLOCK_PAIRS; owner += totalWarps) {
        const uint2 pair = decodeBlockPair(owner);
        const unsigned int x = pair.x;
        const unsigned int y = pair.y;
        const unsigned int atom1 = x*TILE_SIZE + tgx;
        const unsigned int atomY = y*TILE_SIZE + tgx;
        real4 force = make_real4(0);
        real4 forceY = make_real4(0);
        if (blocksInRange(blockCenter[x], blockSize[x], blockCenter[y], blockSize[y], periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ)) {
            const real4 posq1 = posq[atom1];
            const real4 atomData1 = bornAtomData(atom1, bornSum, params, charge);
            const real chargeProduct1 = PREFACTOR*atomData1.x;
            const real quarterInvRadius1 = 0.25f*atomData1.z;
            const real4 posqAtomY = posq[atomY];
            const real4 atomDataAtomY = (x == y ? atomData1 : bornAtomData(atomY, bornSum, params, charge));
            real3 posY = make_real3(posqAtomY.x, posqAtomY.y, posqAtomY.z);
            real3 atomDataY = make_real3(atomDataAtomY.x, atomDataAtomY.y, atomDataAtomY.z);
            if (x == y) {
                for (unsigned int j = 0; j < TILE_SIZE; j++) {
                    const unsigned int atom2 = y*TILE_SIZE + j;
                    const real3 pos2 = simd_broadcast(posY, (ushort) j);
                    const real3 atomData2 = simd_broadcast(atomDataY, (ushort) j);
                    real3 delta = pos2-make_real3(posq1.x, posq1.y, posq1.z);
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(delta)
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && r2 < CUTOFF_SQUARED) {
                        real4 terms = gbsaPair(r2, atomData1, quarterInvRadius1, chargeProduct1, atomData2, atom1 == atom2);
                        force.w += terms.y*atomData2.y;
                        if (needEnergy)
                            energy += 0.5f*terms.z;
                        delta *= terms.x;
                        force.x -= delta.x;
                        force.y -= delta.y;
                        force.z -= delta.z;
                    }
                }
            }
            else {
                for (unsigned int j = 0; j < TILE_SIZE; j++) {
                    const unsigned int atom2 = y*TILE_SIZE + ((tgx+j) & (TILE_SIZE-1));
                    real3 delta = posY-make_real3(posq1.x, posq1.y, posq1.z);
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(delta)
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && r2 < CUTOFF_SQUARED) {
                        real4 terms = gbsaPair(r2, atomData1, quarterInvRadius1, chargeProduct1, atomDataY, false);
                        force.w += terms.y*atomDataY.y;
                        forceY.w += terms.y*atomData1.y;
                        if (needEnergy)
                            energy += terms.z;
                        delta *= terms.x;
                        force.x -= delta.x;
                        force.y -= delta.y;
                        force.z -= delta.z;
                        forceY.x += delta.x;
                        forceY.y += delta.y;
                        forceY.z += delta.z;
                    }
                    posY = simd_shuffle_rotate_down(posY, (ushort) 1);
                    atomDataY = simd_shuffle_rotate_down(atomDataY, (ushort) 1);
                    forceY = simd_shuffle_rotate_down(forceY, (ushort) 1);
                }
            }
            if (x == y && atom1 < NUM_ATOMS) {
                float offsetRadius = params[atom1].x;
                real r = offsetRadius+DIELECTRIC_OFFSET+PROBE_RADIUS;
                real ratio6 = POW((offsetRadius+DIELECTRIC_OFFSET)/atomData1.y, (real) 6);
                real saTerm = SURFACE_AREA_FACTOR*r*r*ratio6;
                force.w += saTerm/atomData1.y;
                if (needEnergy)
                    energy -= saTerm/6;
            }
            addAtomForce(forceBuffers, bornForce, atom1, force, atomData1);
            if (x != y)
                addAtomForce(forceBuffers, bornForce, atomY, forceY, atomDataAtomY);
        }
    }
    if (needEnergy)
        energyBuffer[GLOBAL_ID] += energy;
}
