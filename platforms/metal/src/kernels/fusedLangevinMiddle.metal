/**
 * Load an atom's position and velocity, remove the center of mass velocity if removeCM is set, and apply the velocity
 * step.
 */
inline void kickAtom(int atom, GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL mixed4s* RESTRICT velm, GLOBAL const mm_long* RESTRICT force, int paddedNumAtoms, mixed fscale,
        int removeCM, float4 cm, PRIVATE mixed4& pos, PRIVATE mixed4& vel, mm_long shareX, mm_long shareY, mm_long shareZ) {
    pos = loadPos(posq, posqCorrection, atom);
    vel = velm[atom];
    if (removeCM) {
        vel.x -= cm.x;
        vel.y -= cm.y;
        vel.z -= cm.z;
        if (vel.w == 0.0)
            velm[atom] = vel; // The later steps skip massless atoms, but removeCenterOfMassMomentum shifts them too.
    }
    if (vel.w != 0.0) {
        // share is the atom's part of a virtual site's force, in the fixed point distributeVirtualSiteForces adds.
        vel.x += fscale*vel.w*(force[atom]+shareX);
        vel.y += fscale*vel.w*(force[atom+paddedNumAtoms]+shareY);
        vel.z += fscale*vel.w*(force[atom+paddedNumAtoms*2]+shareZ);
    }
}

/**
 * Move an atom a half step, interact with the heat bath, and move it another half step.
 */
inline void bathAtom(int atom, GLOBAL const float4* RESTRICT random, unsigned int randomIndex, mixed halfdt, mixed vscale,
        mixed noisescale, PRIVATE mixed4& vel, PRIVATE mixed4& delta, PRIVATE mixed4& oldDelta) {
    delta = make_mixed4(halfdt*vel.x, halfdt*vel.y, halfdt*vel.z, 0);
    mixed sqrtInvMass = SQRT(vel.w);
    float4 noise = random[randomIndex+atom];
    vel.x = vscale*vel.x + noisescale*sqrtInvMass*noise.x;
    vel.y = vscale*vel.y + noisescale*sqrtInvMass*noise.y;
    vel.z = vscale*vel.z + noisescale*sqrtInvMass*noise.z;
    delta += make_mixed4(halfdt*vel.x, halfdt*vel.y, halfdt*vel.z, 0);
    oldDelta = delta;
}

/**
 * Correct an atom's velocity for the constraints and record its new position, which is also returned.
 */
inline mixed4 finishAtom(int atom, GLOBAL real4* RESTRICT posq, GLOBAL real4* RESTRICT posqCorrection,
        GLOBAL mixed4s* RESTRICT velm, mixed invDt, mixed4 pos, mixed4 vel, mixed4 delta, mixed4 oldDelta) {
    vel.x += (delta.x-oldDelta.x)*invDt;
    vel.y += (delta.y-oldDelta.y)*invDt;
    vel.z += (delta.z-oldDelta.z)*invDt;
    velm[atom] = vel;
    pos.x += delta.x;
    pos.y += delta.y;
    pos.z += delta.z;
    storePos(posq, posqCorrection, atom, pos);
    return pos;
}

#ifdef FOLD_VSITES
/**
 * The position a later loadPos() returns for a position storePos() records: the mixed value rounded through the
 * real and correction halves.  Placing a virtual site from it gives the same position as computeVirtualSites, which
 * reads the stored positions.
 */
inline mixed4 storedPos(mixed4 pos) {
#ifdef USE_MIXED_PRECISION
    real4 pos1 = make_real4((real) pos.x, (real) pos.y, (real) pos.z, (real) pos.w);
    real4 pos2 = make_real4(pos.x-(real) pos.x, pos.y-(real) pos.y, pos.z-(real) pos.z, 0);
    return make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
#else
    return pos;
#endif
}

/**
 * Place a three particle average site from its three atoms' stored positions, the same arithmetic as
 * computeVirtualSites.  site is the site's position as loaded, for its charge in w.
 */
inline void placeVsite(int vsite, GLOBAL real4* RESTRICT posq, GLOBAL real4* RESTRICT posqCorrection, float4 weights,
        mixed4 pos1, mixed4 pos2, mixed4 pos3, mixed4 site) {
    site.x = pos1.x*weights.x + pos2.x*weights.y + pos3.x*weights.z;
    site.y = pos1.y*weights.x + pos2.y*weights.y + pos3.y*weights.z;
    site.z = pos1.z*weights.x + pos2.z*weights.y + pos3.z*weights.z;
    storePos(posq, posqCorrection, vsite, site);
}
#endif

/**
 * Zero an atom's force once the kick has read it.  This is the next step's clear of the force buffer, done by the only
 * thread that reads the atom, so no thread can read a zero it wasn't meant to.
 */
inline void clearAtomForce(GLOBAL mm_long* RESTRICT force, int atom, int paddedNumAtoms) {
    force[atom] = 0;
    force[atom+paddedNumAtoms] = 0;
    force[atom+paddedNumAtoms*2] = 0;
}

/**
 * Zero the first numWords 32 bit words of a buffer with a grid-stride loop, four words per store.
 */
inline void clearWords(GLOBAL int* buffer, int numWords, int id, int size) {
    GLOBAL int4* buffer4 = (GLOBAL int4*) buffer;
    for (int i = id; i < numWords/4; i += size)
        buffer4[i] = make_int4(0);
    int tail = 4*(numWords/4)+id;
    if (tail < numWords)
        buffer[tail] = 0;
}

/**
 * The rest of the next step's clear of the autoclear buffers: the force buffer's padding atoms, which no unit reads,
 * and up to four other buffers.  The host passes 0 words for unused slots.
 */
inline void clearOtherBuffers(GLOBAL mm_long* RESTRICT force, int numAtoms, int paddedNumAtoms, GLOBAL int* clear0,
        GLOBAL int* clear1, GLOBAL int* clear2, GLOBAL int* clear3, int4 clearWords4, int id, int size) {
    int numPadding = paddedNumAtoms-numAtoms;
    for (int i = id; i < 3*numPadding; i += size)
        force[numAtoms+i%numPadding+(i/numPadding)*paddedNumAtoms] = 0;
    clearWords(clear0, clearWords4.x, id, size);
    clearWords(clear1, clearWords4.y, id, size);
    clearWords(clear2, clearWords4.z, id, size);
    clearWords(clear3, clearWords4.w, id, size);
}

/**
 * Take one LangevinMiddleIntegrator step.  Each thread owns a unit: a SETTLE cluster, a SHAKE cluster, or an atom in
 * neither, taken in that order.  It runs the velocity step, the velocity constraints, the position update with the heat
 * bath, the position constraints and the final velocity correction on its atoms in registers, so the displacements
 * never pass through device memory.  Each atom sees the same operations as in integrateLangevinMiddlePart1-3.
 *
 * When removeCM is nonzero, it also does removeCenterOfMassMomentum's work: it subtracts the
 * center of mass velocity that calcCenterOfMassVelocity wrote to cmVelocity[0] from every atom as it loads it.  That
 * saves removeCenterOfMassMomentum's dispatch and its pass over velm.
 *
 * When clearBuffers is nonzero, it also does the next step's clearAutoclearBuffers(): each thread zeroes the forces it
 * read, and the grid clears the other autoclear buffers, which nothing reads after the force pass.  That saves the
 * clear's dispatch and the boundary between it and this kernel.
 */
KERNEL void integrateLangevinMiddleFused(int numSettle, int numShake, int numUnits, int paddedNumAtoms, mixed tol,
        GLOBAL real4* RESTRICT posq, GLOBAL real4* RESTRICT posqCorrection, GLOBAL mixed4s* RESTRICT velm,
        GLOBAL mm_long* RESTRICT force, GLOBAL const mixed* RESTRICT paramBuffer, GLOBAL const mixed2* RESTRICT dt,
        GLOBAL const float4* RESTRICT random, unsigned int randomIndex, GLOBAL const int4* RESTRICT settleAtoms,
        GLOBAL const float2* RESTRICT settleParams, GLOBAL const int4* RESTRICT shakeAtoms,
        GLOBAL const float4* RESTRICT shakeParams, GLOBAL const int* RESTRICT freeAtoms, int removeCM,
        GLOBAL const float4* RESTRICT cmVelocity, int clearBuffers, int numAtoms, GLOBAL int* clear0, GLOBAL int* clear1,
        GLOBAL int* clear2, GLOBAL int* clear3, int4 clearWords4
#ifdef FOLD_VSITES
        , GLOBAL const int* RESTRICT settleVsites, GLOBAL const float4* RESTRICT settleVsiteWeights, int applyVsiteShare
#endif
        ) {
    float4 cm = (removeCM ? cmVelocity[0] : make_float4(0));
    mixed fscale = dt[0].y/(mixed) 0x100000000;
    mixed vscale = paramBuffer[0];
    mixed noisescale = paramBuffer[1];
    mixed halfdt = 0.5f*dt[0].y;
    mixed invDt = 1/dt[0].y;
    for (int unit = GLOBAL_ID; unit < numUnits; unit += GLOBAL_SIZE) {
        bool isSettle = (unit < numSettle);
        bool isShake = (!isSettle && unit < numSettle+numShake);
        int4 atoms;
        int vsite = -1;
        float4 vsiteWeights = make_float4(0);
        mm_long share[4][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        if (isSettle) {
            atoms = settleAtoms[unit];
            atoms.w = -1; // The host leaves 0 in the unused fourth slot of a SETTLE cluster.
#ifdef FOLD_VSITES
            // A cluster's virtual site rides in that slot: its force goes to the three atoms before their kick, with
            // the rounding of distributeVirtualSiteForces, and the site is placed from their final positions below.
            vsite = settleVsites[unit];
            if (vsite != -1) {
                atoms.w = vsite;
                vsiteWeights = settleVsiteWeights[unit];
                if (applyVsiteShare) {
                    real3 f = loadForce(vsite, force);
                    share[0][0] = realToFixedPoint(f.x*vsiteWeights.x);
                    share[0][1] = realToFixedPoint(f.y*vsiteWeights.x);
                    share[0][2] = realToFixedPoint(f.z*vsiteWeights.x);
                    share[1][0] = realToFixedPoint(f.x*vsiteWeights.y);
                    share[1][1] = realToFixedPoint(f.y*vsiteWeights.y);
                    share[1][2] = realToFixedPoint(f.z*vsiteWeights.y);
                    share[2][0] = realToFixedPoint(f.x*vsiteWeights.z);
                    share[2][1] = realToFixedPoint(f.y*vsiteWeights.z);
                    share[2][2] = realToFixedPoint(f.z*vsiteWeights.z);
                }
            }
#endif
        }
        else if (isShake)
            atoms = shakeAtoms[unit-numSettle];
        else
            atoms = make_int4(freeAtoms[unit-numSettle-numShake], -1, -1, -1);
        mixed4 pos[4], vel[4], delta[4], oldDelta[4];
        for (int i = 0; i < 4; i++) {
            pos[i] = make_mixed4(0);
            vel[i] = make_mixed4(0);
            delta[i] = make_mixed4(0);
            oldDelta[i] = make_mixed4(0);
            if (atoms[i] != -1) {
                kickAtom(atoms[i], posq, posqCorrection, velm, force, paddedNumAtoms, fscale, removeCM, cm, pos[i], vel[i],
                        share[i][0], share[i][1], share[i][2]);
                if (clearBuffers)
                    clearAtomForce(force, atoms[i], paddedNumAtoms);
            }
        }
        if (isSettle)
            applySettleToClusterVelocities(pos, vel);
        else if (isShake)
            applyShakeToClusterVelocities(atoms, shakeParams[unit-numSettle], tol, pos, vel);
        for (int i = 0; i < 4; i++)
            if (atoms[i] != -1 && vel[i].w != 0.0)
                bathAtom(atoms[i], random, randomIndex, halfdt, vscale, noisescale, vel[i], delta[i], oldDelta[i]);
        if (isSettle)
            applySettleToCluster(settleParams[unit], make_mixed3(vel[0].w, vel[1].w, vel[2].w), pos, delta);
        else if (isShake)
            applyShakeToCluster(atoms, shakeParams[unit-numSettle], tol, pos, delta);
        mixed4 finalPos[4];
        for (int i = 0; i < 4; i++) {
            finalPos[i] = pos[i];
            if (atoms[i] != -1 && vel[i].w != 0.0)
                finalPos[i] = finishAtom(atoms[i], posq, posqCorrection, velm, invDt, pos[i], vel[i], delta[i], oldDelta[i]);
        }
#ifdef FOLD_VSITES
        if (vsite != -1)
            placeVsite(vsite, posq, posqCorrection, vsiteWeights, storedPos(finalPos[0]), storedPos(finalPos[1]),
                    storedPos(finalPos[2]), pos[3]);
#endif
    }    if (clearBuffers)
        clearOtherBuffers(force, numAtoms, paddedNumAtoms, clear0, clear1, clear2, clear3, clearWords4, GLOBAL_ID, GLOBAL_SIZE);
}

#ifdef USE_MIXED_PRECISION
inline mixed quad_shuffle(mixed v, ushort lane) { return mixed(quad_shuffle(v.hi, lane), quad_shuffle(v.lo, lane)); }
#endif

inline mixed4 shuffleMixed4(mixed4 v, ushort lane) {
    return make_mixed4(quad_shuffle(v.x, lane), quad_shuffle(v.y, lane), quad_shuffle(v.z, lane), quad_shuffle(v.w, lane));
}

// A select chain, so the private arrays are never indexed by a lane-dependent value and stay in registers.
inline mixed4 laneElement(PRIVATE const mixed4* values, int lane) {
    return lane == 0 ? values[0] : (lane == 1 ? values[1] : (lane == 2 ? values[2] : values[3]));
}

/**
 * The same step with each quad of threads owning a unit: a SETTLE cluster, a SHAKE cluster, or four atoms in neither
 * (freeAtoms is padded with -1 to a multiple of four).  Each lane kicks, heat-baths and finishes its own atom, so that
 * work runs four wide.  For a cluster, every lane gathers the four atoms and runs the same constraint code, then keeps
 * its own atom's result.  Each atom sees the same operations as in integrateLangevinMiddleFused.
 */
KERNEL void integrateLangevinMiddleFusedQuads(int numSettle, int numShake, int numUnits, int paddedNumAtoms, mixed tol,
        GLOBAL real4* RESTRICT posq, GLOBAL real4* RESTRICT posqCorrection, GLOBAL mixed4s* RESTRICT velm,
        GLOBAL mm_long* RESTRICT force, GLOBAL const mixed* RESTRICT paramBuffer, GLOBAL const mixed2* RESTRICT dt,
        GLOBAL const float4* RESTRICT random, unsigned int randomIndex, GLOBAL const int4* RESTRICT settleAtoms,
        GLOBAL const float2* RESTRICT settleParams, GLOBAL const int4* RESTRICT shakeAtoms,
        GLOBAL const float4* RESTRICT shakeParams, GLOBAL const int* RESTRICT freeAtoms, int removeCM,
        GLOBAL const float4* RESTRICT cmVelocity, int clearBuffers, int numAtoms, GLOBAL int* clear0, GLOBAL int* clear1,
        GLOBAL int* clear2, GLOBAL int* clear3, int4 clearWords4
#ifdef FOLD_VSITES
        , GLOBAL const int* RESTRICT settleVsites, GLOBAL const float4* RESTRICT settleVsiteWeights, int applyVsiteShare
#endif
        ) {
    float4 cm = (removeCM ? cmVelocity[0] : make_float4(0));
    mixed fscale = dt[0].y/(mixed) 0x100000000;
    mixed vscale = paramBuffer[0];
    mixed noisescale = paramBuffer[1];
    mixed halfdt = 0.5f*dt[0].y;
    mixed invDt = 1/dt[0].y;
    const int lane = GLOBAL_ID & 3;
    for (int unit = GLOBAL_ID/4; unit < numUnits; unit += GLOBAL_SIZE/4) {
        bool isSettle = (unit < numSettle);
        bool isShake = (!isSettle && unit < numSettle+numShake);
        int4 atoms = make_int4(-1);
        int atom;
        int vsite = -1;
        float4 vsiteWeights = make_float4(0);
        mm_long shareX = 0, shareY = 0, shareZ = 0;
        if (isSettle) {
            atoms = settleAtoms[unit];
            atoms.w = -1; // The host leaves 0 in the unused fourth slot of a SETTLE cluster.
#ifdef FOLD_VSITES
            // A cluster's virtual site rides in that slot, on lane 3: lane 3 reads its force, every lane takes its
            // atom's share with the rounding of distributeVirtualSiteForces, and lane 3 places the site at the end.
            vsite = settleVsites[unit];
            if (vsite != -1) {
                atoms.w = vsite;
                vsiteWeights = settleVsiteWeights[unit];
                if (applyVsiteShare) {
                    real3 f = (lane == 3 ? loadForce(vsite, force) : make_real3(0));
                    f = make_real3(quad_shuffle(f.x, 3), quad_shuffle(f.y, 3), quad_shuffle(f.z, 3));
                    real w = (lane == 0 ? vsiteWeights.x : (lane == 1 ? vsiteWeights.y : (lane == 2 ? vsiteWeights.z : 0)));
                    if (lane < 3) {
                        shareX = realToFixedPoint(f.x*w);
                        shareY = realToFixedPoint(f.y*w);
                        shareZ = realToFixedPoint(f.z*w);
                    }
                }
            }
#endif
            atom = atoms[lane];
        }
        else if (isShake) {
            atoms = shakeAtoms[unit-numSettle];
            atom = atoms[lane];
        }
        else
            atom = freeAtoms[4*(unit-numSettle-numShake)+lane];
        mixed4 pos = make_mixed4(0);
        mixed4 vel = make_mixed4(0);
        mixed4 delta = make_mixed4(0);
        mixed4 oldDelta = make_mixed4(0);
        if (atom != -1) {
            kickAtom(atom, posq, posqCorrection, velm, force, paddedNumAtoms, fscale, removeCM, cm, pos, vel, shareX, shareY, shareZ);
            if (clearBuffers)
                clearAtomForce(force, atom, paddedNumAtoms);
        }
        mixed4 clusterPos[4], clusterVel[4];
        if (isSettle || isShake) {
            for (int i = 0; i < 4; i++) {
                clusterPos[i] = shuffleMixed4(pos, i);
                clusterVel[i] = shuffleMixed4(vel, i);
            }
            if (isSettle)
                applySettleToClusterVelocities(clusterPos, clusterVel);
            else
                applyShakeToClusterVelocities(atoms, shakeParams[unit-numSettle], tol, clusterPos, clusterVel);
            vel = laneElement(clusterVel, lane);
        }
        if (atom != -1 && vel.w != 0.0)
            bathAtom(atom, random, randomIndex, halfdt, vscale, noisescale, vel, delta, oldDelta);
        if (isSettle || isShake) {
            mixed4 clusterDelta[4];
            for (int i = 0; i < 4; i++)
                clusterDelta[i] = shuffleMixed4(delta, i);
            if (isSettle)
                applySettleToCluster(settleParams[unit], make_mixed3(clusterVel[0].w, clusterVel[1].w, clusterVel[2].w), clusterPos, clusterDelta);
            else
                applyShakeToCluster(atoms, shakeParams[unit-numSettle], tol, clusterPos, clusterDelta);
            delta = laneElement(clusterDelta, lane);
        }
        mixed4 finalPos = pos;
        if (atom != -1 && vel.w != 0.0)
            finalPos = finishAtom(atom, posq, posqCorrection, velm, invDt, pos, vel, delta, oldDelta);
#ifdef FOLD_VSITES
        if (vsite != -1) {
            mixed4 stored = storedPos(finalPos);
            mixed4 pos1 = shuffleMixed4(stored, 0);
            mixed4 pos2 = shuffleMixed4(stored, 1);
            mixed4 pos3 = shuffleMixed4(stored, 2);
            if (lane == 3)
                placeVsite(vsite, posq, posqCorrection, vsiteWeights, pos1, pos2, pos3, pos);
        }
#endif
    }    if (clearBuffers)
        clearOtherBuffers(force, numAtoms, paddedNumAtoms, clear0, clear1, clear2, clear3, clearWords4, GLOBAL_ID, GLOBAL_SIZE);
}
