/**
 * Load an atom's position and velocity, remove the center of mass velocity if removeCM is set, and apply the velocity
 * step.
 */
inline void kickAtom(int atom, GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL mixed4* RESTRICT velm, GLOBAL const mm_long* RESTRICT force, int paddedNumAtoms, mixed fscale,
        int removeCM, float4 cm, PRIVATE mixed4& pos, PRIVATE mixed4& vel) {
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
        vel.x += fscale*vel.w*force[atom];
        vel.y += fscale*vel.w*force[atom+paddedNumAtoms];
        vel.z += fscale*vel.w*force[atom+paddedNumAtoms*2];
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
 * Correct an atom's velocity for the constraints and record its new position.
 */
inline void finishAtom(int atom, GLOBAL real4* RESTRICT posq, GLOBAL real4* RESTRICT posqCorrection,
        GLOBAL mixed4* RESTRICT velm, mixed invDt, mixed4 pos, mixed4 vel, mixed4 delta, mixed4 oldDelta) {
    vel.x += (delta.x-oldDelta.x)*invDt;
    vel.y += (delta.y-oldDelta.y)*invDt;
    vel.z += (delta.z-oldDelta.z)*invDt;
    velm[atom] = vel;
    pos.x += delta.x;
    pos.y += delta.y;
    pos.z += delta.z;
    storePos(posq, posqCorrection, atom, pos);
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
 */
KERNEL void integrateLangevinMiddleFused(int numSettle, int numShake, int numUnits, int paddedNumAtoms, mixed tol,
        GLOBAL real4* RESTRICT posq, GLOBAL real4* RESTRICT posqCorrection, GLOBAL mixed4* RESTRICT velm,
        GLOBAL const mm_long* RESTRICT force, GLOBAL const mixed* RESTRICT paramBuffer, GLOBAL const mixed2* RESTRICT dt,
        GLOBAL const float4* RESTRICT random, unsigned int randomIndex, GLOBAL const int4* RESTRICT settleAtoms,
        GLOBAL const float2* RESTRICT settleParams, GLOBAL const int4* RESTRICT shakeAtoms,
        GLOBAL const float4* RESTRICT shakeParams, GLOBAL const int* RESTRICT freeAtoms, int removeCM,
        GLOBAL const float4* RESTRICT cmVelocity) {
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
        if (isSettle) {
            atoms = settleAtoms[unit];
            atoms.w = -1; // The host leaves 0 in the unused fourth slot of a SETTLE cluster.
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
            if (atoms[i] != -1)
                kickAtom(atoms[i], posq, posqCorrection, velm, force, paddedNumAtoms, fscale, removeCM, cm, pos[i], vel[i]);
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
        for (int i = 0; i < 4; i++)
            if (atoms[i] != -1 && vel[i].w != 0.0)
                finishAtom(atoms[i], posq, posqCorrection, velm, invDt, pos[i], vel[i], delta[i], oldDelta[i]);
    }
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
        GLOBAL real4* RESTRICT posq, GLOBAL real4* RESTRICT posqCorrection, GLOBAL mixed4* RESTRICT velm,
        GLOBAL const mm_long* RESTRICT force, GLOBAL const mixed* RESTRICT paramBuffer, GLOBAL const mixed2* RESTRICT dt,
        GLOBAL const float4* RESTRICT random, unsigned int randomIndex, GLOBAL const int4* RESTRICT settleAtoms,
        GLOBAL const float2* RESTRICT settleParams, GLOBAL const int4* RESTRICT shakeAtoms,
        GLOBAL const float4* RESTRICT shakeParams, GLOBAL const int* RESTRICT freeAtoms, int removeCM,
        GLOBAL const float4* RESTRICT cmVelocity) {
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
        if (isSettle) {
            atoms = settleAtoms[unit];
            atoms.w = -1; // The host leaves 0 in the unused fourth slot of a SETTLE cluster.
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
        if (atom != -1)
            kickAtom(atom, posq, posqCorrection, velm, force, paddedNumAtoms, fscale, removeCM, cm, pos, vel);
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
        if (atom != -1 && vel.w != 0.0)
            finishAtom(atom, posq, posqCorrection, velm, invDt, pos, vel, delta, oldDelta);
    }
}
