/**
 * Take one LangevinMiddleIntegrator step.  Each thread owns a unit: a SETTLE cluster, a SHAKE cluster, or an atom in
 * neither, taken in that order.  It runs the velocity step, the velocity constraints, the position update with the heat
 * bath, the position constraints and the final velocity correction on its atoms in registers, so the displacements
 * never pass through device memory.  Each atom sees the same operations as in integrateLangevinMiddlePart1-3.
 */
KERNEL void integrateLangevinMiddleFused(int numSettle, int numShake, int numUnits, int paddedNumAtoms, mixed tol,
        GLOBAL real4* RESTRICT posq, GLOBAL real4* RESTRICT posqCorrection, GLOBAL mixed4* RESTRICT velm,
        GLOBAL const mm_long* RESTRICT force, GLOBAL const mixed* RESTRICT paramBuffer, GLOBAL const mixed2* RESTRICT dt,
        GLOBAL const float4* RESTRICT random, unsigned int randomIndex, GLOBAL const int4* RESTRICT settleAtoms,
        GLOBAL const float2* RESTRICT settleParams, GLOBAL const int4* RESTRICT shakeAtoms,
        GLOBAL const float4* RESTRICT shakeParams, GLOBAL const int* RESTRICT freeAtoms) {
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

        // Load the atoms and apply the velocity step.

        mixed4 pos[4], vel[4], delta[4], oldDelta[4];
        for (int i = 0; i < 4; i++) {
            pos[i] = make_mixed4(0);
            vel[i] = make_mixed4(0);
            delta[i] = make_mixed4(0);
            oldDelta[i] = make_mixed4(0);
            int atom = atoms[i];
            if (atom == -1)
                continue;
            pos[i] = loadPos(posq, posqCorrection, atom);
            mixed4 velocity = velm[atom];
            if (velocity.w != 0.0) {
                velocity.x += fscale*velocity.w*force[atom];
                velocity.y += fscale*velocity.w*force[atom+paddedNumAtoms];
                velocity.z += fscale*velocity.w*force[atom+paddedNumAtoms*2];
            }
            vel[i] = velocity;
        }
        if (isSettle)
            applySettleToClusterVelocities(pos, vel);
        else if (isShake)
            applyShakeToClusterVelocities(atoms, shakeParams[unit-numSettle], tol, pos, vel);

        // Move the atoms a half step, interact with the heat bath, and move them another half step.

        for (int i = 0; i < 4; i++) {
            int atom = atoms[i];
            mixed4 velocity = vel[i];
            if (atom == -1 || velocity.w == 0.0)
                continue;
            delta[i] = make_mixed4(halfdt*velocity.x, halfdt*velocity.y, halfdt*velocity.z, 0);
            mixed sqrtInvMass = SQRT(velocity.w);
            float4 noise = random[randomIndex+atom];
            velocity.x = vscale*velocity.x + noisescale*sqrtInvMass*noise.x;
            velocity.y = vscale*velocity.y + noisescale*sqrtInvMass*noise.y;
            velocity.z = vscale*velocity.z + noisescale*sqrtInvMass*noise.z;
            vel[i] = velocity;
            delta[i] += make_mixed4(halfdt*velocity.x, halfdt*velocity.y, halfdt*velocity.z, 0);
            oldDelta[i] = delta[i];
        }
        if (isSettle)
            applySettleToCluster(settleParams[unit], make_mixed3(vel[0].w, vel[1].w, vel[2].w), pos, delta);
        else if (isShake)
            applyShakeToCluster(atoms, shakeParams[unit-numSettle], tol, pos, delta);

        // Correct the velocities for the constraints and record the new positions.

        for (int i = 0; i < 4; i++) {
            int atom = atoms[i];
            mixed4 velocity = vel[i];
            if (atom == -1 || velocity.w == 0.0)
                continue;
            velocity.x += (delta[i].x-oldDelta[i].x)*invDt;
            velocity.y += (delta[i].y-oldDelta[i].y)*invDt;
            velocity.z += (delta[i].z-oldDelta[i].z)*invDt;
            velm[atom] = velocity;
            mixed4 newPos = pos[i];
            newPos.x += delta[i].x;
            newPos.y += delta[i].y;
            newPos.z += delta[i].z;
            storePos(posq, posqCorrection, atom, newPos);
        }
    }
}
