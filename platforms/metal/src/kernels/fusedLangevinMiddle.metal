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

#ifdef EMULATED_MIXED_PRECISION
/**
 * Solve the 3x3 system a l = g for the multipliers of a cluster's three bonds by cofactors, as applySettleToCluster
 * does.  A cluster with fewer bonds passes identity rows and zero residuals for the missing ones and gets zero
 * multipliers there.
 */
inline real3 solveBondMultipliers(real a11, real a12, real a13, real a21, real a22, real a23, real a31, real a32, real a33,
        real g1, real g2, real g3) {
    real c11 = a22*a33-a23*a32, c12 = a23*a31-a21*a33, c13 = a21*a32-a22*a31;
    real scale = 1/(a11*c11 + a12*c12 + a13*c13);
    real l1 = (g1*c11 + g2*(a13*a32-a12*a33) + g3*(a12*a23-a13*a22))*scale;
    real l2 = (g1*c12 + g2*(a11*a33-a13*a31) + g3*(a13*a21-a11*a23))*scale;
    real l3 = (g1*c13 + g2*(a12*a31-a11*a32) + g3*(a11*a22-a12*a21))*scale;
    return make_real3(l1, l2, l3);
}

/**
 * A SHAKE cluster's position constraints solved together by Newton's method, the way applySettleToCluster solves a
 * water in emulated mixed precision: one linear solve per iteration for the multipliers of the 1 to 3 bonds, in
 * place of applyShakeToCluster's Gauss-Seidel sweep over them.  The bonds couple through the central atom, and the
 * sweep converges slowly when the hydrogens are heavy (about 9 sweeps on a 4 fs HMR unit); Newton takes 2 or 3
 * steps at any mass.  Iterations run in real precision until the residuals are near its resolution, then one step
 * from residuals in mixed precision, so the constraint error ends near 1e-12 of the squared bond length, below the
 * tolerance the sweep stops at.  The same equations, another path: the displacements are not bitwise with the sweep.
 */
inline void applyShakeToClusterNewton(int4 atoms, float4 params, PRIVATE const mixed4* clusterPos, PRIVATE mixed4* delta) {
    mixed4 pos = clusterPos[0];
    mixed4 xpi = delta[0];
    mixed4 xpj1 = delta[1], xpj2 = delta[2], xpj3 = delta[3];
    real invMassCentral = params.x;
    real d2 = params.z;
    real invMassPeripheral = params.w;
    bool has2 = (atoms.z != -1), has3 = (atoms.w != -1);

    // Bond vectors at the start of the step, and the residuals' constant parts, as in applyShakeToCluster.

    mixed3 rij1 = make_mixed3(pos.x-clusterPos[1].x, pos.y-clusterPos[1].y, pos.z-clusterPos[1].z);
    mixed3 rij2 = make_mixed3(pos.x-clusterPos[2].x, pos.y-clusterPos[2].y, pos.z-clusterPos[2].z);
    mixed3 rij3 = make_mixed3(pos.x-clusterPos[3].x, pos.y-clusterPos[3].y, pos.z-clusterPos[3].z);
    mixed ld1 = d2-(rij1.x*rij1.x + rij1.y*rij1.y + rij1.z*rij1.z);
    mixed ld2 = d2-(rij2.x*rij2.x + rij2.y*rij2.y + rij2.z*rij2.z);
    mixed ld3 = d2-(rij3.x*rij3.x + rij3.y*rij3.y + rij3.z*rij3.z);
    real3 r1 = make_real3((real) rij1.x, (real) rij1.y, (real) rij1.z);
    real3 r2 = (has2 ? make_real3((real) rij2.x, (real) rij2.y, (real) rij2.z) : make_real3(0));
    real3 r3 = (has3 ? make_real3((real) rij3.x, (real) rij3.y, (real) rij3.z) : make_real3(0));
    real3 p1 = r1+make_real3((real) (xpi.x-xpj1.x), (real) (xpi.y-xpj1.y), (real) (xpi.z-xpj1.z));
    real3 p2 = r2+make_real3((real) (xpi.x-xpj2.x), (real) (xpi.y-xpj2.y), (real) (xpi.z-xpj2.z));
    real3 p3 = r3+make_real3((real) (xpi.x-xpj3.x), (real) (xpi.y-xpj3.y), (real) (xpi.z-xpj3.z));
    real3 dxi = make_real3(0), dxj1 = make_real3(0), dxj2 = make_real3(0), dxj3 = make_real3(0);
    bool last = false;
    for (int iteration = 0; !last; iteration++) {
        real3 f1 = p1+dxi-dxj1;
        real3 f2 = p2+dxi-dxj2;
        real3 f3 = p3+dxi-dxj3;
        real g1 = d2-dot(f1, f1);
        real g2 = (has2 ? d2-dot(f2, f2) : 0);
        real g3 = (has3 ? d2-dot(f3, f3) : 0);
        // After 15 iterations the loop stops without an error, even if the residuals have not converged.
        last = (iteration == 14 || (fabs(g1) < 1e-6f*d2 && fabs(g2) < 1e-6f*d2 && fabs(g3) < 1e-6f*d2));
        if (last) {
            // Move the corrections so far into the mixed precision displacements, so the last step is not
            // rounded to the resolution of the earlier ones, and measure the residuals in mixed precision.

            xpi.x += dxi.x;
            xpi.y += dxi.y;
            xpi.z += dxi.z;
            xpj1.x += dxj1.x;
            xpj1.y += dxj1.y;
            xpj1.z += dxj1.z;
            xpj2.x += dxj2.x;
            xpj2.y += dxj2.y;
            xpj2.z += dxj2.z;
            xpj3.x += dxj3.x;
            xpj3.y += dxj3.y;
            xpj3.z += dxj3.z;
            dxi = make_real3(0);
            dxj1 = make_real3(0);
            dxj2 = make_real3(0);
            dxj3 = make_real3(0);
            mixed3 rp = make_mixed3(xpi.x-xpj1.x, xpi.y-xpj1.y, xpi.z-xpj1.z);
            g1 = (real) (ld1 - 2.0f*(rij1.x*rp.x + rij1.y*rp.y + rij1.z*rp.z) - (rp.x*rp.x + rp.y*rp.y + rp.z*rp.z));
            if (has2) {
                rp = make_mixed3(xpi.x-xpj2.x, xpi.y-xpj2.y, xpi.z-xpj2.z);
                g2 = (real) (ld2 - 2.0f*(rij2.x*rp.x + rij2.y*rp.y + rij2.z*rp.z) - (rp.x*rp.x + rp.y*rp.y + rp.z*rp.z));
            }
            if (has3) {
                rp = make_mixed3(xpi.x-xpj3.x, xpi.y-xpj3.y, xpi.z-xpj3.z);
                g3 = (real) (ld3 - 2.0f*(rij3.x*rp.x + rij3.y*rp.y + rij3.z*rp.z) - (rp.x*rp.x + rp.y*rp.y + rp.z*rp.z));
            }
        }

        // Linearize: bond k moves by invMassCentral*sum_j r_j*l_j on the central atom and invMassPeripheral*r_k*l_k
        // on its own atom, so 2*f_k.(that) = g_k.  The factor 2 is in the residuals' half below.

        real a11 = dot(f1, r1)*(invMassCentral+invMassPeripheral);
        real a12 = (has2 ? dot(f1, r2)*invMassCentral : 0), a13 = (has3 ? dot(f1, r3)*invMassCentral : 0);
        real a21 = (has2 ? dot(f2, r1)*invMassCentral : 0), a22 = (has2 ? dot(f2, r2)*(invMassCentral+invMassPeripheral) : 1);
        real a23 = (has3 ? dot(f2, r3)*invMassCentral : 0);
        real a31 = (has3 ? dot(f3, r1)*invMassCentral : 0), a32 = (has3 ? dot(f3, r2)*invMassCentral : 0);
        real a33 = (has3 ? dot(f3, r3)*(invMassCentral+invMassPeripheral) : 1);
        real3 l = solveBondMultipliers(a11, a12, a13, a21, a22, a23, a31, a32, a33, 0.5f*g1, 0.5f*g2, 0.5f*g3);
        dxi += (r1*l.x + r2*l.y + r3*l.z)*invMassCentral;
        dxj1 -= r1*(l.x*invMassPeripheral);
        dxj2 -= r2*(l.y*invMassPeripheral);
        dxj3 -= r3*(l.z*invMassPeripheral);
    }
    xpi.x += dxi.x;
    xpi.y += dxi.y;
    xpi.z += dxi.z;
    xpj1.x += dxj1.x;
    xpj1.y += dxj1.y;
    xpj1.z += dxj1.z;
    xpj2.x += dxj2.x;
    xpj2.y += dxj2.y;
    xpj2.z += dxj2.z;
    xpj3.x += dxj3.x;
    xpj3.y += dxj3.y;
    xpj3.z += dxj3.z;
    delta[0] = xpi;
    delta[1] = xpj1;
    delta[2] = xpj2;
    delta[3] = xpj3;
}

/**
 * A SHAKE cluster's velocity constraints solved together: they are linear in the multipliers, so one solve removes
 * the relative velocity along every bond at once, in place of applyShakeToClusterVelocities' sweeps.  The relative
 * velocities are measured in mixed precision and the solve runs in real, which leaves about 1e-6 of them; the next
 * pass measures what is left and removes it, until a pass's multipliers are all within the tolerance, the sweep's
 * own stopping rule.  Two or three passes.
 */
inline void applyShakeToClusterVelocitiesSolve(int4 atoms, float4 params, mixed tol, PRIVATE const mixed4* clusterPos, PRIVATE mixed4* vel) {
    mixed4 pos = clusterPos[0];
    mixed4 vi = vel[0];
    mixed4 vj1 = vel[1], vj2 = vel[2], vj3 = vel[3];
    real invMassCentral = params.x;
    real invMassPeripheral = params.w;
    bool has2 = (atoms.z != -1), has3 = (atoms.w != -1);
    mixed3 rij1 = make_mixed3(pos.x-clusterPos[1].x, pos.y-clusterPos[1].y, pos.z-clusterPos[1].z);
    mixed3 rij2 = make_mixed3(pos.x-clusterPos[2].x, pos.y-clusterPos[2].y, pos.z-clusterPos[2].z);
    mixed3 rij3 = make_mixed3(pos.x-clusterPos[3].x, pos.y-clusterPos[3].y, pos.z-clusterPos[3].z);
    real3 r1 = make_real3((real) rij1.x, (real) rij1.y, (real) rij1.z);
    real3 r2 = (has2 ? make_real3((real) rij2.x, (real) rij2.y, (real) rij2.z) : make_real3(0));
    real3 r3 = (has3 ? make_real3((real) rij3.x, (real) rij3.y, (real) rij3.z) : make_real3(0));

    // Bond k's relative velocity changes by invMassCentral*sum_j r_k.r_j*l_j + invMassPeripheral*r_k.r_k*l_k.

    real a11 = dot(r1, r1)*(invMassCentral+invMassPeripheral);
    real a12 = (has2 ? dot(r1, r2)*invMassCentral : 0), a13 = (has3 ? dot(r1, r3)*invMassCentral : 0);
    real a22 = (has2 ? dot(r2, r2)*(invMassCentral+invMassPeripheral) : 1);
    real a23 = (has3 ? dot(r2, r3)*invMassCentral : 0);
    real a33 = (has3 ? dot(r3, r3)*(invMassCentral+invMassPeripheral) : 1);
    real realTol = (real) tol;
    bool converged = false;
    for (int pass = 0; pass < 15 && !converged; pass++) {
        mixed3 rv = make_mixed3(vi.x-vj1.x, vi.y-vj1.y, vi.z-vj1.z);
        real g1 = -(real) (rij1.x*rv.x + rij1.y*rv.y + rij1.z*rv.z);
        real g2 = 0, g3 = 0;
        if (has2) {
            rv = make_mixed3(vi.x-vj2.x, vi.y-vj2.y, vi.z-vj2.z);
            g2 = -(real) (rij2.x*rv.x + rij2.y*rv.y + rij2.z*rv.z);
        }
        if (has3) {
            rv = make_mixed3(vi.x-vj3.x, vi.y-vj3.y, vi.z-vj3.z);
            g3 = -(real) (rij3.x*rv.x + rij3.y*rv.y + rij3.z*rv.z);
        }
        real3 l = solveBondMultipliers(a11, a12, a13, a12, a22, a23, a13, a23, a33, g1, g2, g3);
        real3 dvi = (r1*l.x + r2*l.y + r3*l.z)*invMassCentral;
        vi.x += dvi.x;
        vi.y += dvi.y;
        vi.z += dvi.z;
        vj1.x -= r1.x*(l.x*invMassPeripheral);
        vj1.y -= r1.y*(l.x*invMassPeripheral);
        vj1.z -= r1.z*(l.x*invMassPeripheral);
        vj2.x -= r2.x*(l.y*invMassPeripheral);
        vj2.y -= r2.y*(l.y*invMassPeripheral);
        vj2.z -= r2.z*(l.y*invMassPeripheral);
        vj3.x -= r3.x*(l.z*invMassPeripheral);
        vj3.y -= r3.y*(l.z*invMassPeripheral);
        vj3.z -= r3.z*(l.z*invMassPeripheral);
        converged = (fabs(l.x) <= realTol && fabs(l.y) <= realTol && fabs(l.z) <= realTol);
    }
    vel[0] = vi;
    vel[1] = vj1;
    vel[2] = vj2;
    vel[3] = vj3;
}
#endif

/**
 * The SHAKE cluster code the fused kernels run: the joint solves in emulated mixed precision, the common sweeps
 * otherwise.
 */
inline void applyShakeToClusterFused(int4 atoms, float4 params, mixed tol, PRIVATE const mixed4* clusterPos, PRIVATE mixed4* delta) {
#ifdef EMULATED_MIXED_PRECISION
    applyShakeToClusterNewton(atoms, params, clusterPos, delta);
#else
    applyShakeToCluster(atoms, params, tol, clusterPos, delta);
#endif
}

inline void applyShakeToClusterVelocitiesFused(int4 atoms, float4 params, mixed tol, PRIVATE const mixed4* clusterPos, PRIVATE mixed4* vel) {
#ifdef EMULATED_MIXED_PRECISION
    applyShakeToClusterVelocitiesSolve(atoms, params, tol, clusterPos, vel);
#else
    applyShakeToClusterVelocities(atoms, params, tol, clusterPos, vel);
#endif
}

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
            applyShakeToClusterVelocitiesFused(atoms, shakeParams[unit-numSettle], tol, pos, vel);
        for (int i = 0; i < 4; i++)
            if (atoms[i] != -1 && vel[i].w != 0.0)
                bathAtom(atoms[i], random, randomIndex, halfdt, vscale, noisescale, vel[i], delta[i], oldDelta[i]);
        if (isSettle)
            applySettleToCluster(settleParams[unit], make_mixed3(vel[0].w, vel[1].w, vel[2].w), pos, delta);
        else if (isShake)
            applyShakeToClusterFused(atoms, shakeParams[unit-numSettle], tol, pos, delta);
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

#ifdef EMULATED_MIXED_PRECISION
inline mixed3 shuffleMixed3(mixed3 v, ushort lane) {
    return make_mixed3(quad_shuffle(v.x, lane), quad_shuffle(v.y, lane), quad_shuffle(v.z, lane));
}

inline real3 shuffleReal3(real3 v, ushort lane) {
    return make_real3(quad_shuffle(v.x, lane), quad_shuffle(v.y, lane), quad_shuffle(v.z, lane));
}

/**
 * applySettleToClusterVelocities spread over the quad.  Lane i holds atom i and works bond i, AB, BC or CA, from its
 * atom to the next.  Every value is the same expression on the same operands as in the cluster form, computed once in
 * the lane that owns it and fetched by the lanes that need it, so the velocities come out bitwise; the cluster form
 * computes each of them four times.  Lane 3 (a virtual site or nothing) runs along on junk that no lane reads, and
 * keeps its velocity.
 */
inline void applySettleToVelocitiesQuad(int lane, mixed4 pos, PRIVATE mixed4& vel) {
    const ushort next = (lane+1)%3;
    const ushort prev = (lane+2)%3;
    real m = 1/(real) vel.w;
    mixed3 posNext = shuffleMixed3(make_mixed3(pos.x, pos.y, pos.z), next);
    mixed3 b = make_mixed3(posNext.x-pos.x, posNext.y-pos.y, posNext.z-pos.z);
    real3 e = make_real3((real) b.x, (real) b.y, (real) b.z);
    real invLength = RSQRT(e.x*e.x + e.y*e.y + e.z*e.z);
    e *= invLength;
    real3 ePrev = shuffleReal3(e, prev);
    // cA = -(eAB.eCA), cB = -(eAB.eBC), cC = -(eBC.eCA): each bond with the one before it.
    real c = -(e.x*ePrev.x + e.y*ePrev.y + e.z*ePrev.z);
    real mA = quad_shuffle(m, 0), mB = quad_shuffle(m, 1), mC = quad_shuffle(m, 2);
    real cA = quad_shuffle(c, 0), cB = quad_shuffle(c, 1), cC = quad_shuffle(c, 2);
    real s2A = 1-cA*cA;
    real s2B = 1-cB*cB;
    real s2C = 1-cC*cC;
    real mABCinv = 1/(mA*mB*mC);
    real denom = (((s2A*mB+s2B*mA)*mC+(s2A*mB*mB+2*(cA*cB*cC+1)*mA*mB+s2B*mA*mA))*mC+s2C*mA*mB*(mA+mB))*mABCinv;
    real invMass = (real) vel.w;
    for (int pass = 0; pass < 2; pass++) {
        mixed3 velNext = shuffleMixed3(make_mixed3(vel.x, vel.y, vel.z), next);
        real v = (real) ((velNext.x-vel.x)*b.x + (velNext.y-vel.y)*b.y + (velNext.z-vel.z)*b.z)*invLength;
        real vAB = quad_shuffle(v, 0), vBC = quad_shuffle(v, 1), vCA = quad_shuffle(v, 2);
        real t; // tab, tbc or tca: the multiplier of this lane's bond.
        if (lane == 0)
            t = ((cB*cC*mA-cA*mB-cA*mC)*vCA + (cA*cC*mB-cB*mC-cB*mA)*vBC + (s2C*mA*mA*mB*mB*mABCinv+(mA+mB+mC))*vAB)/denom;
        else if (lane == 1)
            t = ((cA*cB*mC-cC*mB-cC*mA)*vCA + (s2A*mB*mB*mC*mC*mABCinv+(mA+mB+mC))*vBC + (cA*cC*mB-cB*mA-cB*mC)*vAB)/denom;
        else
            t = ((s2B*mA*mA*mC*mC*mABCinv+(mA+mB+mC))*vCA + (cA*cB*mC-cC*mB-cC*mA)*vBC + (cB*cC*mA-cA*mB-cA*mC)*vAB)/denom;
        real tPrev = quad_shuffle(t, prev);
        if (lane < 3) {
            vel.x += (t*e.x - tPrev*ePrev.x)*invMass;
            vel.y += (t*e.y - tPrev*ePrev.y)*invMass;
            vel.z += (t*e.z - tPrev*ePrev.z)*invMass;
        }
    }
}

/**
 * applySettleToCluster, the Newton form of emulated mixed precision, spread over the quad.  Lane i holds atom i and
 * its displacement.  Lanes 0 and 2 work bonds 01 and 02 in the cluster form's orientation, from atom 0 outward;
 * lane 1 derives bond 12 from those two, as the cluster form does.  Every value is the same expression on the same
 * operands as there, so the displacements come out bitwise.  Lane 3 runs along on junk and keeps its displacement.
 */
inline void applySettleToPositionsQuad(int lane, float2 params, mixed4 pos, mixed4 vel, PRIVATE mixed4& delta) {
    // Lane 0's partner is atom 1 and lane 2's is atom 0; lane 1 fetches from lanes 0 and 2 instead.
    const ushort partner = (lane == 0 ? 1 : 0);
    real w = (real) vel.w;
    real w0 = quad_shuffle(w, 0), w1 = quad_shuffle(w, 1), w2 = quad_shuffle(w, 2);
    mixed3 xp = make_mixed3(delta.x, delta.y, delta.z);
    mixed3 posPartner = shuffleMixed3(make_mixed3(pos.x, pos.y, pos.z), partner);
    mixed3 xpPartner = shuffleMixed3(xp, partner);
    // b01 = apos1-apos0 in lane 0, b02 = apos2-apos0 in lane 2.
    mixed3 b = (lane == 0 ? make_mixed3(posPartner.x-pos.x, posPartner.y-pos.y, posPartner.z-pos.z)
                          : make_mixed3(pos.x-posPartner.x, pos.y-posPartner.y, pos.z-posPartner.z));
    real3 s = make_real3((real) b.x, (real) b.y, (real) b.z);
    real3 s01 = shuffleReal3(s, 0);
    real3 s02 = shuffleReal3(s, 2);
    real3 s12 = s02-s01;
    if (lane == 1)
        s = s12;
    // r01 = s01+(real)(xp1-xp0) in lane 0, r02 = s02+(real)(xp2-xp0) in lane 2.
    real3 r = (lane == 0 ? s+make_real3((real) (xpPartner.x-xp.x), (real) (xpPartner.y-xp.y), (real) (xpPartner.z-xp.z))
                         : s+make_real3((real) (xp.x-xpPartner.x), (real) (xp.y-xpPartner.y), (real) (xp.z-xpPartner.z)));
    real d01sq = params.x*params.x;
    real d12sq = params.y*params.y;
    real dsq = (lane == 1 ? d12sq : d01sq);
    real3 dx = make_real3(0);
    bool last = false;
    for (int iteration = 0; !last; iteration++) {
        // f01 = r01+dx1-dx0 in lane 0, f02 = r02+dx2-dx0 in lane 2, f12 = f02-f01 in lane 1.
        real3 dxPartner = shuffleReal3(dx, partner);
        real3 f = (lane == 0 ? r+dxPartner-dx : r+dx-dxPartner);
        real3 f01 = shuffleReal3(f, 0);
        real3 f02 = shuffleReal3(f, 2);
        if (lane == 1)
            f = f02-f01;
        real g = dsq-dot(f, f);
        real g01 = quad_shuffle(g, 0), g02 = quad_shuffle(g, 2), g12 = quad_shuffle(g, 1);
        last = (iteration == 14 || (fabs(g01) < 1e-6f*d01sq && fabs(g02) < 1e-6f*d01sq && fabs(g12) < 1e-6f*d12sq));
        if (last) {
            xp.x += dx.x;
            xp.y += dx.y;
            xp.z += dx.z;
            dx = make_real3(0);
            xpPartner = shuffleMixed3(xp, partner);
            // m01 = b01+xp1-xp0 in lane 0, m02 = b02+xp2-xp0 in lane 2, m12 = m02-m01 in lane 1.
            mixed3 m = (lane == 0 ? make_mixed3(b.x+xpPartner.x-xp.x, b.y+xpPartner.y-xp.y, b.z+xpPartner.z-xp.z)
                                  : make_mixed3(b.x+xp.x-xpPartner.x, b.y+xp.y-xpPartner.y, b.z+xp.z-xpPartner.z));
            mixed3 m01 = shuffleMixed3(m, 0);
            mixed3 m02 = shuffleMixed3(m, 2);
            if (lane == 1)
                m = make_mixed3(m02.x-m01.x, m02.y-m01.y, m02.z-m01.z);
            real d = (lane == 1 ? params.y : params.x);
            g = (real) ((mixed) d*d - (m.x*m.x + m.y*m.y + m.z*m.z));
            g01 = quad_shuffle(g, 0);
            g02 = quad_shuffle(g, 2);
            g12 = quad_shuffle(g, 1);
        }

        // This lane's row of the linearized equations, then the solve in every lane.

        real k1 = (lane == 0 ? w0+w1 : (lane == 2 ? w0 : -w1));
        real k2 = (lane == 0 ? w0 : (lane == 2 ? w0+w2 : w2));
        real k3 = (lane == 0 ? -w1 : (lane == 2 ? w2 : w1+w2));
        real a1 = k1*dot(f, s01), a2 = k2*dot(f, s02), a3 = k3*dot(f, s12);
        real a11 = quad_shuffle(a1, 0), a12 = quad_shuffle(a2, 0), a13 = quad_shuffle(a3, 0);
        real a21 = quad_shuffle(a1, 2), a22 = quad_shuffle(a2, 2), a23 = quad_shuffle(a3, 2);
        real a31 = quad_shuffle(a1, 1), a32 = quad_shuffle(a2, 1), a33 = quad_shuffle(a3, 1);
        real c11 = a22*a33-a23*a32, c12 = a23*a31-a21*a33, c13 = a21*a32-a22*a31;
        real scale = 0.5f/(a11*c11 + a12*c12 + a13*c13);
        real l01 = (g01*c11 + g02*(a13*a32-a12*a33) + g12*(a12*a23-a13*a22))*scale;
        real l02 = (g01*c12 + g02*(a11*a33-a13*a31) + g12*(a13*a21-a11*a23))*scale;
        real l12 = (g01*c13 + g02*(a12*a31-a11*a32) + g12*(a11*a22-a12*a21))*scale;
        if (lane == 0)
            dx -= (s01*l01 + s02*l02)*w;
        else if (lane == 1)
            dx += (s01*l01 - s12*l12)*w;
        else
            dx += (s02*l02 + s12*l12)*w;
    }
    if (lane < 3) {
        xp.x += dx.x;
        xp.y += dx.y;
        xp.z += dx.z;
        delta = make_mixed4(xp.x, xp.y, xp.z, delta.w);
    }
}
#endif

/**
 * The same step with each quad of threads owning a unit: a SETTLE cluster, a SHAKE cluster, or four atoms in neither
 * (freeAtoms is padded with -1 to a multiple of four).  Each lane kicks, heat-baths and finishes its own atom, so that
 * work runs four wide.  For a SHAKE cluster (and a SETTLE cluster outside emulated mixed precision), every lane gathers
 * the four atoms and runs the same constraint code, then keeps its own atom's result; a SETTLE cluster in emulated
 * mixed precision spreads its constraint work over the lanes.  Each atom sees the same operations as in
 * integrateLangevinMiddleFused.
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
#ifdef EMULATED_MIXED_PRECISION
        // A SETTLE cluster's constraints run spread over the quad's lanes instead of four times over: see
        // applySettleToVelocitiesQuad.
        bool gatherCluster = isShake;
#else
        bool gatherCluster = (isSettle || isShake);
#endif
        mixed4 clusterPos[4], clusterVel[4];
        if (gatherCluster) {
            for (int i = 0; i < 4; i++) {
                clusterPos[i] = shuffleMixed4(pos, i);
                clusterVel[i] = shuffleMixed4(vel, i);
            }
            if (isSettle)
                applySettleToClusterVelocities(clusterPos, clusterVel);
            else
                applyShakeToClusterVelocitiesFused(atoms, shakeParams[unit-numSettle], tol, clusterPos, clusterVel);
            vel = laneElement(clusterVel, lane);
        }
#ifdef EMULATED_MIXED_PRECISION
        else if (isSettle)
            applySettleToVelocitiesQuad(lane, pos, vel);
#endif
        if (atom != -1 && vel.w != 0.0)
            bathAtom(atom, random, randomIndex, halfdt, vscale, noisescale, vel, delta, oldDelta);
        if (gatherCluster) {
            mixed4 clusterDelta[4];
            for (int i = 0; i < 4; i++)
                clusterDelta[i] = shuffleMixed4(delta, i);
            if (isSettle)
                applySettleToCluster(settleParams[unit], make_mixed3(clusterVel[0].w, clusterVel[1].w, clusterVel[2].w), clusterPos, clusterDelta);
            else
                applyShakeToClusterFused(atoms, shakeParams[unit-numSettle], tol, clusterPos, clusterDelta);
            delta = laneElement(clusterDelta, lane);
        }
#ifdef EMULATED_MIXED_PRECISION
        else if (isSettle)
            applySettleToPositionsQuad(lane, settleParams[unit], pos, vel, delta);
#endif
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
