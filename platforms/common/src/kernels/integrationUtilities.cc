/**
 * Generate random numbers
 */
KERNEL void generateRandomNumbers(int numValues, GLOBAL float4* RESTRICT random, GLOBAL uint4* RESTRICT seed) {
    int index = GLOBAL_ID;
    uint4 state = seed[index];
    unsigned int carry = 0;
    while (index < numValues) {
        float4 value;

        // Generate first two values.

        state.x = state.x * 69069 + 1;
        state.y ^= state.y << 13;
        state.y ^= state.y >> 17;
        state.y ^= state.y << 5;
        unsigned int k = (state.z >> 2) + (state.w >> 3) + (carry >> 2);
        unsigned int m = state.w + state.w + state.z + carry;
        state.z = state.w;
        state.w = m;
        carry = k >> 30;
        float x1 = (float)max(state.x + state.y + state.w, 0x00000001u) / (float)0xffffffff;
        state.x = state.x * 69069 + 1;
        state.y ^= state.y << 13;
        state.y ^= state.y >> 17;
        state.y ^= state.y << 5;
        x1 = SQRT(-2.0f * LOG(x1));
        k = (state.z >> 2) + (state.w >> 3) + (carry >> 2);
        m = state.w + state.w + state.z + carry;
        state.z = state.w;
        state.w = m;
        carry = k >> 30;
        float x2 = (float)(state.x + state.y + state.w) / (float)0xffffffff;
        value.x = x1 * COS(2.0f * 3.14159265f * x2);
        value.y = x1 * SIN(2.0f * 3.14159265f * x2);

        // Generate next two values.

        state.x = state.x * 69069 + 1;
        state.y ^= state.y << 13;
        state.y ^= state.y >> 17;
        state.y ^= state.y << 5;
        k = (state.z >> 2) + (state.w >> 3) + (carry >> 2);
        m = state.w + state.w + state.z + carry;
        state.z = state.w;
        state.w = m;
        carry = k >> 30;
        float x3 = (float)max(state.x + state.y + state.w, 0x00000001u) / (float)0xffffffff;
        state.x = state.x * 69069 + 1;
        state.y ^= state.y << 13;
        state.y ^= state.y >> 17;
        state.y ^= state.y << 5;
        x3 = SQRT(-2.0f * LOG(x3));
        k = (state.z >> 2) + (state.w >> 3) + (carry >> 2);
        m = state.w + state.w + state.z + carry;
        state.z = state.w;
        state.w = m;
        carry = k >> 30;
        float x4 = (float)(state.x + state.y + state.w) / (float)0xffffffff;
        value.z = x3 * COS(2.0f * 3.14159265f * x4);
        value.w = x3 * SIN(2.0f * 3.14159265f * x4);

        // Record the values.

        random[index] = value;
        index += GLOBAL_SIZE;
    }
    seed[GLOBAL_ID] = state;
}

/**
 * Load the position of a particle.
 */
inline DEVICE mixed4 loadPos(GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection, int index) {
#ifdef USE_MIXED_PRECISION
    real4 pos1 = posq[index];
    real4 pos2 = posqCorrection[index];
    return make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
#else
    return posq[index];
#endif
}

/**
 * Store the position of a particle.
 */
inline DEVICE void storePos(GLOBAL real4* RESTRICT posq, GLOBAL real4* RESTRICT posqCorrection, int index, mixed4 pos) {
#ifdef USE_MIXED_PRECISION
    posq[index] = make_real4((real) pos.x, (real) pos.y, (real) pos.z, (real) pos.w);
    posqCorrection[index] = make_real4(pos.x-(real) pos.x, pos.y-(real) pos.y, pos.z-(real) pos.z, 0);
#else
    posq[index] = pos;
#endif
}

/**
 * Metal emulates mixed precision with pairs of floats, which makes every mixed operation several times the cost of
 * a real one.  There the constraint kernels evaluate residuals in mixed precision but compute the corrections in
 * real precision, and iterate until the residuals meet the tolerance.
 */
#if defined(USE_METAL) && defined(USE_MIXED_PRECISION)
#define EMULATED_MIXED_PRECISION
#endif

/**
 * Load the positions of the atoms in a SHAKE cluster at the start of the step, and their displacements or velocities.
 * The peripheral atoms a cluster lacks get zeros.
 */
inline DEVICE void loadShakeCluster(int4 atoms, GLOBAL const real4* RESTRICT oldPos, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL const mixed4s* RESTRICT values, PRIVATE mixed4* pos, PRIVATE mixed4* value) {
    pos[0] = loadPos(oldPos, posqCorrection, atoms.x);
    value[0] = values[atoms.x];
    pos[1] = loadPos(oldPos, posqCorrection, atoms.y);
    value[1] = values[atoms.y];
    pos[2] = make_mixed4(0);
    value[2] = make_mixed4(0);
    if (atoms.z != -1) {
        pos[2] = loadPos(oldPos, posqCorrection, atoms.z);
        value[2] = values[atoms.z];
    }
    pos[3] = make_mixed4(0);
    value[3] = make_mixed4(0);
    if (atoms.w != -1) {
        pos[3] = loadPos(oldPos, posqCorrection, atoms.w);
        value[3] = values[atoms.w];
    }
}

/**
 * Store the displacements or velocities of the atoms in a SHAKE cluster.
 */
inline DEVICE void storeShakeCluster(int4 atoms, GLOBAL mixed4s* RESTRICT values, PRIVATE const mixed4* value) {
    values[atoms.x] = value[0];
    values[atoms.y] = value[1];
    if (atoms.z != -1)
        values[atoms.z] = value[2];
    if (atoms.w != -1)
        values[atoms.w] = value[3];
}

/**
 * Enforce constraints on one SHAKE cluster.  clusterPos holds the positions of its atoms at the start of the step, and
 * delta their displacements, which this updates.
 */
#ifndef SHAKE_MAX_ITER
#define SHAKE_MAX_ITER 15
#endif

DEVICE void applyShakeToCluster(int4 atoms, float4 params, mixed tol, PRIVATE const mixed4* clusterPos, PRIVATE mixed4* delta) {
    mixed4 pos = clusterPos[0], pos1 = clusterPos[1], pos2 = clusterPos[2], pos3 = clusterPos[3];
    mixed4 xpi = delta[0], xpj1 = delta[1], xpj2 = delta[2], xpj3 = delta[3];
    float invMassCentral = params.x;
    float avgMass = params.y;
    float d2 = params.z;
    float invMassPeripheral = params.w;

    // Precompute quantities.

    mixed3 rij1 = make_mixed3(pos.x-pos1.x, pos.y-pos1.y, pos.z-pos1.z);
    mixed3 rij2 = make_mixed3(pos.x-pos2.x, pos.y-pos2.y, pos.z-pos2.z);
    mixed3 rij3 = make_mixed3(pos.x-pos3.x, pos.y-pos3.y, pos.z-pos3.z);
    mixed rij1sq = rij1.x*rij1.x + rij1.y*rij1.y + rij1.z*rij1.z;
    mixed rij2sq = rij2.x*rij2.x + rij2.y*rij2.y + rij2.z*rij2.z;
    mixed rij3sq = rij3.x*rij3.x + rij3.y*rij3.y + rij3.z*rij3.z;
    mixed ld1 = d2-rij1sq;
    mixed ld2 = d2-rij2sq;
    mixed ld3 = d2-rij3sq;
#ifdef EMULATED_MIXED_PRECISION
    real3 r1 = make_real3((real) rij1.x, (real) rij1.y, (real) rij1.z);
    real3 r2 = make_real3((real) rij2.x, (real) rij2.y, (real) rij2.z);
    real3 r3 = make_real3((real) rij3.x, (real) rij3.y, (real) rij3.z);
    real maxResidual = d2*(real) tol;
#endif

    // Iterate until convergence.

    bool converged = false;
    int iteration = 0;
    while (iteration < SHAKE_MAX_ITER && !converged) {
        converged = true;
        mixed3 rpij = make_mixed3(xpi.x-xpj1.x, xpi.y-xpj1.y, xpi.z-xpj1.z);
        mixed rpsqij = rpij.x*rpij.x + rpij.y*rpij.y + rpij.z*rpij.z;
        mixed rrpr = rij1.x*rpij.x + rij1.y*rpij.y + rij1.z*rpij.z;
#ifdef EMULATED_MIXED_PRECISION
        real residual = (real) (ld1-2.0f*rrpr-rpsqij);
        if (fabs(residual) >= maxResidual) {
            real acor = residual*avgMass / ((real) rrpr+(real) rij1sq);
            real3 dr = r1*acor;
#else
        mixed diff = fabs(ld1-2.0f*rrpr-rpsqij) / (d2*tol);
        if (diff >= 1.0f) {
            mixed acor  = (ld1-2.0f*rrpr-rpsqij)*avgMass / (rrpr+rij1sq);
            mixed3 dr = rij1*acor;
#endif
            xpi.x += dr.x*invMassCentral;
            xpi.y += dr.y*invMassCentral;
            xpi.z += dr.z*invMassCentral;
            xpj1.x -= dr.x*invMassPeripheral;
            xpj1.y -= dr.y*invMassPeripheral;
            xpj1.z -= dr.z*invMassPeripheral;
            converged = false;
        }
        if (atoms.z != -1) {
            rpij = make_mixed3(xpi.x-xpj2.x, xpi.y-xpj2.y, xpi.z-xpj2.z);
            rpsqij = rpij.x*rpij.x + rpij.y*rpij.y + rpij.z*rpij.z;
            rrpr = rij2.x*rpij.x + rij2.y*rpij.y + rij2.z*rpij.z;
#ifdef EMULATED_MIXED_PRECISION
            residual = (real) (ld2-2.0f*rrpr-rpsqij);
            if (fabs(residual) >= maxResidual) {
                real acor = residual*avgMass / ((real) rrpr+(real) rij2sq);
                real3 dr = r2*acor;
#else
            diff = fabs(ld2-2.0f*rrpr-rpsqij) / (d2*tol);
            if (diff >= 1.0f) {
                mixed acor  = (ld2 - 2.0f*rrpr - rpsqij)*avgMass / (rrpr + rij2sq);
                mixed3 dr = rij2*acor;
#endif
                xpi.x += dr.x*invMassCentral;
                xpi.y += dr.y*invMassCentral;
                xpi.z += dr.z*invMassCentral;
                xpj2.x -= dr.x*invMassPeripheral;
                xpj2.y -= dr.y*invMassPeripheral;
                xpj2.z -= dr.z*invMassPeripheral;
                converged = false;
            }
        }
        if (atoms.w != -1) {
            rpij = make_mixed3(xpi.x-xpj3.x, xpi.y-xpj3.y, xpi.z-xpj3.z);
            rpsqij = rpij.x*rpij.x + rpij.y*rpij.y + rpij.z*rpij.z;
            rrpr = rij3.x*rpij.x + rij3.y*rpij.y + rij3.z*rpij.z;
#ifdef EMULATED_MIXED_PRECISION
            residual = (real) (ld3-2.0f*rrpr-rpsqij);
            if (fabs(residual) >= maxResidual) {
                real acor = residual*avgMass / ((real) rrpr+(real) rij3sq);
                real3 dr = r3*acor;
#else
            diff = fabs(ld3 - 2.0f*rrpr - rpsqij) / (d2*tol);
            if (diff >= 1.0f) {
                mixed acor  = (ld3-2.0f*rrpr-rpsqij)*avgMass / (rrpr+rij3sq);
                mixed3 dr = rij3*acor;
#endif
                xpi.x += dr.x*invMassCentral;
                xpi.y += dr.y*invMassCentral;
                xpi.z += dr.z*invMassCentral;
                xpj3.x -= dr.x*invMassPeripheral;
                xpj3.y -= dr.y*invMassPeripheral;
                xpj3.z -= dr.z*invMassPeripheral;
                converged = false;
            }
        }
        iteration++;
    }
    delta[0] = xpi;
    delta[1] = xpj1;
    delta[2] = xpj2;
    delta[3] = xpj3;
}

/**
 * Enforce constraints on SHAKE clusters
 */
KERNEL void applyShakeToPositions(int numClusters, mixed tol, GLOBAL const real4* RESTRICT oldPos,
        GLOBAL mixed4s* RESTRICT posDelta, GLOBAL const int4* RESTRICT clusterAtoms, GLOBAL const float4* RESTRICT clusterParams
#ifdef USE_MIXED_PRECISION
        , GLOBAL const real4* RESTRICT posqCorrection
#endif
        ) {
#ifndef USE_MIXED_PRECISION
        GLOBAL real4* posqCorrection = 0;
#endif
    int index = GLOBAL_ID;
    while (index < numClusters) {
        int4 atoms = clusterAtoms[index];
        mixed4 pos[4], xp[4];
        loadShakeCluster(atoms, oldPos, posqCorrection, posDelta, pos, xp);
        applyShakeToCluster(atoms, clusterParams[index], tol, pos, xp);
        storeShakeCluster(atoms, posDelta, xp);
        index += GLOBAL_SIZE;
    }
}

/**
 * Enforce velocity constraints on one SHAKE cluster.  clusterPos holds the positions of its atoms at the start of the
 * step, and vel their velocities, which this updates.
 */
DEVICE void applyShakeToClusterVelocities(int4 atoms, float4 params, mixed tol, PRIVATE const mixed4* clusterPos, PRIVATE mixed4* vel) {
    mixed4 pos = clusterPos[0], pos1 = clusterPos[1], pos2 = clusterPos[2], pos3 = clusterPos[3];
    mixed4 xpi = vel[0], xpj1 = vel[1], xpj2 = vel[2], xpj3 = vel[3];
    float invMassCentral = params.x;
    float avgMass = params.y;
    float invMassPeripheral = params.w;

    // Precompute quantities.

    mixed3 rij1 = make_mixed3(pos.x-pos1.x, pos.y-pos1.y, pos.z-pos1.z);
    mixed3 rij2 = make_mixed3(pos.x-pos2.x, pos.y-pos2.y, pos.z-pos2.z);
    mixed3 rij3 = make_mixed3(pos.x-pos3.x, pos.y-pos3.y, pos.z-pos3.z);
#ifdef EMULATED_MIXED_PRECISION
    real3 r1 = make_real3((real) rij1.x, (real) rij1.y, (real) rij1.z);
    real3 r2 = make_real3((real) rij2.x, (real) rij2.y, (real) rij2.z);
    real3 r3 = make_real3((real) rij3.x, (real) rij3.y, (real) rij3.z);
    real rij1sq = r1.x*r1.x + r1.y*r1.y + r1.z*r1.z;
    real rij2sq = r2.x*r2.x + r2.y*r2.y + r2.z*r2.z;
    real rij3sq = r3.x*r3.x + r3.y*r3.y + r3.z*r3.z;
#else
    mixed rij1sq = rij1.x*rij1.x + rij1.y*rij1.y + rij1.z*rij1.z;
    mixed rij2sq = rij2.x*rij2.x + rij2.y*rij2.y + rij2.z*rij2.z;
    mixed rij3sq = rij3.x*rij3.x + rij3.y*rij3.y + rij3.z*rij3.z;
#endif

    // Iterate until convergence.

    bool converged = false;
    int iteration = 0;
    while (iteration < SHAKE_MAX_ITER && !converged) {
        converged = true;
        mixed3 rpij = make_mixed3(xpi.x-xpj1.x, xpi.y-xpj1.y, xpi.z-xpj1.z);
        mixed rrpr = rpij.x*rij1.x + rpij.y*rij1.y + rpij.z*rij1.z;
#ifdef EMULATED_MIXED_PRECISION
        real delta = -2.0f*avgMass*(real) rrpr/rij1sq;
        real3 dr = r1*delta;
#else
        mixed delta = -2.0f*avgMass*rrpr/rij1sq;
        mixed3 dr = rij1*delta;
#endif
        xpi.x += dr.x*invMassCentral;
        xpi.y += dr.y*invMassCentral;
        xpi.z += dr.z*invMassCentral;
        xpj1.x -= dr.x*invMassPeripheral;
        xpj1.y -= dr.y*invMassPeripheral;
        xpj1.z -= dr.z*invMassPeripheral;
        if (fabs(delta) > tol)
            converged = false;
        if (atoms.z != -1) {
            rpij = make_mixed3(xpi.x-xpj2.x, xpi.y-xpj2.y, xpi.z-xpj2.z);
            rrpr = rpij.x*rij2.x + rpij.y*rij2.y + rpij.z*rij2.z;
#ifdef EMULATED_MIXED_PRECISION
            delta = -2.0f*avgMass*(real) rrpr/rij2sq;
            dr = r2*delta;
#else
            delta = -2.0f*avgMass*rrpr/rij2sq;
            dr = rij2*delta;
#endif
            xpi.x += dr.x*invMassCentral;
            xpi.y += dr.y*invMassCentral;
            xpi.z += dr.z*invMassCentral;
            xpj2.x -= dr.x*invMassPeripheral;
            xpj2.y -= dr.y*invMassPeripheral;
            xpj2.z -= dr.z*invMassPeripheral;
            if (fabs(delta) > tol)
                converged = false;
        }
        if (atoms.w != -1) {
            rpij = make_mixed3(xpi.x-xpj3.x, xpi.y-xpj3.y, xpi.z-xpj3.z);
            rrpr = rpij.x*rij3.x + rpij.y*rij3.y + rpij.z*rij3.z;
#ifdef EMULATED_MIXED_PRECISION
            delta = -2.0f*avgMass*(real) rrpr/rij3sq;
            dr = r3*delta;
#else
            delta = -2.0f*avgMass*rrpr/rij3sq;
            dr = rij3*delta;
#endif
            xpi.x += dr.x*invMassCentral;
            xpi.y += dr.y*invMassCentral;
            xpi.z += dr.z*invMassCentral;
            xpj3.x -= dr.x*invMassPeripheral;
            xpj3.y -= dr.y*invMassPeripheral;
            xpj3.z -= dr.z*invMassPeripheral;
            if (fabs(delta) > tol)
                converged = false;
        }
        iteration++;
    }
    vel[0] = xpi;
    vel[1] = xpj1;
    vel[2] = xpj2;
    vel[3] = xpj3;
}

/**
 * Enforce velocity constraints on SHAKE clusters
 */
KERNEL void applyShakeToVelocities(int numClusters, mixed tol, GLOBAL const real4* RESTRICT oldPos,
        GLOBAL mixed4s* RESTRICT posDelta, GLOBAL const int4* RESTRICT clusterAtoms, GLOBAL const float4* RESTRICT clusterParams
#ifdef USE_MIXED_PRECISION
        , GLOBAL const real4* RESTRICT posqCorrection
#endif
        ) {
#ifndef USE_MIXED_PRECISION
        GLOBAL real4* posqCorrection = 0;
#endif
    int index = GLOBAL_ID;
    while (index < numClusters) {
        int4 atoms = clusterAtoms[index];
        mixed4 pos[4], vel[4];
        loadShakeCluster(atoms, oldPos, posqCorrection, posDelta, pos, vel);
        applyShakeToClusterVelocities(atoms, clusterParams[index], tol, pos, vel);
        storeShakeCluster(atoms, posDelta, vel);
        index += GLOBAL_SIZE;
    }
}

/**
 * Enforce constraints on one SETTLE cluster.  pos holds the positions of its atoms at the start of the step, invMass
 * their inverse masses, and delta their displacements, which this updates.
 */
DEVICE void applySettleToCluster(float2 params, mixed3 invMass, PRIVATE const mixed4* pos, PRIVATE mixed4* delta) {
    mixed4 apos0 = pos[0], apos1 = pos[1], apos2 = pos[2];
    mixed4 xp0 = delta[0], xp1 = delta[1], xp2 = delta[2];
#ifdef EMULATED_MIXED_PRECISION
    // Solve the same equations as SETTLE by Newton's method: move the atoms along the old bonds, weighted by
    // inverse mass, until the bonds have the right lengths.  The analytic solution below needs much more
    // arithmetic in mixed precision.  The iterations run in real precision until the residuals are near its
    // resolution, then one last step from residuals in mixed precision brings them to about 1e-12 of the
    // squared bond lengths.

    real w0 = (real) invMass.x;
    real w1 = (real) invMass.y;
    real w2 = (real) invMass.z;
    mixed3 b01 = make_mixed3(apos1.x-apos0.x, apos1.y-apos0.y, apos1.z-apos0.z);
    mixed3 b02 = make_mixed3(apos2.x-apos0.x, apos2.y-apos0.y, apos2.z-apos0.z);
    real3 s01 = make_real3((real) b01.x, (real) b01.y, (real) b01.z);
    real3 s02 = make_real3((real) b02.x, (real) b02.y, (real) b02.z);
    real3 s12 = s02-s01;
    real3 r01 = s01+make_real3((real) (xp1.x-xp0.x), (real) (xp1.y-xp0.y), (real) (xp1.z-xp0.z));
    real3 r02 = s02+make_real3((real) (xp2.x-xp0.x), (real) (xp2.y-xp0.y), (real) (xp2.z-xp0.z));
    real d01sq = params.x*params.x;
    real d12sq = params.y*params.y;
    real3 dx0 = make_real3(0), dx1 = make_real3(0), dx2 = make_real3(0);
    bool last = false;
    for (int iteration = 0; !last; iteration++) {
        real3 f01 = r01+dx1-dx0;
        real3 f02 = r02+dx2-dx0;
        real3 f12 = f02-f01;
        real g01 = d01sq-dot(f01, f01);
        real g02 = d01sq-dot(f02, f02);
        real g12 = d12sq-dot(f12, f12);
        // After 15 iterations the loop stops without an error, even if the residuals have not converged.
        last = (iteration == 14 || (fabs(g01) < 1e-6f*d01sq && fabs(g02) < 1e-6f*d01sq && fabs(g12) < 1e-6f*d12sq));
        if (last) {
            // Move the corrections so far into the mixed precision displacements, so the last step is not
            // rounded to the resolution of the earlier ones.

            xp0.x += dx0.x;
            xp0.y += dx0.y;
            xp0.z += dx0.z;
            xp1.x += dx1.x;
            xp1.y += dx1.y;
            xp1.z += dx1.z;
            xp2.x += dx2.x;
            xp2.y += dx2.y;
            xp2.z += dx2.z;
            dx0 = make_real3(0);
            dx1 = make_real3(0);
            dx2 = make_real3(0);
            mixed3 m01 = make_mixed3(b01.x+xp1.x-xp0.x, b01.y+xp1.y-xp0.y, b01.z+xp1.z-xp0.z);
            mixed3 m02 = make_mixed3(b02.x+xp2.x-xp0.x, b02.y+xp2.y-xp0.y, b02.z+xp2.z-xp0.z);
            mixed3 m12 = make_mixed3(m02.x-m01.x, m02.y-m01.y, m02.z-m01.z);
            g01 = (real) ((mixed) params.x*params.x - (m01.x*m01.x + m01.y*m01.y + m01.z*m01.z));
            g02 = (real) ((mixed) params.x*params.x - (m02.x*m02.x + m02.y*m02.y + m02.z*m02.z));
            g12 = (real) ((mixed) params.y*params.y - (m12.x*m12.x + m12.y*m12.y + m12.z*m12.z));
        }

        // Solve the linearized equations for the multipliers of the three bonds.

        real a11 = (w0+w1)*dot(f01, s01), a12 = w0*dot(f01, s02), a13 = -w1*dot(f01, s12);
        real a21 = w0*dot(f02, s01), a22 = (w0+w2)*dot(f02, s02), a23 = w2*dot(f02, s12);
        real a31 = -w1*dot(f12, s01), a32 = w2*dot(f12, s02), a33 = (w1+w2)*dot(f12, s12);
        real c11 = a22*a33-a23*a32, c12 = a23*a31-a21*a33, c13 = a21*a32-a22*a31;
        real scale = 0.5f/(a11*c11 + a12*c12 + a13*c13);
        real l01 = (g01*c11 + g02*(a13*a32-a12*a33) + g12*(a12*a23-a13*a22))*scale;
        real l02 = (g01*c12 + g02*(a11*a33-a13*a31) + g12*(a13*a21-a11*a23))*scale;
        real l12 = (g01*c13 + g02*(a12*a31-a11*a32) + g12*(a11*a22-a12*a21))*scale;
        dx0 -= (s01*l01 + s02*l02)*w0;
        dx1 += (s01*l01 - s12*l12)*w1;
        dx2 += (s02*l02 + s12*l12)*w2;
    }
    xp0.x += dx0.x;
    xp0.y += dx0.y;
    xp0.z += dx0.z;
    xp1.x += dx1.x;
    xp1.y += dx1.y;
    xp1.z += dx1.z;
    xp2.x += dx2.x;
    xp2.y += dx2.y;
    xp2.z += dx2.z;
#else
    mixed m0 = 1/invMass.x;
    mixed m1 = 1/invMass.y;
    mixed m2 = 1/invMass.z;

    // Apply the SETTLE algorithm.

    mixed xb0 = apos1.x-apos0.x;
    mixed yb0 = apos1.y-apos0.y;
    mixed zb0 = apos1.z-apos0.z;
    mixed xc0 = apos2.x-apos0.x;
    mixed yc0 = apos2.y-apos0.y;
    mixed zc0 = apos2.z-apos0.z;

    mixed invTotalMass = 1/(m0+m1+m2);
    mixed xcom = (xp0.x*m0 + (xb0+xp1.x)*m1 + (xc0+xp2.x)*m2) * invTotalMass;
    mixed ycom = (xp0.y*m0 + (yb0+xp1.y)*m1 + (yc0+xp2.y)*m2) * invTotalMass;
    mixed zcom = (xp0.z*m0 + (zb0+xp1.z)*m1 + (zc0+xp2.z)*m2) * invTotalMass;

    mixed xa1 = xp0.x - xcom;
    mixed ya1 = xp0.y - ycom;
    mixed za1 = xp0.z - zcom;
    mixed xb1 = xb0 + xp1.x - xcom;
    mixed yb1 = yb0 + xp1.y - ycom;
    mixed zb1 = zb0 + xp1.z - zcom;
    mixed xc1 = xc0 + xp2.x - xcom;
    mixed yc1 = yc0 + xp2.y - ycom;
    mixed zc1 = zc0 + xp2.z - zcom;

    mixed xaksZd = yb0*zc0 - zb0*yc0;
    mixed yaksZd = zb0*xc0 - xb0*zc0;
    mixed zaksZd = xb0*yc0 - yb0*xc0;
    mixed xaksXd = ya1*zaksZd - za1*yaksZd;
    mixed yaksXd = za1*xaksZd - xa1*zaksZd;
    mixed zaksXd = xa1*yaksZd - ya1*xaksZd;
    mixed xaksYd = yaksZd*zaksXd - zaksZd*yaksXd;
    mixed yaksYd = zaksZd*xaksXd - xaksZd*zaksXd;
    mixed zaksYd = xaksZd*yaksXd - yaksZd*xaksXd;

    mixed axlng = sqrt(xaksXd*xaksXd + yaksXd*yaksXd + zaksXd*zaksXd);
    mixed aylng = sqrt(xaksYd*xaksYd + yaksYd*yaksYd + zaksYd*zaksYd);
    mixed azlng = sqrt(xaksZd*xaksZd + yaksZd*yaksZd + zaksZd*zaksZd);
    mixed trns11 = xaksXd / axlng;
    mixed trns21 = yaksXd / axlng;
    mixed trns31 = zaksXd / axlng;
    mixed trns12 = xaksYd / aylng;
    mixed trns22 = yaksYd / aylng;
    mixed trns32 = zaksYd / aylng;
    mixed trns13 = xaksZd / azlng;
    mixed trns23 = yaksZd / azlng;
    mixed trns33 = zaksZd / azlng;

    mixed xb0d = trns11*xb0 + trns21*yb0 + trns31*zb0;
    mixed yb0d = trns12*xb0 + trns22*yb0 + trns32*zb0;
    mixed xc0d = trns11*xc0 + trns21*yc0 + trns31*zc0;
    mixed yc0d = trns12*xc0 + trns22*yc0 + trns32*zc0;
    mixed za1d = trns13*xa1 + trns23*ya1 + trns33*za1;
    mixed xb1d = trns11*xb1 + trns21*yb1 + trns31*zb1;
    mixed yb1d = trns12*xb1 + trns22*yb1 + trns32*zb1;
    mixed zb1d = trns13*xb1 + trns23*yb1 + trns33*zb1;
    mixed xc1d = trns11*xc1 + trns21*yc1 + trns31*zc1;
    mixed yc1d = trns12*xc1 + trns22*yc1 + trns32*zc1;
    mixed zc1d = trns13*xc1 + trns23*yc1 + trns33*zc1;

    //                                        --- Step2  A2' ---

    float rc = 0.5f*params.y;
    mixed rb = sqrt(params.x*params.x-rc*rc);
    mixed ra = rb*(m1+m2)*invTotalMass;
    rb -= ra;
    mixed sinphi = za1d/ra;
    mixed cosphi = sqrt(1-sinphi*sinphi);
    mixed sinpsi = (zb1d-zc1d) / (2*rc*cosphi);
    mixed cospsi = sqrt(1-sinpsi*sinpsi);

    mixed ya2d =   ra*cosphi;
    mixed xb2d = - rc*cospsi;
    mixed yb2d = - rb*cosphi - rc*sinpsi*sinphi;
    mixed yc2d = - rb*cosphi + rc*sinpsi*sinphi;
    mixed xb2d2 = xb2d*xb2d;
    mixed hh2 = 4.0f*xb2d2 + (yb2d-yc2d)*(yb2d-yc2d) + (zb1d-zc1d)*(zb1d-zc1d);
    mixed deltx = 2.0f*xb2d + sqrt(4.0f*xb2d2 - hh2 + params.y*params.y);
    xb2d -= deltx*0.5f;

    //                                        --- Step3  al,be,ga ---

    mixed alpha = (xb2d*(xb0d-xc0d) + yb0d*yb2d + yc0d*yc2d);
    mixed beta = (xb2d*(yc0d-yb0d) + xb0d*yb2d + xc0d*yc2d);
    mixed gamma = xb0d*yb1d - xb1d*yb0d + xc0d*yc1d - xc1d*yc0d;

    mixed al2be2 = alpha*alpha + beta*beta;
    mixed sintheta = (alpha*gamma - beta*sqrt(al2be2 - gamma*gamma)) / al2be2;

    //                                        --- Step4  A3' ---

    mixed costheta = sqrt(1-sintheta*sintheta);
    mixed xa3d = - ya2d*sintheta;
    mixed ya3d =   ya2d*costheta;
    mixed za3d = za1d;
    mixed xb3d =   xb2d*costheta - yb2d*sintheta;
    mixed yb3d =   xb2d*sintheta + yb2d*costheta;
    mixed zb3d = zb1d;
    mixed xc3d = - xb2d*costheta - yc2d*sintheta;
    mixed yc3d = - xb2d*sintheta + yc2d*costheta;
    mixed zc3d = zc1d;

    //                                        --- Step5  A3 ---

    mixed xa3 = trns11*xa3d + trns12*ya3d + trns13*za3d;
    mixed ya3 = trns21*xa3d + trns22*ya3d + trns23*za3d;
    mixed za3 = trns31*xa3d + trns32*ya3d + trns33*za3d;
    mixed xb3 = trns11*xb3d + trns12*yb3d + trns13*zb3d;
    mixed yb3 = trns21*xb3d + trns22*yb3d + trns23*zb3d;
    mixed zb3 = trns31*xb3d + trns32*yb3d + trns33*zb3d;
    mixed xc3 = trns11*xc3d + trns12*yc3d + trns13*zc3d;
    mixed yc3 = trns21*xc3d + trns22*yc3d + trns23*zc3d;
    mixed zc3 = trns31*xc3d + trns32*yc3d + trns33*zc3d;

    xp0.x = xcom + xa3;
    xp0.y = ycom + ya3;
    xp0.z = zcom + za3;
    xp1.x = xcom + xb3 - xb0;
    xp1.y = ycom + yb3 - yb0;
    xp1.z = zcom + zb3 - zb0;
    xp2.x = xcom + xc3 - xc0;
    xp2.y = ycom + yc3 - yc0;
    xp2.z = zcom + zc3 - zc0;
#endif
    delta[0] = xp0;
    delta[1] = xp1;
    delta[2] = xp2;
}

/**
 * Enforce constraints on SETTLE clusters
 */
KERNEL void applySettleToPositions(int numClusters, mixed tol, GLOBAL const real4* RESTRICT oldPos,
        GLOBAL mixed4s* RESTRICT posDelta, GLOBAL const mixed4s* RESTRICT velm, GLOBAL const int4* RESTRICT clusterAtoms,
        GLOBAL const float2* RESTRICT clusterParams
#ifdef USE_MIXED_PRECISION
        , GLOBAL const real4* RESTRICT posqCorrection
#endif
        ) {
#ifndef USE_MIXED_PRECISION
        GLOBAL real4* posqCorrection = 0;
#endif
    int index = GLOBAL_ID;
    while (index < numClusters) {
        int4 atoms = clusterAtoms[index];
        mixed4 pos[3], xp[3];
        pos[0] = loadPos(oldPos, posqCorrection, atoms.x);
        xp[0] = posDelta[atoms.x];
        pos[1] = loadPos(oldPos, posqCorrection, atoms.y);
        xp[1] = posDelta[atoms.y];
        pos[2] = loadPos(oldPos, posqCorrection, atoms.z);
        xp[2] = posDelta[atoms.z];
        applySettleToCluster(clusterParams[index], make_mixed3(velm[atoms.x].w, velm[atoms.y].w, velm[atoms.z].w), pos, xp);
        posDelta[atoms.x] = xp[0];
        posDelta[atoms.y] = xp[1];
        posDelta[atoms.z] = xp[2];
        index += GLOBAL_SIZE;
    }
}

/**
 * Enforce velocity constraints on one SETTLE cluster.  pos holds the positions of its atoms at the start of the step,
 * and vel their velocities, which this updates.
 */
DEVICE void applySettleToClusterVelocities(PRIVATE const mixed4* pos, PRIVATE mixed4* vel) {
    mixed4 apos0 = pos[0], apos1 = pos[1], apos2 = pos[2];
    mixed4 v0 = vel[0], v1 = vel[1], v2 = vel[2];
#ifdef EMULATED_MIXED_PRECISION

    // Compute intermediate quantities: the atom masses, the bond directions, and the angle cosines and
    // sines.  They only set the size of the correction, which only needs to be as accurate as real.

    real mA = 1/(real) v0.w;
    real mB = 1/(real) v1.w;
    real mC = 1/(real) v2.w;
    mixed3 bAB = make_mixed3(apos1.x-apos0.x, apos1.y-apos0.y, apos1.z-apos0.z);
    mixed3 bBC = make_mixed3(apos2.x-apos1.x, apos2.y-apos1.y, apos2.z-apos1.z);
    mixed3 bCA = make_mixed3(apos0.x-apos2.x, apos0.y-apos2.y, apos0.z-apos2.z);
    real3 eAB = make_real3((real) bAB.x, (real) bAB.y, (real) bAB.z);
    real3 eBC = make_real3((real) bBC.x, (real) bBC.y, (real) bBC.z);
    real3 eCA = make_real3((real) bCA.x, (real) bCA.y, (real) bCA.z);
    real invLengthAB = RSQRT(eAB.x*eAB.x + eAB.y*eAB.y + eAB.z*eAB.z);
    real invLengthBC = RSQRT(eBC.x*eBC.x + eBC.y*eBC.y + eBC.z*eBC.z);
    real invLengthCA = RSQRT(eCA.x*eCA.x + eCA.y*eCA.y + eCA.z*eCA.z);
    eAB *= invLengthAB;
    eBC *= invLengthBC;
    eCA *= invLengthCA;
    real cA = -(eAB.x*eCA.x + eAB.y*eCA.y + eAB.z*eCA.z);
    real cB = -(eAB.x*eBC.x + eAB.y*eBC.y + eAB.z*eBC.z);
    real cC = -(eBC.x*eCA.x + eBC.y*eCA.y + eBC.z*eCA.z);
    real s2A = 1-cA*cA;
    real s2B = 1-cB*cB;
    real s2C = 1-cC*cC;
    
    // Solve the equations.  These are different from those in the SETTLE paper (JCC 13(8), pp. 952-962, 1992), because
    // in going from equations B1 to B2, they make the assumption that mB=mC (but don't bother to mention they're
    // making that assumption).  We allow all three atoms to have different masses.
    
    real mABCinv = 1/(mA*mB*mC);
    real denom = (((s2A*mB+s2B*mA)*mC+(s2A*mB*mB+2*(cA*cB*cC+1)*mA*mB+s2B*mA*mA))*mC+s2C*mA*mB*(mA+mB))*mABCinv;
    real invMass0 = (real) v0.w;
    real invMass1 = (real) v1.w;
    real invMass2 = (real) v2.w;

    // The relative velocities along the bonds are measured in mixed precision.  Solving in real leaves an error
    // of about 1e-6 of them, so a second pass measures what is left and removes it.

    for (int pass = 0; pass < 2; pass++) {
        real vAB = (real) ((v1.x-v0.x)*bAB.x + (v1.y-v0.y)*bAB.y + (v1.z-v0.z)*bAB.z)*invLengthAB;
        real vBC = (real) ((v2.x-v1.x)*bBC.x + (v2.y-v1.y)*bBC.y + (v2.z-v1.z)*bBC.z)*invLengthBC;
        real vCA = (real) ((v0.x-v2.x)*bCA.x + (v0.y-v2.y)*bCA.y + (v0.z-v2.z)*bCA.z)*invLengthCA;
        real tab = ((cB*cC*mA-cA*mB-cA*mC)*vCA + (cA*cC*mB-cB*mC-cB*mA)*vBC + (s2C*mA*mA*mB*mB*mABCinv+(mA+mB+mC))*vAB)/denom;
        real tbc = ((cA*cB*mC-cC*mB-cC*mA)*vCA + (s2A*mB*mB*mC*mC*mABCinv+(mA+mB+mC))*vBC + (cA*cC*mB-cB*mA-cB*mC)*vAB)/denom;
        real tca = ((s2B*mA*mA*mC*mC*mABCinv+(mA+mB+mC))*vCA + (cA*cB*mC-cC*mB-cC*mA)*vBC + (cB*cC*mA-cA*mB-cA*mC)*vAB)/denom;
        v0.x += (tab*eAB.x - tca*eCA.x)*invMass0;
        v0.y += (tab*eAB.y - tca*eCA.y)*invMass0;
        v0.z += (tab*eAB.z - tca*eCA.z)*invMass0;
        v1.x += (tbc*eBC.x - tab*eAB.x)*invMass1;
        v1.y += (tbc*eBC.y - tab*eAB.y)*invMass1;
        v1.z += (tbc*eBC.z - tab*eAB.z)*invMass1;
        v2.x += (tca*eCA.x - tbc*eBC.x)*invMass2;
        v2.y += (tca*eCA.y - tbc*eBC.y)*invMass2;
        v2.z += (tca*eCA.z - tbc*eBC.z)*invMass2;
    }
#else
    
    // Compute intermediate quantities: the atom masses, the bond directions, the relative velocities,
    // and the angle cosines and sines.
    
    mixed mA = 1/v0.w;
    mixed mB = 1/v1.w;
    mixed mC = 1/v2.w;
    mixed3 eAB = make_mixed3(apos1.x-apos0.x, apos1.y-apos0.y, apos1.z-apos0.z);
    mixed3 eBC = make_mixed3(apos2.x-apos1.x, apos2.y-apos1.y, apos2.z-apos1.z);
    mixed3 eCA = make_mixed3(apos0.x-apos2.x, apos0.y-apos2.y, apos0.z-apos2.z);
    eAB *= RSQRT(eAB.x*eAB.x + eAB.y*eAB.y + eAB.z*eAB.z);
    eBC *= RSQRT(eBC.x*eBC.x + eBC.y*eBC.y + eBC.z*eBC.z);
    eCA *= RSQRT(eCA.x*eCA.x + eCA.y*eCA.y + eCA.z*eCA.z);
    mixed vAB = (v1.x-v0.x)*eAB.x + (v1.y-v0.y)*eAB.y + (v1.z-v0.z)*eAB.z;
    mixed vBC = (v2.x-v1.x)*eBC.x + (v2.y-v1.y)*eBC.y + (v2.z-v1.z)*eBC.z;
    mixed vCA = (v0.x-v2.x)*eCA.x + (v0.y-v2.y)*eCA.y + (v0.z-v2.z)*eCA.z;
    mixed cA = -(eAB.x*eCA.x + eAB.y*eCA.y + eAB.z*eCA.z);
    mixed cB = -(eAB.x*eBC.x + eAB.y*eBC.y + eAB.z*eBC.z);
    mixed cC = -(eBC.x*eCA.x + eBC.y*eCA.y + eBC.z*eCA.z);
    mixed s2A = 1-cA*cA;
    mixed s2B = 1-cB*cB;
    mixed s2C = 1-cC*cC;
    
    // Solve the equations.  These are different from those in the SETTLE paper (JCC 13(8), pp. 952-962, 1992), because
    // in going from equations B1 to B2, they make the assumption that mB=mC (but don't bother to mention they're
    // making that assumption).  We allow all three atoms to have different masses.
    
    mixed mABCinv = 1/(mA*mB*mC);
    mixed denom = (((s2A*mB+s2B*mA)*mC+(s2A*mB*mB+2*(cA*cB*cC+1)*mA*mB+s2B*mA*mA))*mC+s2C*mA*mB*(mA+mB))*mABCinv;
    mixed tab = ((cB*cC*mA-cA*mB-cA*mC)*vCA + (cA*cC*mB-cB*mC-cB*mA)*vBC + (s2C*mA*mA*mB*mB*mABCinv+(mA+mB+mC))*vAB)/denom;
    mixed tbc = ((cA*cB*mC-cC*mB-cC*mA)*vCA + (s2A*mB*mB*mC*mC*mABCinv+(mA+mB+mC))*vBC + (cA*cC*mB-cB*mA-cB*mC)*vAB)/denom;
    mixed tca = ((s2B*mA*mA*mC*mC*mABCinv+(mA+mB+mC))*vCA + (cA*cB*mC-cC*mB-cC*mA)*vBC + (cB*cC*mA-cA*mB-cA*mC)*vAB)/denom;
    v0.x += (tab*eAB.x - tca*eCA.x)*v0.w;
    v0.y += (tab*eAB.y - tca*eCA.y)*v0.w;
    v0.z += (tab*eAB.z - tca*eCA.z)*v0.w;
    v1.x += (tbc*eBC.x - tab*eAB.x)*v1.w;
    v1.y += (tbc*eBC.y - tab*eAB.y)*v1.w;
    v1.z += (tbc*eBC.z - tab*eAB.z)*v1.w;
    v2.x += (tca*eCA.x - tbc*eBC.x)*v2.w;
    v2.y += (tca*eCA.y - tbc*eBC.y)*v2.w;
    v2.z += (tca*eCA.z - tbc*eBC.z)*v2.w;
#endif
    vel[0] = v0;
    vel[1] = v1;
    vel[2] = v2;
}

/**
 * Enforce velocity constraints on SETTLE clusters
 */
KERNEL void applySettleToVelocities(int numClusters, mixed tol, GLOBAL const real4* RESTRICT oldPos,
        GLOBAL mixed4s* RESTRICT posDelta, GLOBAL mixed4s* RESTRICT velm, GLOBAL const int4* RESTRICT clusterAtoms,
        GLOBAL const float2* RESTRICT clusterParams
#ifdef USE_MIXED_PRECISION
        , GLOBAL const real4* RESTRICT posqCorrection
#endif
        ) {
#ifndef USE_MIXED_PRECISION
        GLOBAL real4* posqCorrection = 0;
#endif
    for (int index = GLOBAL_ID; index < numClusters; index += GLOBAL_SIZE) {
        int4 atoms = clusterAtoms[index];
        mixed4 pos[3], vel[3];
        pos[0] = loadPos(oldPos, posqCorrection, atoms.x);
        pos[1] = loadPos(oldPos, posqCorrection, atoms.y);
        pos[2] = loadPos(oldPos, posqCorrection, atoms.z);
        vel[0] = velm[atoms.x];
        vel[1] = velm[atoms.y];
        vel[2] = velm[atoms.z];
        applySettleToClusterVelocities(pos, vel);
        velm[atoms.x] = vel[0];
        velm[atoms.y] = vel[1];
        velm[atoms.z] = vel[2];
    }
}

/**
 * Compute the direction each CCMA constraint is pointing in.  This is called once at the beginning of constraint evaluation.
 */
DEVICE void computeCCMAConstraintDirections(GLOBAL const int2* RESTRICT constraintAtoms, GLOBAL mixed4* RESTRICT constraintDistance,
        GLOBAL const real4* RESTRICT atomPositions
#ifdef USE_MIXED_PRECISION
        , GLOBAL const real4* RESTRICT posqCorrection
#endif
        ) {
#ifndef USE_MIXED_PRECISION
        GLOBAL real4* posqCorrection = 0;
#endif
    for (int index = GLOBAL_ID; index < NUM_CCMA_CONSTRAINTS; index += GLOBAL_SIZE) {
        // Compute the direction for this constraint.

        int2 atoms = constraintAtoms[index];
        mixed4 dir = constraintDistance[index];
        mixed4 oldPos1 = loadPos(atomPositions, posqCorrection, atoms.x);
        mixed4 oldPos2 = loadPos(atomPositions, posqCorrection, atoms.y);
        dir.x = oldPos1.x-oldPos2.x;
        dir.y = oldPos1.y-oldPos2.y;
        dir.z = oldPos1.z-oldPos2.z;
        constraintDistance[index] = dir;
    }
}

KERNEL void computeCCMAConstraintDirectionsKernel(GLOBAL const int2* RESTRICT constraintAtoms, GLOBAL mixed4* RESTRICT constraintDistance,
        GLOBAL const real4* RESTRICT atomPositions, GLOBAL int* RESTRICT converged
#ifdef USE_MIXED_PRECISION
        , GLOBAL const real4* RESTRICT posqCorrection
#endif
        ) {
#ifdef USE_MIXED_PRECISION
    computeCCMAConstraintDirections(constraintAtoms, constraintDistance, atomPositions, posqCorrection);
#else
    computeCCMAConstraintDirections(constraintAtoms, constraintDistance, atomPositions);
#endif
    if (GLOBAL_ID == 0) {
        converged[0] = 1;
        converged[1] = 0;
    }
}

/**
 * Compute the force applied by each CCMA position constraint.
 */
DEVICE void computeCCMAPositionConstraintForce(GLOBAL const int2* RESTRICT constraintAtoms, GLOBAL const mixed4* RESTRICT constraintDistance,
        GLOBAL const mixed4s* RESTRICT atomPositions, GLOBAL const mixed* RESTRICT reducedMass, GLOBAL mixed* RESTRICT delta1,
        mixed tol, int iteration, LOCAL_ARG int* groupConverged) {
    if (LOCAL_ID == 0)
        *groupConverged = 1;
    SYNC_THREADS;
    mixed lowerTol = 1-2*tol+tol*tol;
    mixed upperTol = 1+2*tol+tol*tol;
    bool threadConverged = true;
    for (int index = GLOBAL_ID; index < NUM_CCMA_CONSTRAINTS; index += GLOBAL_SIZE) {
        // Compute the force due to this constraint.

        int2 atoms = constraintAtoms[index];
        mixed4 dir = constraintDistance[index];
        mixed4 rp_ij = atomPositions[atoms.x]-atomPositions[atoms.y];
        rp_ij.x += dir.x;
        rp_ij.y += dir.y;
        rp_ij.z += dir.z;
        mixed rrpr = rp_ij.x*dir.x + rp_ij.y*dir.y + rp_ij.z*dir.z;
        mixed d_ij2 = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
        mixed rp2 = rp_ij.x*rp_ij.x + rp_ij.y*rp_ij.y + rp_ij.z*rp_ij.z;
        mixed dist2 = dir.w*dir.w;
        mixed diff = dist2 - rp2;
        delta1[index] = (rrpr > d_ij2*1e-6f ? reducedMass[index]*diff/rrpr : (mixed) 0);
        threadConverged &= (rp2 > lowerTol*dist2 && rp2 < upperTol*dist2);
    }
    if (*groupConverged && !threadConverged)
        *groupConverged = 0;
}

KERNEL void computeCCMAPositionConstraintForceKernel(GLOBAL const int2* RESTRICT constraintAtoms, GLOBAL const mixed4* RESTRICT constraintDistance,
        GLOBAL const mixed4s* RESTRICT atomPositions, GLOBAL const mixed* RESTRICT reducedMass, GLOBAL mixed* RESTRICT delta1,
        GLOBAL int* RESTRICT converged, GLOBAL int* RESTRICT hostConvergedFlag, mixed tol, int iteration) {
    LOCAL int groupConverged;
    if (converged[1-iteration%2]) {
        if (GLOBAL_ID == 0) {
            converged[iteration%2] = 1;
            hostConvergedFlag[0] = 1;
        }
        return; // The constraint iteration has already converged.
    }
    computeCCMAPositionConstraintForce(constraintAtoms, constraintDistance, atomPositions, reducedMass,
            delta1, tol, iteration, &groupConverged);
    SYNC_THREADS;
    if (LOCAL_ID == 0 && !groupConverged)
        converged[iteration%2] = 0;
}

/**
 * Compute the force applied by each CCMA velocity constraint.
 */
DEVICE void computeCCMAVelocityConstraintForce(GLOBAL const int2* RESTRICT constraintAtoms, GLOBAL const mixed4* RESTRICT constraintDistance,
        GLOBAL const mixed4s* RESTRICT atomPositions, GLOBAL const mixed* RESTRICT reducedMass, GLOBAL mixed* RESTRICT delta1,
        mixed tol, int iteration, LOCAL_ARG int* groupConverged) {
    if (LOCAL_ID == 0)
        *groupConverged = 1;
    SYNC_THREADS;
    bool threadConverged = true;
    for (int index = GLOBAL_ID; index < NUM_CCMA_CONSTRAINTS; index += GLOBAL_SIZE) {
        // Compute the force due to this constraint.

        int2 atoms = constraintAtoms[index];
        mixed4 dir = constraintDistance[index];
        mixed4 rp_ij = atomPositions[atoms.x]-atomPositions[atoms.y];
        mixed rrpr = rp_ij.x*dir.x + rp_ij.y*dir.y + rp_ij.z*dir.z;
        mixed d_ij2 = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
        delta1[index] = -2*reducedMass[index]*rrpr/d_ij2;
        threadConverged &= (fabs(delta1[index]) <= tol);
    }
    if (*groupConverged && !threadConverged)
        *groupConverged = 0;
}

KERNEL void computeCCMAVelocityConstraintForceKernel(GLOBAL const int2* RESTRICT constraintAtoms, GLOBAL const mixed4* RESTRICT constraintDistance,
        GLOBAL const mixed4s* RESTRICT atomPositions, GLOBAL const mixed* RESTRICT reducedMass, GLOBAL mixed* RESTRICT delta1,
        GLOBAL int* RESTRICT converged, GLOBAL int* RESTRICT hostConvergedFlag, mixed tol, int iteration) {
    LOCAL int groupConverged;
    if (converged[1-iteration%2]) {
        if (GROUP_ID == 0 && LOCAL_ID == 0) {
            converged[iteration%2] = 1;
            hostConvergedFlag[0] = 1;
        }
        return; // The constraint iteration has already converged.
    }
    computeCCMAVelocityConstraintForce(constraintAtoms, constraintDistance, atomPositions, reducedMass,
            delta1, tol, iteration, &groupConverged);
    if (LOCAL_ID == 0 && !groupConverged)
        converged[iteration%2] = 0;
}

/**
 * Multiply the vector of CCMA constraint forces by the constraint matrix.
 */
DEVICE void multiplyByCCMAConstraintMatrix(GLOBAL const mixed* RESTRICT delta1, GLOBAL mixed* RESTRICT delta2, GLOBAL const int* RESTRICT constraintMatrixColumn,
        GLOBAL const mixed* RESTRICT constraintMatrixValue, int iteration) {
    // Multiply by the inverse constraint matrix.

    for (int index = GLOBAL_ID; index < NUM_CCMA_CONSTRAINTS; index += GLOBAL_SIZE) {
        mixed sum = 0;
        for (int i = 0; ; i++) {
            int element = index+i*NUM_CCMA_CONSTRAINTS;
            int column = constraintMatrixColumn[element];
            if (column >= NUM_CCMA_CONSTRAINTS)
                break;
            sum += delta1[column]*constraintMatrixValue[element];
        }
        delta2[index] = sum;
    }
}

KERNEL void multiplyByCCMAConstraintMatrixKernel(GLOBAL const mixed* RESTRICT delta1, GLOBAL mixed* RESTRICT delta2, GLOBAL const int* RESTRICT constraintMatrixColumn,
        GLOBAL const mixed* RESTRICT constraintMatrixValue, GLOBAL const int* RESTRICT converged, int iteration) {
    if (converged[iteration%2])
        return; // The constraint iteration has already converged.
    multiplyByCCMAConstraintMatrix(delta1, delta2, constraintMatrixColumn, constraintMatrixValue, iteration);
}

/**
 * Update the atom positions based on CCMA constraint forces.
 */
DEVICE void updateCCMAAtomPositions(GLOBAL const int* RESTRICT atoms, GLOBAL const int* RESTRICT numAtomConstraints, GLOBAL const int* RESTRICT atomConstraints,
        GLOBAL const mixed4* RESTRICT constraintDistance, GLOBAL mixed4s* RESTRICT atomPositions, GLOBAL const mixed4s* RESTRICT velm,
        GLOBAL const mixed* RESTRICT delta1, GLOBAL const mixed* RESTRICT delta2, int iteration) {
    mixed damping = (iteration < 2 ? 0.5f : 1.0f);
    for (int i = GLOBAL_ID; i < NUM_CCMA_ATOMS; i += GLOBAL_SIZE) {
        // Compute the new position of this atom.

        int index = atoms[i];
        mixed4 atomPos = atomPositions[index];
        mixed invMass = velm[index].w;
        int num = numAtomConstraints[index];
        for (int i = 0; i < num; i++) {
            int constraint = atomConstraints[index+i*NUM_ATOMS];
            bool forward = (constraint > 0);
            constraint = (forward ? constraint-1 : -constraint-1);
            mixed constraintForce = damping*invMass*delta2[constraint];
            constraintForce = (forward ? constraintForce : -constraintForce);
            mixed4 dir = constraintDistance[constraint];
            atomPos.x += constraintForce*dir.x;
            atomPos.y += constraintForce*dir.y;
            atomPos.z += constraintForce*dir.z;
        }
        atomPositions[index] = atomPos;
    }
}

KERNEL void updateCCMAAtomPositionsKernel(GLOBAL const int* RESTRICT atoms, GLOBAL const int* RESTRICT numAtomConstraints, GLOBAL const int* RESTRICT atomConstraints,
        GLOBAL const mixed4* RESTRICT constraintDistance, GLOBAL mixed4s* RESTRICT atomPositions, GLOBAL const mixed4s* RESTRICT velm,
        GLOBAL const mixed* RESTRICT delta1, GLOBAL const mixed* RESTRICT delta2, GLOBAL int* RESTRICT converged, int iteration) {
    if (GROUP_ID == 0 && LOCAL_ID == 0)
        converged[1-iteration%2] = 1;
    if (converged[iteration%2])
        return; // The constraint iteration has already converged.
    updateCCMAAtomPositions(atoms, numAtomConstraints, atomConstraints, constraintDistance, atomPositions, velm,
            delta1, delta2, iteration);
}

/**
 * Run the entire CCMA iteration within a single kernel.  This has far less overhead than
 * using multiple kernels, but requires the calculation to use only a single workgroup.
 * That makes it faster for small numbers of constraints, but slower for large numbers.
 */
KERNEL void runCCMA(int constrainVelocities, GLOBAL const int* RESTRICT atoms, GLOBAL const int* RESTRICT numAtomConstraints, GLOBAL const int* RESTRICT atomConstraints,
        GLOBAL const int2* RESTRICT constraintAtoms, GLOBAL mixed4* RESTRICT constraintDistance, GLOBAL const real4* RESTRICT atomPositions,
        GLOBAL mixed4s* RESTRICT velm, GLOBAL mixed4s* RESTRICT posDelta, GLOBAL const mixed* RESTRICT reducedMass,
        GLOBAL mixed* RESTRICT delta1, GLOBAL mixed* RESTRICT delta2, GLOBAL const int* RESTRICT constraintMatrixColumn,
        GLOBAL const mixed* RESTRICT constraintMatrixValue, mixed tol
#ifdef USE_MIXED_PRECISION
        , GLOBAL const real4* RESTRICT posqCorrection
#endif
        ) {
    LOCAL int groupConverged;
#ifdef USE_MIXED_PRECISION
    computeCCMAConstraintDirections(constraintAtoms, constraintDistance, atomPositions, posqCorrection);
#else
    computeCCMAConstraintDirections(constraintAtoms, constraintDistance, atomPositions);
#endif
    for (int iteration = 0; iteration < 150; iteration++) {
        SYNC_THREADS
        if (constrainVelocities)
            computeCCMAVelocityConstraintForce(constraintAtoms, constraintDistance, velm, reducedMass,
                    delta1, tol, iteration, &groupConverged);
        else
            computeCCMAPositionConstraintForce(constraintAtoms, constraintDistance, posDelta, reducedMass,
                    delta1, tol, iteration, &groupConverged);
        SYNC_THREADS
        multiplyByCCMAConstraintMatrix(delta1, delta2, constraintMatrixColumn, constraintMatrixValue, iteration);
        SYNC_THREADS
        if (constrainVelocities)
            updateCCMAAtomPositions(atoms, numAtomConstraints, atomConstraints, constraintDistance, velm, velm,
                    delta1, delta2, iteration);
        else
            updateCCMAAtomPositions(atoms, numAtomConstraints, atomConstraints, constraintDistance, posDelta, velm,
                    delta1, delta2, iteration);
        SYNC_THREADS
        if (groupConverged)
            return;
    }
}

/**
 * Compute the positions of virtual sites
 */
KERNEL void computeVirtualSites(GLOBAL real4* RESTRICT posq, GLOBAL real4* RESTRICT posqCorrection,
        GLOBAL const int4* RESTRICT avg2Atoms, GLOBAL const real2* RESTRICT avg2Weights,
        GLOBAL const int4* RESTRICT avg3Atoms, GLOBAL const real4* RESTRICT avg3Weights,
        GLOBAL const int4* RESTRICT outOfPlaneAtoms, GLOBAL const real4* RESTRICT outOfPlaneWeights,
        GLOBAL const int* RESTRICT localCoordsIndex, GLOBAL const int* RESTRICT localCoordsAtoms,
        GLOBAL const real* RESTRICT localCoordsWeights, GLOBAL const real4* RESTRICT localCoordsPos,
        GLOBAL const int* RESTRICT localCoordsStartIndex, GLOBAL const int2* RESTRICT symmetryAtoms,
        GLOBAL const real4* RESTRICT symmetryMatrix, GLOBAL const real4* RESTRICT symmetryOffset,
        GLOBAL const int* RESTRICT symmetryUseBox, real4 periodicBoxVecX, real4 periodicBoxVecY,
        real4 periodicBoxVecZ, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ,
        GLOBAL const int* RESTRICT vsiteStage, int currentStage) {
    
    // Two particle average sites.
    
    for (int index = GLOBAL_ID; index < NUM_2_AVERAGE; index += GLOBAL_SIZE) {
        int4 atoms = avg2Atoms[index];
#ifdef MULTIPLE_VSITE_STAGES
        if (vsiteStage[atoms.x] != currentStage)
            continue;
#endif
        real2 weights = avg2Weights[index];
        mixed4 pos = loadPos(posq, posqCorrection, atoms.x);
        mixed4 pos1 = loadPos(posq, posqCorrection, atoms.y);
        mixed4 pos2 = loadPos(posq, posqCorrection, atoms.z);
        pos.x = pos1.x*weights.x + pos2.x*weights.y;
        pos.y = pos1.y*weights.x + pos2.y*weights.y;
        pos.z = pos1.z*weights.x + pos2.z*weights.y;
        storePos(posq, posqCorrection, atoms.x, pos);
    }
    
    // Three particle average sites.
    
    for (int index = GLOBAL_ID; index < NUM_3_AVERAGE; index += GLOBAL_SIZE) {
        int4 atoms = avg3Atoms[index];
#ifdef MULTIPLE_VSITE_STAGES
        if (vsiteStage[atoms.x] != currentStage)
            continue;
#endif
        real4 weights = avg3Weights[index];
        mixed4 pos = loadPos(posq, posqCorrection, atoms.x);
        mixed4 pos1 = loadPos(posq, posqCorrection, atoms.y);
        mixed4 pos2 = loadPos(posq, posqCorrection, atoms.z);
        mixed4 pos3 = loadPos(posq, posqCorrection, atoms.w);
        pos.x = pos1.x*weights.x + pos2.x*weights.y + pos3.x*weights.z;
        pos.y = pos1.y*weights.x + pos2.y*weights.y + pos3.y*weights.z;
        pos.z = pos1.z*weights.x + pos2.z*weights.y + pos3.z*weights.z;
        storePos(posq, posqCorrection, atoms.x, pos);
    }
    
    // Out of plane sites.
    
    for (int index = GLOBAL_ID; index < NUM_OUT_OF_PLANE; index += GLOBAL_SIZE) {
        int4 atoms = outOfPlaneAtoms[index];
#ifdef MULTIPLE_VSITE_STAGES
        if (vsiteStage[atoms.x] != currentStage)
            continue;
#endif
        real4 weights = outOfPlaneWeights[index];
        mixed4 pos = loadPos(posq, posqCorrection, atoms.x);
        mixed4 pos1 = loadPos(posq, posqCorrection, atoms.y);
        mixed4 pos2 = loadPos(posq, posqCorrection, atoms.z);
        mixed4 pos3 = loadPos(posq, posqCorrection, atoms.w);
        mixed4 v12 = pos2-pos1;
        mixed4 v13 = pos3-pos1;
        mixed4 cr = cross(v12, v13);
        pos.x = pos1.x + v12.x*weights.x + v13.x*weights.y + cr.x*weights.z;
        pos.y = pos1.y + v12.y*weights.x + v13.y*weights.y + cr.y*weights.z;
        pos.z = pos1.z + v12.z*weights.x + v13.z*weights.y + cr.z*weights.z;
        storePos(posq, posqCorrection, atoms.x, pos);
    }
    
    // Local coordinates sites.
    
    for (int index = GLOBAL_ID; index < NUM_LOCAL_COORDS; index += GLOBAL_SIZE) {
        int siteAtomIndex = localCoordsIndex[index];
#ifdef MULTIPLE_VSITE_STAGES
        if (vsiteStage[siteAtomIndex] != currentStage)
            continue;
#endif
        int start = localCoordsStartIndex[index];
        int end = localCoordsStartIndex[index+1];
        mixed3 origin = make_mixed3(0), xdir = make_mixed3(0), ydir = make_mixed3(0);
        for (int j = start; j < end; j++) {
            mixed3 pos = trimTo3(loadPos(posq, posqCorrection, localCoordsAtoms[j]));
            origin += pos*localCoordsWeights[3*j];
            xdir += pos*localCoordsWeights[3*j+1];
            ydir += pos*localCoordsWeights[3*j+2];
        }
        mixed3 zdir = cross(xdir, ydir);
        mixed normXdir = sqrt(xdir.x*xdir.x+xdir.y*xdir.y+xdir.z*xdir.z);
        mixed normZdir = sqrt(zdir.x*zdir.x+zdir.y*zdir.y+zdir.z*zdir.z);
        mixed invNormXdir = (normXdir > 0 ? 1/normXdir : 0);
        mixed invNormZdir = (normZdir > 0 ? 1/normZdir : 0);
        xdir *= invNormXdir;
        zdir *= invNormZdir;
        ydir = cross(zdir, xdir);
        real4 localPosition_4 = localCoordsPos[index];
        mixed3 localPosition = make_mixed3(localPosition_4.x, localPosition_4.y, localPosition_4.z);
        mixed4 pos = loadPos(posq, posqCorrection, siteAtomIndex);
        pos.x = origin.x + xdir.x*localPosition.x + ydir.x*localPosition.y + zdir.x*localPosition.z;
        pos.y = origin.y + xdir.y*localPosition.x + ydir.y*localPosition.y + zdir.y*localPosition.z;
        pos.z = origin.z + xdir.z*localPosition.x + ydir.z*localPosition.y + zdir.z*localPosition.z;
        storePos(posq, posqCorrection, siteAtomIndex, pos);
    }

    // Symmetry sites.

    for (int index = GLOBAL_ID; index < NUM_SYMMETRY; index += GLOBAL_SIZE) {
        int2 atoms = symmetryAtoms[index];
        real4 Rx = symmetryMatrix[3*index];
        real4 Ry = symmetryMatrix[3*index+1];
        real4 Rz = symmetryMatrix[3*index+2];
        real4 v = symmetryOffset[index];
        int useBoxVectors = symmetryUseBox[index];
        mixed4 pos = loadPos(posq, posqCorrection, atoms.x);
        mixed4 r = loadPos(posq, posqCorrection, atoms.y);
        if (useBoxVectors) {
            r.x = r.x*recipBoxVecX.x + r.y*recipBoxVecY.x + r.z*recipBoxVecZ.x;
            r.y = r.y*recipBoxVecY.y + r.z*recipBoxVecZ.y,
            r.z = r.z*recipBoxVecZ.z;
        }
        pos.x = Rx.x*r.x + Rx.y*r.y + Rx.z*r.z + v.x;
        pos.y = Ry.x*r.x + Ry.y*r.y + Ry.z*r.z + v.y;
        pos.z = Rz.x*r.x + Rz.y*r.y + Rz.z*r.z + v.z;
        if (useBoxVectors) {
            pos.x = pos.x*periodicBoxVecX.x + pos.y*periodicBoxVecY.x + pos.z*periodicBoxVecZ.x;
            pos.y = pos.y*periodicBoxVecY.y + pos.z*periodicBoxVecZ.y,
            pos.z = pos.z*periodicBoxVecZ.z;
        }
        storePos(posq, posqCorrection, atoms.x, pos);
    }
}

inline DEVICE real3 loadForce(int index, GLOBAL const mm_long* RESTRICT force) {
    real scale = 1/((real) 0x100000000);
    return make_real3(scale*force[index], scale*force[index+PADDED_NUM_ATOMS], scale*force[index+PADDED_NUM_ATOMS*2]);
}

inline DEVICE void addForce(int index, GLOBAL mm_long* RESTRICT force, real3 value) {
    GLOBAL mm_ulong* f = (GLOBAL mm_ulong*) force;
#ifdef HAS_OVERLAPPING_VSITES
    ATOMIC_ADD(&f[index], (mm_ulong) realToFixedPoint(value.x));
    ATOMIC_ADD(&f[index+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(value.y));
    ATOMIC_ADD(&f[index+PADDED_NUM_ATOMS*2], (mm_ulong) realToFixedPoint(value.z));
#else
    f[index] += (mm_ulong) realToFixedPoint(value.x);
    f[index+PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(value.y);
    f[index+PADDED_NUM_ATOMS*2] += (mm_ulong) realToFixedPoint(value.z);
#endif
}

/**
 * Distribute forces from virtual sites to the atoms they are based on.
 */
KERNEL void distributeVirtualSiteForces(GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection, GLOBAL mm_long* RESTRICT force,
        GLOBAL const int4* RESTRICT avg2Atoms, GLOBAL const real2* RESTRICT avg2Weights,
        GLOBAL const int4* RESTRICT avg3Atoms, GLOBAL const real4* RESTRICT avg3Weights,
        GLOBAL const int4* RESTRICT outOfPlaneAtoms, GLOBAL const real4* RESTRICT outOfPlaneWeights,
        GLOBAL const int* RESTRICT localCoordsIndex, GLOBAL const int* RESTRICT localCoordsAtoms,
        GLOBAL const real* RESTRICT localCoordsWeights, GLOBAL const real4* RESTRICT localCoordsPos,
        GLOBAL const int* RESTRICT localCoordsStartIndex, GLOBAL const int2* RESTRICT symmetryAtoms,
        GLOBAL const real4* RESTRICT symmetryMatrix, GLOBAL const real4* RESTRICT symmetryOffset,
        GLOBAL const int* RESTRICT symmetryUseBox, real4 periodicBoxVecX, real4 periodicBoxVecY,
        real4 periodicBoxVecZ, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ,
        GLOBAL const int* RESTRICT vsiteStage, int currentStage) {
    
    // Two particle average sites.
    
    for (int index = GLOBAL_ID; index < NUM_2_AVERAGE; index += GLOBAL_SIZE) {
        int4 atoms = avg2Atoms[index];
#ifdef MULTIPLE_VSITE_STAGES
        if (vsiteStage[atoms.x] != currentStage)
            continue;
#endif
        real2 weights = avg2Weights[index];
        real3 f = loadForce(atoms.x, force);
        addForce(atoms.y, force, f*weights.x);
        addForce(atoms.z, force, f*weights.y);
    }
    
    // Three particle average sites.
    
    for (int index = GLOBAL_ID; index < NUM_3_AVERAGE; index += GLOBAL_SIZE) {
        int4 atoms = avg3Atoms[index];
#ifdef MULTIPLE_VSITE_STAGES
        if (vsiteStage[atoms.x] != currentStage)
            continue;
#endif
        real4 weights = avg3Weights[index];
        real3 f = loadForce(atoms.x, force);
        addForce(atoms.y, force, f*weights.x);
        addForce(atoms.z, force, f*weights.y);
        addForce(atoms.w, force, f*weights.z);
    }
    
    // Out of plane sites.
    
    for (int index = GLOBAL_ID; index < NUM_OUT_OF_PLANE; index += GLOBAL_SIZE) {
        int4 atoms = outOfPlaneAtoms[index];
#ifdef MULTIPLE_VSITE_STAGES
        if (vsiteStage[atoms.x] != currentStage)
            continue;
#endif
        real4 weights = outOfPlaneWeights[index];
        mixed4 pos1 = loadPos(posq, posqCorrection, atoms.y);
        mixed4 pos2 = loadPos(posq, posqCorrection, atoms.z);
        mixed4 pos3 = loadPos(posq, posqCorrection, atoms.w);
        mixed4 v12 = pos2-pos1;
        mixed4 v13 = pos3-pos1;
        real3 f = loadForce(atoms.x, force);
        real3 fp2 = make_real3((real) (weights.x*f.x - weights.z*v13.z*f.y + weights.z*v13.y*f.z),
                   (real) (weights.z*v13.z*f.x + weights.x*f.y - weights.z*v13.x*f.z),
                   (real) (-weights.z*v13.y*f.x + weights.z*v13.x*f.y + weights.x*f.z));
        real3 fp3 = make_real3((real) (weights.y*f.x + weights.z*v12.z*f.y - weights.z*v12.y*f.z),
                   (real) (-weights.z*v12.z*f.x + weights.y*f.y + weights.z*v12.x*f.z),
                   (real) (weights.z*v12.y*f.x - weights.z*v12.x*f.y + weights.y*f.z));
        addForce(atoms.y, force, f-fp2-fp3);
        addForce(atoms.z, force, fp2);
        addForce(atoms.w, force, fp3);
    }
    
    // Local coordinates sites.
    
    for (int index = GLOBAL_ID; index < NUM_LOCAL_COORDS; index += GLOBAL_SIZE) {
        int siteAtomIndex = localCoordsIndex[index];
#ifdef MULTIPLE_VSITE_STAGES
        if (vsiteStage[siteAtomIndex] != currentStage)
            continue;
#endif
        int start = localCoordsStartIndex[index];
        int end = localCoordsStartIndex[index+1];
        mixed3 origin = make_mixed3(0), xdir = make_mixed3(0), ydir = make_mixed3(0);
        for (int j = start; j < end; j++) {
            mixed3 pos = trimTo3(loadPos(posq, posqCorrection, localCoordsAtoms[j]));
            origin += pos*localCoordsWeights[3*j];
            xdir += pos*localCoordsWeights[3*j+1];
            ydir += pos*localCoordsWeights[3*j+2];
        }
        mixed3 zdir = cross(xdir, ydir);
        mixed normXdir = sqrt(xdir.x*xdir.x+xdir.y*xdir.y+xdir.z*xdir.z);
        mixed normZdir = sqrt(zdir.x*zdir.x+zdir.y*zdir.y+zdir.z*zdir.z);
        mixed invNormXdir = (normXdir > 0 ? 1/normXdir : 0);
        mixed invNormZdir = (normZdir > 0 ? 1/normZdir : 0);
        mixed3 dx = xdir*invNormXdir;
        mixed3 dz = zdir*invNormZdir;
        mixed3 dy = cross(dz, dx);
        real4 localPosition_4 = localCoordsPos[index];
        mixed3 localPosition = make_mixed3(localPosition_4.x, localPosition_4.y, localPosition_4.z);

        // The derivatives for this case are very complicated.  They were computed with SymPy then simplified by hand.

        real3 f = loadForce(siteAtomIndex, force);
        mixed3 fp1 = localPosition*f.x;
        mixed3 fp2 = localPosition*f.y;
        mixed3 fp3 = localPosition*f.z;
        for (int j = start; j < end; j++) {
            real originWeight = localCoordsWeights[3*j];
            real wx = localCoordsWeights[3*j+1];
            real wy = localCoordsWeights[3*j+2];
            mixed wxScaled = wx*invNormXdir;
            mixed t1 = (wx*ydir.x-wy*xdir.x)*invNormZdir;
            mixed t2 = (wx*ydir.y-wy*xdir.y)*invNormZdir;
            mixed t3 = (wx*ydir.z-wy*xdir.z)*invNormZdir;
            mixed sx = t3*dz.y-t2*dz.z;
            mixed sy = t1*dz.z-t3*dz.x;
            mixed sz = t2*dz.x-t1*dz.y;
            real3 fresult = make_real3(0);
            fresult.x += fp1.x*wxScaled*(1-dx.x*dx.x) + fp1.z*(dz.x*sx   ) + fp1.y*((-dx.x*dy.x     )*wxScaled + dy.x*sx - dx.y*t2 - dx.z*t3) + f.x*originWeight;
            fresult.y += fp1.x*wxScaled*( -dx.x*dx.y) + fp1.z*(dz.x*sy+t3) + fp1.y*((-dx.y*dy.x-dz.z)*wxScaled + dy.x*sy + dx.y*t1);
            fresult.z += fp1.x*wxScaled*( -dx.x*dx.z) + fp1.z*(dz.x*sz-t2) + fp1.y*((-dx.z*dy.x+dz.y)*wxScaled + dy.x*sz + dx.z*t1);
            fresult.x += fp2.x*wxScaled*( -dx.y*dx.x) + fp2.z*(dz.y*sx-t3) - fp2.y*(( dx.x*dy.y-dz.z)*wxScaled - dy.y*sx - dx.x*t2);
            fresult.y += fp2.x*wxScaled*(1-dx.y*dx.y) + fp2.z*(dz.y*sy   ) - fp2.y*(( dx.y*dy.y     )*wxScaled - dy.y*sy + dx.x*t1 + dx.z*t3) + f.y*originWeight;
            fresult.z += fp2.x*wxScaled*( -dx.y*dx.z) + fp2.z*(dz.y*sz+t1) - fp2.y*(( dx.z*dy.y+dz.x)*wxScaled - dy.y*sz - dx.z*t2);
            fresult.x += fp3.x*wxScaled*( -dx.z*dx.x) + fp3.z*(dz.z*sx+t2) + fp3.y*((-dx.x*dy.z-dz.y)*wxScaled + dy.z*sx + dx.x*t3);
            fresult.y += fp3.x*wxScaled*( -dx.z*dx.y) + fp3.z*(dz.z*sy-t1) + fp3.y*((-dx.y*dy.z+dz.x)*wxScaled + dy.z*sy + dx.y*t3);
            fresult.z += fp3.x*wxScaled*(1-dx.z*dx.z) + fp3.z*(dz.z*sz   ) + fp3.y*((-dx.z*dy.z     )*wxScaled + dy.z*sz - dx.x*t1 - dx.y*t2) + f.z*originWeight;
            addForce(localCoordsAtoms[j], force, fresult);
        }
    }

    // Symmetry sites.

    for (int index = GLOBAL_ID; index < NUM_SYMMETRY; index += GLOBAL_SIZE) {
        int2 atoms = symmetryAtoms[index];
        real4 Rx = symmetryMatrix[3*index];
        real4 Ry = symmetryMatrix[3*index+1];
        real4 Rz = symmetryMatrix[3*index+2];
        real4 v = symmetryOffset[index];
        int useBoxVectors = symmetryUseBox[index];
        real3 f = loadForce(atoms.x, force);
        if (useBoxVectors) {
            f.x = f.x*periodicBoxVecX.x + f.y*periodicBoxVecY.x + f.z*periodicBoxVecZ.x;
            f.y = f.y*periodicBoxVecY.y + f.z*periodicBoxVecZ.y,
            f.z = f.z*periodicBoxVecZ.z;
        }
        real3 f1 = make_real3(Rx.x*f.x + Ry.x*f.y + Rz.x*f.z,
                              Rx.y*f.x + Ry.y*f.y + Rz.y*f.z,
                              Rx.z*f.x + Ry.z*f.y + Rz.z*f.z);
        if (useBoxVectors) {
            f1.x = f1.x*recipBoxVecX.x + f1.y*recipBoxVecY.x + f1.z*recipBoxVecZ.x;
            f1.y = f1.y*recipBoxVecY.y + f1.z*recipBoxVecZ.y,
            f1.z = f1.z*recipBoxVecZ.z;
        }
        addForce(atoms.y, force, f1);
    }
}

/**
 * Copy the distributed forces from the long buffer back to the float buffer.
 */
KERNEL void saveDistributedForces(GLOBAL const mm_long* RESTRICT longForces, GLOBAL real4* RESTRICT forces) {
    for (int index = GLOBAL_ID; index < NUM_ATOMS; index += GLOBAL_SIZE) {
        real3 f = loadForce(index, longForces);
        forces[index] = make_real4(f.x, f.y, f.z, 0);
    }
}

/**
 * Apply a time shift to the velocities before computing kinetic energy.
 */
KERNEL void timeShiftVelocities(GLOBAL mixed4s* RESTRICT velm, GLOBAL const mm_long* RESTRICT force, real timeShift) {
    const mixed scale = timeShift/(mixed) 0x100000000;
    for (int index = GLOBAL_ID; index < NUM_ATOMS; index += GLOBAL_SIZE) {
        mixed4 velocity = velm[index];
        if (velocity.w != 0.0) {
            velocity.x += scale*force[index]*velocity.w;
            velocity.y += scale*force[index+PADDED_NUM_ATOMS]*velocity.w;
            velocity.z += scale*force[index+PADDED_NUM_ATOMS*2]*velocity.w;
            velm[index] = velocity;
        }
    }
}

/**
 * Compute the total kinetic energy.
 */
KERNEL void computeKineticEnergy(GLOBAL mixed4s* RESTRICT velm, GLOBAL mixed* result) {
    LOCAL mixed tempBuffer[KE_WORK_GROUP_SIZE];
    mixed sum = 0;
    for (unsigned int index = LOCAL_ID; index < NUM_ATOMS; index += LOCAL_SIZE) {
        mixed4 v = velm[index];
        if (v.w != 0)
            sum += (v.x*v.x+v.y*v.y+v.z*v.z)/v.w;
    }
    tempBuffer[LOCAL_ID] = sum;
    for (int i = 1; i < KE_WORK_GROUP_SIZE; i *= 2) {
        SYNC_THREADS;
        if (LOCAL_ID%(i*2) == 0 && LOCAL_ID+i < KE_WORK_GROUP_SIZE)
            tempBuffer[LOCAL_ID] += tempBuffer[LOCAL_ID+i];
    }
    if (LOCAL_ID == 0)
        *result = 0.5f*tempBuffer[0];
}
