/**
 * SHAKE and SETTLE in mixed precision.  df64 makes every mixed operation several times the cost of a real one, so
 * these kernels evaluate residuals in mixed precision but compute the corrections in real precision, and iterate
 * until the residuals meet the tolerance.  They replace the kernels of the same names without "Mixed" in
 * integrationUtilities.cc and take the same arguments.  The functions that constrain one cluster are also called by
 * the fused LangevinMiddle kernels.
 */

/**
 * Load the positions of the atoms in a SHAKE cluster at the start of the step, and their displacements or velocities.
 * The peripheral atoms a cluster lacks get zeros.
 */
inline DEVICE void loadShakeCluster(int4 atoms, GLOBAL const real4* RESTRICT oldPos, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL const mixed4* RESTRICT values, PRIVATE mixed4* pos, PRIVATE mixed4* value) {
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
inline DEVICE void storeShakeCluster(int4 atoms, GLOBAL mixed4* RESTRICT values, PRIVATE const mixed4* value) {
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
    real3 r1 = make_real3((real) rij1.x, (real) rij1.y, (real) rij1.z);
    real3 r2 = make_real3((real) rij2.x, (real) rij2.y, (real) rij2.z);
    real3 r3 = make_real3((real) rij3.x, (real) rij3.y, (real) rij3.z);
    real maxResidual = d2*(real) tol;

    // Iterate until convergence.

    bool converged = false;
    int iteration = 0;
    while (iteration < 15 && !converged) {
        converged = true;
        mixed3 rpij = make_mixed3(xpi.x-xpj1.x, xpi.y-xpj1.y, xpi.z-xpj1.z);
        mixed rpsqij = rpij.x*rpij.x + rpij.y*rpij.y + rpij.z*rpij.z;
        mixed rrpr = rij1.x*rpij.x + rij1.y*rpij.y + rij1.z*rpij.z;
        real residual = (real) (ld1-2.0f*rrpr-rpsqij);
        if (fabs(residual) >= maxResidual) {
            real acor = residual*avgMass / ((real) rrpr+(real) rij1sq);
            real3 dr = r1*acor;
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
            residual = (real) (ld2-2.0f*rrpr-rpsqij);
            if (fabs(residual) >= maxResidual) {
                real acor = residual*avgMass / ((real) rrpr+(real) rij2sq);
                real3 dr = r2*acor;
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
            residual = (real) (ld3-2.0f*rrpr-rpsqij);
            if (fabs(residual) >= maxResidual) {
                real acor = residual*avgMass / ((real) rrpr+(real) rij3sq);
                real3 dr = r3*acor;
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
KERNEL void applyShakeToPositionsMixed(int numClusters, mixed tol, GLOBAL const real4* RESTRICT oldPos,
        GLOBAL mixed4* RESTRICT posDelta, GLOBAL const int4* RESTRICT clusterAtoms, GLOBAL const float4* RESTRICT clusterParams,
        GLOBAL const real4* RESTRICT posqCorrection) {
    for (int index = GLOBAL_ID; index < numClusters; index += GLOBAL_SIZE) {
        int4 atoms = clusterAtoms[index];
        mixed4 pos[4], xp[4];
        loadShakeCluster(atoms, oldPos, posqCorrection, posDelta, pos, xp);
        applyShakeToCluster(atoms, clusterParams[index], tol, pos, xp);
        storeShakeCluster(atoms, posDelta, xp);
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
    real3 r1 = make_real3((real) rij1.x, (real) rij1.y, (real) rij1.z);
    real3 r2 = make_real3((real) rij2.x, (real) rij2.y, (real) rij2.z);
    real3 r3 = make_real3((real) rij3.x, (real) rij3.y, (real) rij3.z);
    real rij1sq = r1.x*r1.x + r1.y*r1.y + r1.z*r1.z;
    real rij2sq = r2.x*r2.x + r2.y*r2.y + r2.z*r2.z;
    real rij3sq = r3.x*r3.x + r3.y*r3.y + r3.z*r3.z;

    // Iterate until convergence.

    bool converged = false;
    int iteration = 0;
    while (iteration < 15 && !converged) {
        converged = true;
        mixed3 rpij = make_mixed3(xpi.x-xpj1.x, xpi.y-xpj1.y, xpi.z-xpj1.z);
        mixed rrpr = rpij.x*rij1.x + rpij.y*rij1.y + rpij.z*rij1.z;
        real delta = -2.0f*avgMass*(real) rrpr/rij1sq;
        real3 dr = r1*delta;
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
            delta = -2.0f*avgMass*(real) rrpr/rij2sq;
            dr = r2*delta;
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
            delta = -2.0f*avgMass*(real) rrpr/rij3sq;
            dr = r3*delta;
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
KERNEL void applyShakeToVelocitiesMixed(int numClusters, mixed tol, GLOBAL const real4* RESTRICT oldPos,
        GLOBAL mixed4* RESTRICT posDelta, GLOBAL const int4* RESTRICT clusterAtoms, GLOBAL const float4* RESTRICT clusterParams,
        GLOBAL const real4* RESTRICT posqCorrection) {
    for (int index = GLOBAL_ID; index < numClusters; index += GLOBAL_SIZE) {
        int4 atoms = clusterAtoms[index];
        mixed4 pos[4], vel[4];
        loadShakeCluster(atoms, oldPos, posqCorrection, posDelta, pos, vel);
        applyShakeToClusterVelocities(atoms, clusterParams[index], tol, pos, vel);
        storeShakeCluster(atoms, posDelta, vel);
    }
}

// SETTLE positions are found by Newton iteration on the three bond residuals, in real precision.  The residual
// tolerance is relative to the squared bond length, and 1e-6 is about 8 float ulps of it.  Like SHAKE, the loop stops
// after 15 iterations without an error.
#define SETTLE_NEWTON_MAX_ITERATIONS 15
#define SETTLE_NEWTON_RESIDUAL 1e-6f

/**
 * Enforce constraints on one SETTLE cluster.  pos holds the positions of its atoms at the start of the step, invMass
 * their inverse masses, and delta their displacements, which this updates.
 */
DEVICE void applySettleToCluster(float2 params, mixed3 invMass, PRIVATE const mixed4* pos, PRIVATE mixed4* delta) {
    mixed4 apos0 = pos[0], apos1 = pos[1], apos2 = pos[2];
    mixed4 xp0 = delta[0], xp1 = delta[1], xp2 = delta[2];

    // Solve the same equations as SETTLE by Newton's method: move the atoms along the old bonds, weighted by
    // inverse mass, until the bonds have the right lengths.  The analytic solution in applySettleToPositions()
    // needs much more arithmetic in mixed precision.  The iterations run in real precision until the residuals
    // are near its resolution, then one last step from residuals in mixed precision brings them to about 1e-12 of
    // the squared bond lengths.

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
        last = (iteration == SETTLE_NEWTON_MAX_ITERATIONS-1 || (fabs(g01) < SETTLE_NEWTON_RESIDUAL*d01sq &&
                fabs(g02) < SETTLE_NEWTON_RESIDUAL*d01sq && fabs(g12) < SETTLE_NEWTON_RESIDUAL*d12sq));
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
    delta[0] = xp0;
    delta[1] = xp1;
    delta[2] = xp2;
}

/**
 * Enforce constraints on SETTLE clusters
 */
KERNEL void applySettleToPositionsMixed(int numClusters, mixed tol, GLOBAL const real4* RESTRICT oldPos,
        GLOBAL mixed4* RESTRICT posDelta, GLOBAL const mixed4* RESTRICT velm, GLOBAL const int4* RESTRICT clusterAtoms,
        GLOBAL const float2* RESTRICT clusterParams, GLOBAL const real4* RESTRICT posqCorrection) {
    for (int index = GLOBAL_ID; index < numClusters; index += GLOBAL_SIZE) {
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
    }
}

/**
 * Enforce velocity constraints on one SETTLE cluster.  pos holds the positions of its atoms at the start of the step,
 * and vel their velocities, which this updates.  The equations are those of applySettleToVelocities().
 */
DEVICE void applySettleToClusterVelocities(PRIVATE const mixed4* pos, PRIVATE mixed4* vel) {
    mixed4 apos0 = pos[0], apos1 = pos[1], apos2 = pos[2];
    mixed4 v0 = vel[0], v1 = vel[1], v2 = vel[2];

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
    vel[0] = v0;
    vel[1] = v1;
    vel[2] = v2;
}

/**
 * Enforce velocity constraints on SETTLE clusters
 */
KERNEL void applySettleToVelocitiesMixed(int numClusters, mixed tol, GLOBAL const real4* RESTRICT oldPos,
        GLOBAL mixed4* RESTRICT posDelta, GLOBAL mixed4* RESTRICT velm, GLOBAL const int4* RESTRICT clusterAtoms,
        GLOBAL const float2* RESTRICT clusterParams, GLOBAL const real4* RESTRICT posqCorrection) {
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
