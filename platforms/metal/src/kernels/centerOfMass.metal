/**
 * Compute the center of mass velocity in one thread group and write it to cmVelocity[0].  The group has a power of 2
 * threads, at most CM_BLOCK_SIZE.  One group needs no second pass to combine partial sums, so the fused LangevinMiddle
 * kernel can read the result directly.  The kernel runs alongside the force kernels, which hide it only while it is
 * shorter than they are.  A single group is bound by memory latency, so each thread keeps CM_LOADS velocities in
 * flight.  Each term is a float product, which in mixed precision skips a df64 reciprocal and three df64 products per
 * atom, but the sums are mixed.  With 256 threads, one load at a time and df64 arithmetic it took 207 us per step on
 * the Ultra for 92k atoms, and slowed the steps it was meant to hide in.  Float sums of hundreds of terms per thread
 * left 1.8x (92k atoms) to 3.2x (1M atoms) the momentum residual of removeCenterOfMassMomentum's many short sums.
 */
#define CM_LOADS 4
KERNEL void calcCenterOfMassVelocity(int numAtoms, GLOBAL const mixed4* RESTRICT velm, GLOBAL float4* RESTRICT cmVelocity) {
    LOCAL mixed tempX[CM_BLOCK_SIZE];
    LOCAL mixed tempY[CM_BLOCK_SIZE];
    LOCAL mixed tempZ[CM_BLOCK_SIZE];
    mixed cmX = 0, cmY = 0, cmZ = 0;
    for (int first = LOCAL_ID; first < numAtoms; first += CM_LOADS*LOCAL_SIZE) {
        float4 velocity[CM_LOADS];
        for (int i = 0; i < CM_LOADS; i++) {
            int index = first+i*LOCAL_SIZE;
            velocity[i] = (index < numAtoms ? (float4) mixed4(velm[index]) : make_float4(0));
        }
        for (int i = 0; i < CM_LOADS; i++) {
            if (velocity[i].w != 0) {
                float mass = RECIP(velocity[i].w);
                cmX += velocity[i].x*mass;
                cmY += velocity[i].y*mass;
                cmZ += velocity[i].z*mass;
            }
        }
    }
    int thread = LOCAL_ID;
    tempX[thread] = cmX;
    tempY[thread] = cmY;
    tempZ[thread] = cmZ;
    for (int step = LOCAL_SIZE/2; step > 0; step /= 2) {
        SYNC_THREADS;
        if (thread < step) {
            tempX[thread] += tempX[thread+step];
            tempY[thread] += tempY[thread+step];
            tempZ[thread] += tempZ[thread+step];
        }
    }
    if (thread == 0)
        cmVelocity[0] = make_float4((float) (CM_INVERSE_TOTAL_MASS*tempX[0]), (float) (CM_INVERSE_TOTAL_MASS*tempY[0]),
                (float) (CM_INVERSE_TOTAL_MASS*tempZ[0]), 0);
}
