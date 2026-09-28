/**
 * Compute the center of mass velocity in one thread group and write it to cmVelocity[0].  The group has a power of 2
 * threads, at most CM_BLOCK_SIZE.  One group needs no second pass to combine partial sums, so the fused LangevinMiddle
 * kernel can read the result directly.  The kernel runs alongside the force kernels, which hide it only while it is
 * shorter than they are.  A single group is bound by memory latency, so each thread keeps CM_LOADS velocities in
 * flight.  It sums in float, the precision of the result, which in mixed precision skips a df64 reciprocal and three
 * df64 products per atom.  With 256 threads, one load at a time and df64 arithmetic it took 207 us per step on the
 * Ultra for 92k atoms, and slowed the steps it was meant to hide in.
 */
#define CM_LOADS 4
KERNEL void calcCenterOfMassVelocity(int numAtoms, GLOBAL const mixed4s* RESTRICT velm, GLOBAL float4* RESTRICT cmVelocity) {
    LOCAL float4 temp[CM_BLOCK_SIZE];
    float4 cm = make_float4(0);
    for (int first = LOCAL_ID; first < numAtoms; first += CM_LOADS*LOCAL_SIZE) {
        float4 velocity[CM_LOADS];
        for (int i = 0; i < CM_LOADS; i++) {
            int index = first+i*LOCAL_SIZE;
            velocity[i] = (index < numAtoms ? (float4) mixed4(velm[index]) : make_float4(0));
        }
        for (int i = 0; i < CM_LOADS; i++) {
            if (velocity[i].w != 0) {
                float mass = RECIP(velocity[i].w);
                cm.x += velocity[i].x*mass;
                cm.y += velocity[i].y*mass;
                cm.z += velocity[i].z*mass;
            }
        }
    }
    int thread = LOCAL_ID;
    temp[thread] = cm;
    for (int step = LOCAL_SIZE/2; step > 0; step /= 2) {
        SYNC_THREADS;
        if (thread < step)
            temp[thread] += temp[thread+step];
    }
    if (thread == 0)
        cmVelocity[0] = make_float4(INVERSE_TOTAL_MASS*temp[0].x, INVERSE_TOTAL_MASS*temp[0].y, INVERSE_TOTAL_MASS*temp[0].z, 0);
}
