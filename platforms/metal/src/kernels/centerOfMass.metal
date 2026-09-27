/**
 * Compute the center of mass velocity in one thread group of CM_BLOCK_SIZE threads (a power of 2), and write it to
 * cmVelocity[0].  One group needs no second pass to combine partial sums, so the fused LangevinMiddle kernel can read
 * the result directly.  The kernel runs alongside the force kernels, which hide its length.
 */
KERNEL void calcCenterOfMassVelocity(int numAtoms, GLOBAL const mixed4s* RESTRICT velm, GLOBAL float4* RESTRICT cmVelocity) {
    LOCAL float4 temp[CM_BLOCK_SIZE];
    float4 cm = make_float4(0);
    for (int index = LOCAL_ID; index < numAtoms; index += CM_BLOCK_SIZE) {
        mixed4 velocity = velm[index];
        if (velocity.w != 0) {
            mixed mass = RECIP(velocity.w);
            cm.x += (float) (velocity.x*mass);
            cm.y += (float) (velocity.y*mass);
            cm.z += (float) (velocity.z*mass);
        }
    }
    int thread = LOCAL_ID;
    temp[thread] = cm;
    for (int step = CM_BLOCK_SIZE/2; step > 0; step /= 2) {
        SYNC_THREADS;
        if (thread < step)
            temp[thread] += temp[thread+step];
    }
    if (thread == 0)
        cmVelocity[0] = make_float4(INVERSE_TOTAL_MASS*temp[0].x, INVERSE_TOTAL_MASS*temp[0].y, INVERSE_TOTAL_MASS*temp[0].z, 0);
}
