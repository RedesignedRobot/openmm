/**
 * Kernels for MetalAtomReorder, which applies an atom order computed ahead of time without a host round trip.
 * order[i] is {old index, box shift as three signed bytes, original atom index, 0} for the atom placed at i.
 * A permutation cannot be applied in place, so gatherAtoms copies each atom's shifted position, its correction
 * and its raw velocity words into scratch, and storeAtoms writes them back.  Velocities are copied as stored
 * (float pairs in mixed), so they come through bit for bit.  Padding atoms become zeros, as in
 * ComputeContext::reorderAtomsImpl().
 */
#ifdef USE_MIXED_PRECISION
#define VEL_WORDS 2
#define WORDS 4
#else
#define VEL_WORDS 1
#define WORDS 2
#endif

KERNEL void copyPositions(int numAtoms, GLOBAL const float4* RESTRICT posq, GLOBAL float4* RESTRICT snapshot) {
    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE)
        snapshot[i] = posq[i];
}

KERNEL void gatherAtoms(int numAtoms, int paddedNumAtoms, GLOBAL const int4* RESTRICT order, GLOBAL const float4* RESTRICT posq,
#ifdef USE_MIXED_PRECISION
        GLOBAL const float4* RESTRICT posqCorrection,
#endif
        GLOBAL const mixed4s* RESTRICT velm, GLOBAL float4* RESTRICT scratch, float4 boxXHi, float4 boxYHi, float4 boxZHi,
        float4 boxXLo, float4 boxYLo, float4 boxZLo) {
    GLOBAL const float4* velocityWords = (GLOBAL const float4*) velm;
    for (int i = GLOBAL_ID; i < paddedNumAtoms; i += GLOBAL_SIZE) {
        if (i >= numAtoms) {
            for (int k = 0; k < WORDS; k++)
                scratch[WORDS*i+k] = make_float4(0);
            continue;
        }
        int4 entry = order[i];
        int old = entry.x;
        float4 pos = posq[old];
#ifdef USE_MIXED_PRECISION
        float4 corr = posqCorrection[old];
#endif
        if (entry.y != 0) {
            float cx = (float) ((entry.y << 24) >> 24);
            float cy = (float) ((entry.y << 16) >> 24);
            float cz = (float) ((entry.y << 8) >> 24);
#ifdef USE_MIXED_PRECISION
            // Shift in df64 and split again, as the host path does in double.
            df64 x = df64(pos.x, corr.x) - (df64(boxXHi.x, boxXLo.x)*cx + df64(boxYHi.x, boxYLo.x)*cy + df64(boxZHi.x, boxZLo.x)*cz);
            df64 y = df64(pos.y, corr.y) - (df64(boxYHi.y, boxYLo.y)*cy + df64(boxZHi.y, boxZLo.y)*cz);
            df64 z = df64(pos.z, corr.z) - df64(boxZHi.z, boxZLo.z)*cz;
            pos.x = x.hi;
            pos.y = y.hi;
            pos.z = z.hi;
            corr.x = x.lo;
            corr.y = y.lo;
            corr.z = z.lo;
#else
            pos.x -= boxXHi.x*cx + boxYHi.x*cy + boxZHi.x*cz;
            pos.y -= boxYHi.y*cy + boxZHi.y*cz;
            pos.z -= boxZHi.z*cz;
#endif
        }
        scratch[WORDS*i] = pos;
#ifdef USE_MIXED_PRECISION
        scratch[WORDS*i+1] = corr;
#endif
        for (int k = 0; k < VEL_WORDS; k++)
            scratch[WORDS*i+WORDS-VEL_WORDS+k] = velocityWords[VEL_WORDS*old+k];
    }
}

KERNEL void storeAtoms(int numAtoms, int paddedNumAtoms, GLOBAL const int4* RESTRICT order, GLOBAL const float4* RESTRICT scratch,
        GLOBAL float4* RESTRICT posq,
#ifdef USE_MIXED_PRECISION
        GLOBAL float4* RESTRICT posqCorrection,
#endif
        GLOBAL mixed4s* RESTRICT velm, GLOBAL int* RESTRICT atomIndex) {
    GLOBAL float4* velocityWords = (GLOBAL float4*) velm;
    for (int i = GLOBAL_ID; i < paddedNumAtoms; i += GLOBAL_SIZE) {
        posq[i] = scratch[WORDS*i];
#ifdef USE_MIXED_PRECISION
        posqCorrection[i] = scratch[WORDS*i+1];
#endif
        for (int k = 0; k < VEL_WORDS; k++)
            velocityWords[VEL_WORDS*i+k] = scratch[WORDS*i+WORDS-VEL_WORDS+k];
        if (i < numAtoms)
            atomIndex[i] = order[i].z;
    }
}
