/**
 * Evaluate the fast versions of math functions, so the host can decide whether they are
 * accurate enough to use.  Each value occupies five elements: the input, then rsqrt,
 * reciprocal, exp, and log of it.
 */
KERNEL void determineFastAccuracy(GLOBAL float* RESTRICT values, int numValues) {
    for (int i = GLOBAL_ID; i < numValues; i += GLOBAL_SIZE) {
        float v = values[5*i];
        values[5*i+1] = fast::rsqrt(v);
        values[5*i+2] = fast::divide(1.0f, v);
        values[5*i+3] = fast::exp(v);
        values[5*i+4] = fast::log(v);
    }
}
