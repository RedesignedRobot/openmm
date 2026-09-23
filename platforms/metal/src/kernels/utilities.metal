/**
 * Evaluate the fast versions of math functions, so the host can decide whether they are
 * accurate enough to use.  Each value occupies six elements: the input, then sqrt, rsqrt,
 * reciprocal, exp, and log of it.
 */
KERNEL void determineFastAccuracy(GLOBAL float* RESTRICT values, int numValues) {
    for (int i = GLOBAL_ID; i < numValues; i += GLOBAL_SIZE) {
        float v = values[6*i];
        values[6*i+1] = fast::sqrt(v);
        values[6*i+2] = fast::rsqrt(v);
        values[6*i+3] = fast::divide(1.0f, v);
        values[6*i+4] = fast::exp(v);
        values[6*i+5] = fast::log(v);
    }
}
