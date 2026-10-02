// State entries are IEEE doubles in memory, df64 in registers, in both precisions.
// 0 volumeScale, 1 oldVolume, 2 deltaVolume, 3 w, 4/5 energies, 6 accepted,
// 7/8 adaptation counts, 9/10 lifetime counts, 11 RNG cursor, 12 seed.
// Philox4x32-10, a separate counter/key space owned by this barostat.
inline double barostatUniform(GLOBAL double* state) {
    ulong cursor = (ulong) (long) double(state[11]);
    uint4 counter = uint4((uint) cursor, (uint) (cursor >> 32), 0, 0);
    uint2 key = uint2((uint) (long) double(state[12]), 0);
    for (int round = 0; round < 10; round++) {
        ulong a = (ulong) 0xD2511F53u*counter.x;
        ulong b = (ulong) 0xCD9E8D57u*counter.z;
        counter = uint4((uint) (b >> 32)^counter.y^key.x, (uint) b,
                        (uint) (a >> 32)^counter.w^key.y, (uint) a);
        key += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    state[11] = double(cursor+1);
    return double(counter.x)*double(0x1p-32f);
}

KERNEL void proposeBarostat(GLOBAL double* state, GLOBAL const uchar* box,
        GLOBAL double* factor, int injected, double proposal) {
    double volume = *((GLOBAL const double*) (box+304));
    if (state[0] == 0.0f)
        state[0] = volume*double(0.01f, 2.2351742e-10f);
    double u = injected ? proposal : barostatUniform(state);
    double delta = state[0]*2.0f*(u-0.5f);
    state[1] = volume;
    state[2] = delta;
    factor[0] = exp(log((volume+delta)/volume)/3.0f);
}

KERNEL void decideBarostat(GLOBAL double* state, GLOBAL const mixed* initial,
        GLOBAL const mixed* final, int count, double initialHost, double finalHost,
        double pressure, double kT, int particles, int injected, double acceptance) {
    double e0 = initialHost, e1 = finalHost;
    for (int i = 0; i < count; i++) {
        e0 += double(initial[i]);
        e1 += double(final[i]);
    }
    double volume = state[1], delta = state[2];
    double w = e1-e0+pressure*delta-double(particles)*kT*log((volume+delta)/volume);
    bool accept = true;
    if (w > 0.0f) {
        double u = injected ? acceptance : barostatUniform(state);
        accept = u <= exp(-w/kT);
    }
    state[3] = w;
    state[4] = e0;
    state[5] = e1;
    state[6] = accept ? 1.0f : 0.0f;
    state[7] += 1.0f;
    state[8] += accept ? 1.0f : 0.0f;
    state[9] += 1.0f;
    state[10] += accept ? 1.0f : 0.0f;
    if (state[7] >= 10.0f) {
        double factor = double(1.1f, -2.3841858e-8f);
        if (state[8] < state[7]*0.25f) {
            state[0] /= factor;
            state[7] = 0.0f;
            state[8] = 0.0f;
        }
        else if (state[8] > state[7]*0.75f) {
            state[0] = min(state[0]*factor, volume*double(0.3f, -1.1920929e-8f));
            state[7] = 0.0f;
            state[8] = 0.0f;
        }
    }
}

// Copy words, including float corrections and IEEE box fields, without numeric conversion.
KERNEL void restoreBarostat(GLOBAL const double* state, GLOBAL const uint* saved,
        GLOBAL uint* target, int words) {
    if (state[6] == 0.0f)
        for (int i = GLOBAL_ID; i < words; i += GLOBAL_SIZE)
            target[i] = saved[i];
}
