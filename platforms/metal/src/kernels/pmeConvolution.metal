/**
 * The PME reciprocal space convolution in three kernels, for grids whose y-z plane fits in threadgroup memory.
 * planesForward does the real-to-complex transform along z and the transform along y for one x plane.
 * convolveLines does the transform along x, the convolution and the inverse transform along x for one y row.
 * planesInverse does the inverse transform along y and the complex-to-real transform along z for one x plane.
 * The result equals VkFFT's forward transform, reciprocalConvolution and VkFFT's inverse transform: the same
 * unnormalized transforms and the same half complex layout [x][y][kz], kz in 0..GRID_SIZE_Z/2.
 *
 * Each transform is an in-place mixed radix FFT in threadgroup memory: decimation in frequency leaves frequency k
 * at a digit reversed position, and decimation in time, run with the stages in reverse, undoes it.  The host
 * defines the grid size, the radices of each axis (RADICES_X etc.), and the stages as M(radix, length) lists
 * (STAGES_X(M) and STAGES_X_REV(M) etc.), and the roots of the odd radices (ROOTS_C3, ROOTS_S3 etc.).  The radices
 * are 2, 3, 4, 5, 7 and 8.
 */

#define NX GRID_SIZE_X
#define NY GRID_SIZE_Y
#define NZ GRID_SIZE_Z
#define NZH (GRID_SIZE_Z/2+1)

constant int RX[] = {RADICES_X};
constant int RY[] = {RADICES_Y};
constant int RZ[] = {RADICES_Z};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}

// Multiply by -i for a forward transform, or by i for an inverse one.
template<bool INV> inline float2 mulI(float2 a) {
    return INV ? float2(-a.y, a.x) : float2(a.y, -a.x);
}

// The position of frequency k after a decimation in frequency transform over the radices rad[0..ns-1].
template<int N> inline int posOf(int k, constant int* rad, int ns) {
    int p = 0, m = N;
    for (int s = 0; s < ns; s++) {
        int r = rad[s];
        m /= r;
        p += (k%r)*m;
        k /= r;
    }
    return p;
}

// The frequency at position p after a decimation in frequency transform.
template<int N> inline int freqOf(int p, constant int* rad, int ns) {
    int k = 0, w = 1, m = N;
    for (int s = 0; s < ns; s++) {
        int r = rad[s];
        m /= r;
        int q = p/m;
        p -= q*m;
        k += q*w;
        w *= r;
    }
    return k;
}

template<int R, bool INV> struct Dft;

template<bool INV> struct Dft<2, INV> {
    static inline void run(PRIVATE float2* x) {
        float2 a = x[0], b = x[1];
        x[0] = a+b;
        x[1] = a-b;
    }
};

template<bool INV> struct Dft<4, INV> {
    static inline void run(PRIVATE float2* x) {
        float2 s02 = x[0]+x[2], d02 = x[0]-x[2], s13 = x[1]+x[3], d13 = mulI<INV>(x[1]-x[3]);
        x[0] = s02+s13;
        x[2] = s02-s13;
        x[1] = d02+d13;
        x[3] = d02-d13;
    }
};

template<bool INV> struct Dft<8, INV> {
    static inline void run(PRIVATE float2* x) {
        const float h = M_SQRT1_2_F;
        float2 e[4] = {x[0], x[2], x[4], x[6]}, o[4] = {x[1], x[3], x[5], x[7]};
        Dft<4, INV>::run(e);
        Dft<4, INV>::run(o);
        // Multiply o[k] by the eighth root of unity to the power k.
        float2 o1 = o[1], o3 = o[3];
        o[1] = INV ? float2(h*(o1.x-o1.y), h*(o1.x+o1.y)) : float2(h*(o1.x+o1.y), h*(o1.y-o1.x));
        o[2] = mulI<INV>(o[2]);
        o[3] = INV ? float2(-h*(o3.x+o3.y), h*(o3.x-o3.y)) : float2(h*(o3.y-o3.x), -h*(o3.x+o3.y));
        for (int k = 0; k < 4; k++) {
            x[k] = e[k]+o[k];
            x[k+4] = e[k]-o[k];
        }
    }
};

// An odd radix pairs x_k with x_{R-k}, so each pair costs two real multiplies per root and component.  C and S
// hold cos(2 pi t/R) and sin(2 pi t/R).
#define ODD_DFT(R, C, S) \
template<bool INV> struct Dft<R, INV> { \
    static inline void run(PRIVATE float2* x) { \
        float2 y[R]; \
        float2 sum = x[0]; \
        for (int k = 1; k < R; k++) \
            sum += x[k]; \
        y[0] = sum; \
        for (int q = 1; q <= R/2; q++) { \
            float2 re = x[0], im = float2(0); \
            for (int k = 1; k <= R/2; k++) { \
                int t = (q*k)%R; \
                re += C[t]*(x[k]+x[R-k]); \
                im += S[t]*(x[k]-x[R-k]); \
            } \
            float2 rot = (INV ? float2(-im.y, im.x) : float2(im.y, -im.x)); \
            y[q] = re+rot; \
            y[R-q] = re-rot; \
        } \
        for (int k = 0; k < R; k++) \
            x[k] = y[k]; \
    } \
};
constant float C3[] = {ROOTS_C3};
constant float S3[] = {ROOTS_S3};
constant float C5[] = {ROOTS_C5};
constant float S5[] = {ROOTS_S5};
constant float C7[] = {ROOTS_C7};
constant float S7[] = {ROOTS_S7};
ODD_DFT(3, C3, S3)
ODD_DFT(5, C5, S5)
ODD_DFT(7, C7, S7)

/**
 * One radix R stage over sub-blocks of length L, on LINES lines of an N point transform.  Decimation in frequency
 * does the butterfly and then the twiddle.  Decimation in time is the exact reverse with conjugate roots: twiddle,
 * then butterfly.  Line l starts at l*LINE_STRIDE, or with LINE_Z at the position of z frequency l.  w holds the
 * N roots of unity of a forward transform.
 */
template<int R, int N, int L, int LINES, int LINE_STRIDE, bool LINE_Z, int STRIDE, bool INV, bool DIT>
inline void stage(threadgroup float2* buf, threadgroup const float2* w, int tid, int threads) {
    const int m = L/R;
    const int perLine = N/R;
    for (int i = tid; i < LINES*perLine; i += threads) {
        int line = i/perLine;
        int rem = i-line*perLine;
        int b = rem/m;
        int j = rem-b*m;
        int lineBase = (LINE_Z ? posOf<NZ>(line, RZ, NUM_STAGES_Z) : line*LINE_STRIDE);
        int base = lineBase+(b*L+j)*STRIDE;
        float2 x[R];
        for (int k = 0; k < R; k++)
            x[k] = buf[base+k*m*STRIDE];
        if (DIT && m > 1)
            for (int q = 1; q < R; q++) {
                float2 t = w[j*q*(N/L)];
                x[q] = cmul(x[q], INV ? float2(t.x, -t.y) : t);
            }
        Dft<R, INV>::run(x);
        if (!DIT && m > 1)
            for (int q = 1; q < R; q++) {
                float2 t = w[j*q*(N/L)];
                x[q] = cmul(x[q], INV ? float2(t.x, -t.y) : t);
            }
        for (int k = 0; k < R; k++)
            buf[base+k*m*STRIDE] = x[k];
    }
}

#define LOCAL_BARRIER threadgroup_barrier(mem_flags::mem_threadgroup);

inline void loadRoots(threadgroup float2* dst, GLOBAL const float2* src, int n, int tid, int threads) {
    for (int i = tid; i < n; i += threads)
        dst[i] = src[i];
}

KERNEL void planesForward(GLOBAL const float* RESTRICT in, GLOBAL float2* RESTRICT out,
        GLOBAL const float2* RESTRICT rootsY, GLOBAL const float2* RESTRICT rootsZ) {
    LOCAL float2 buf[NY*NZ];
    LOCAL float2 wY[NY], wZ[NZ];
    const int tid = LOCAL_ID, threads = LOCAL_SIZE, x = GROUP_ID;
    loadRoots(wY, rootsY, NY, tid, threads);
    loadRoots(wZ, rootsZ, NZ, tid, threads);
    GLOBAL const float* src = in+x*NY*NZ;
    for (int i = tid; i < NY*NZ; i += threads)
        buf[i] = float2(src[i], 0);
    LOCAL_BARRIER
#define Z_FORWARD(R, L) stage<R, NZ, L, NY, NZ, false, 1, false, false>(buf, wZ, tid, threads); LOCAL_BARRIER
    STAGES_Z(Z_FORWARD)
#define Y_FORWARD(R, L) stage<R, NY, L, NZH, 0, true, NZ, false, false>(buf, wY, tid, threads); LOCAL_BARRIER
    STAGES_Y(Y_FORWARD)
    GLOBAL float2* dst = out+x*NY*NZH;
    for (int i = tid; i < NY*NZH; i += threads) {
        int y = i/NZH;
        int kz = i-y*NZH;
        dst[i] = buf[posOf<NY>(y, RY, NUM_STAGES_Y)*NZ+posOf<NZ>(kz, RZ, NUM_STAGES_Z)];
    }
}

/**
 * The convolution follows reciprocalConvolution in pme.cc without LJPME.
 */
KERNEL void convolveLines(GLOBAL float2* RESTRICT grid, GLOBAL const float2* RESTRICT rootsX,
        GLOBAL const float* RESTRICT pmeBsplineModuliX, GLOBAL const float* RESTRICT pmeBsplineModuliY,
        GLOBAL const float* RESTRICT pmeBsplineModuliZ, float4 recipBoxVecX, float4 recipBoxVecY, float4 recipBoxVecZ,
        float recipExpFactor) {
    LOCAL float2 buf[NX*NZH];
    LOCAL float2 wX[NX];
    const int tid = LOCAL_ID, threads = LOCAL_SIZE, ky = GROUP_ID;
    loadRoots(wX, rootsX, NX, tid, threads);
    for (int i = tid; i < NX*NZH; i += threads) {
        int x = i/NZH;
        int kz = i-x*NZH;
        buf[i] = grid[(x*NY+ky)*NZH+kz];
    }
    LOCAL_BARRIER
#define X_FORWARD(R, L) stage<R, NX, L, NZH, 1, false, NZH, false, false>(buf, wX, tid, threads); LOCAL_BARRIER
    STAGES_X(X_FORWARD)
    const float recipScaleFactor = RECIP(M_PI_F)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
    int my = (ky < (NY+1)/2) ? ky : (ky-NY);
    float by = pmeBsplineModuliY[ky];
    for (int i = tid; i < NX*NZH; i += threads) {
        int p = i/NZH;
        int kz = i-p*NZH;
        int kx = freqOf<NX>(p, RX, NUM_STAGES_X);
        int mx = (kx < (NX+1)/2) ? kx : (kx-NX);
        int mz = (kz < (NZ+1)/2) ? kz : (kz-NZ);
        float mhx = mx*recipBoxVecX.x;
        float mhy = mx*recipBoxVecY.x+my*recipBoxVecY.y;
        float mhz = mx*recipBoxVecZ.x+my*recipBoxVecZ.y+mz*recipBoxVecZ.z;
        float bx = pmeBsplineModuliX[kx];
        float bz = pmeBsplineModuliZ[kz];
        float m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        float denom = m2*bx*by*bz;
        float eterm = recipScaleFactor*EXP(-recipExpFactor*m2)/denom;
        buf[i] = (kx != 0 || ky != 0 || kz != 0) ? buf[i]*eterm : float2(0);
    }
    LOCAL_BARRIER
#define X_INVERSE(R, L) stage<R, NX, L, NZH, 1, false, NZH, true, true>(buf, wX, tid, threads); LOCAL_BARRIER
    STAGES_X_REV(X_INVERSE)
    for (int i = tid; i < NX*NZH; i += threads) {
        int x = i/NZH;
        int kz = i-x*NZH;
        grid[(x*NY+ky)*NZH+kz] = buf[i];
    }
}

KERNEL void planesInverse(GLOBAL const float2* RESTRICT in, GLOBAL float* RESTRICT out,
        GLOBAL const float2* RESTRICT rootsY, GLOBAL const float2* RESTRICT rootsZ) {
    LOCAL float2 buf[NY*NZ];
    LOCAL float2 wY[NY], wZ[NZ];
    const int tid = LOCAL_ID, threads = LOCAL_SIZE, x = GROUP_ID;
    loadRoots(wY, rootsY, NY, tid, threads);
    loadRoots(wZ, rootsZ, NZ, tid, threads);
    GLOBAL const float2* src = in+x*NY*NZH;
    for (int i = tid; i < NY*NZH; i += threads) {
        int y = i/NZH;
        int kz = i-y*NZH;
        buf[y*NZ+kz] = src[i];
    }
    LOCAL_BARRIER
#define Y_INVERSE(R, L) stage<R, NY, L, NZH, 1, false, NZ, true, false>(buf, wY, tid, threads); LOCAL_BARRIER
    STAGES_Y(Y_INVERSE)
    // Each z line of a real plane is Hermitian: fill the upper half from the lower.
    for (int i = tid; i < NY*(NZ-NZH); i += threads) {
        int row = i/(NZ-NZH);
        int kz = NZH+i-row*(NZ-NZH);
        float2 v = buf[row*NZ+NZ-kz];
        buf[row*NZ+kz] = float2(v.x, -v.y);
    }
    LOCAL_BARRIER
#define Z_INVERSE(R, L) stage<R, NZ, L, NY, NZ, false, 1, true, false>(buf, wZ, tid, threads); LOCAL_BARRIER
    STAGES_Z(Z_INVERSE)
    GLOBAL float* dst = out+x*NY*NZ;
    for (int i = tid; i < NY*NZ; i += threads) {
        int y = i/NZ;
        int z = i-y*NZ;
        dst[i] = buf[posOf<NY>(y, RY, NUM_STAGES_Y)*NZ+posOf<NZ>(z, RZ, NUM_STAGES_Z)].x;
    }
}
