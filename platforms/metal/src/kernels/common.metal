/**
 * This file contains Metal definitions for the macros and functions needed for the
 * common compute framework.
 */

#include <metal_stdlib>

using namespace metal;

uint LOCAL_ID [[thread_position_in_threadgroup]];
uint LOCAL_SIZE [[threads_per_threadgroup]];
uint GLOBAL_ID [[thread_position_in_grid]];
uint GLOBAL_SIZE [[threads_per_grid]];
uint GROUP_ID [[threadgroup_position_in_grid]];
uint NUM_GROUPS [[threadgroups_per_grid]];

#define KERNEL kernel
#define DEVICE
#define LOCAL threadgroup
#define LOCAL_ARG threadgroup
#define GLOBAL device
#define PRIVATE thread
#define RESTRICT
#define SYNC_THREADS threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
#define SYNC_WARPS simdgroup_barrier(mem_flags::mem_threadgroup);
#define MEM_FENCE
#define BALLOT(var) ((int) (uint64_t) simd_ballot(var))

template <class T>
inline T SHFL(T var, ushort srcLane) {
    return simd_shuffle(var, srcLane);
}

template <>
inline long SHFL(long var, ushort srcLane) {
    uint low = SHFL((uint) (var & 0xFFFFFFFF), srcLane);
    int high = SHFL((int) (var >> 32), srcLane);
    return (((long) high) << 32) + (long) low;
}

inline int ATOMIC_ADD(device int* dest, int value) {
    return atomic_fetch_add_explicit((device atomic_int*) dest, value, memory_order_relaxed);
}

inline unsigned int ATOMIC_ADD(device unsigned int* dest, unsigned int value) {
    return atomic_fetch_add_explicit((device atomic_uint*) dest, value, memory_order_relaxed);
}

inline float ATOMIC_ADD(device float* dest, float value) {
    return atomic_fetch_add_explicit((device atomic_float*) dest, value, memory_order_relaxed);
}

inline unsigned long ATOMIC_ADD(device unsigned long* dest, unsigned long value) {
    device unsigned int* word = (device unsigned int*) dest;
    int lowIndex = 0;
    unsigned int lower = value;
    unsigned int upper = value >> 32;
    unsigned int result = ATOMIC_ADD(&word[lowIndex], lower);
    int carry = (lower + (unsigned long) result >= 0x100000000 ? 1 : 0);
    upper += carry;
    if (upper != 0)
        ATOMIC_ADD(&word[1-lowIndex], upper);
    return 0;
}

typedef long mm_long;
typedef unsigned long mm_ulong;

#define make_short2(x...) short2(x)
#define make_short3(x...) short3(x)
#define make_short4(x...) short4(x)
#define make_int2(x...) int2(x)
#define make_int3(x...) int3(x)
#define make_int4(x...) int4(x)
#define make_float2(x...) float2(x)
#define make_float3(x...) float3(x)
#define make_float4(x...) float4(x)

#ifndef USE_MIXED_PRECISION
#define trimTo3(v) (v).xyz
#endif

// Metal does not support the standard names for single precision math functions.

#define sqrtf(x) sqrt(x)
#define rsqrtf(x) rsqrt(x)
#define expf(x) exp(x)
#define logf(x) log(x)
#define powf(x) pow(x)
#define cosf(x) cos(x)
#define sinf(x) sin(x)
#define tanf(x) tan(x)
#define acosf(x) acos(x)
#define asinf(x) asin(x)
#define atanf(x) atan(x)
#define atan2f(x, y) atan2(x, y)

inline float4 cross(float4 a, float4 b) {
    return float4(cross(a.xyz, b.xyz), 0.0);
}

inline long realToFixedPoint(real x) {
    return static_cast<long>(clamp((real) -0x80000000, (real) 0x7FFFFFFF, x)*0x100000000);
}

inline float erfc(float x) {
    // Metal has no erfc.  This one keeps the relative error small for every x (under 5 ulp in
    // tests against a double precision erfc), which matters where erfc(x) is small and a caller
    // subtracts nearly equal terms, such as the LJPME reciprocal space convolution.  With
    // a = |x| and q = (a-2)/(a+2), which maps [0, inf) onto [-1, 1), (1+2a)*exp(a*a)*erfc(a) is
    // approximated as 1+p(q), where p is a degree 12 polynomial fitted at Chebyshev nodes.  The
    // division by 1+2a is refined by one Newton step, and exp(-a*a) is corrected for the
    // rounding error of a*a.

    float a = fabs(x);
    float q = (a-2.0f)/(a+2.0f);
    float p = -2.025598951e-05f;
    p = fma(p, q, 6.336945808e-05f);
    p = fma(p, q, 2.215715504e-04f);
    p = fma(p, q, -3.579204495e-04f);
    p = fma(p, q, -1.464070869e-03f);
    p = fma(p, q, 1.217410550e-03f);
    p = fma(p, q, 8.724607527e-03f);
    p = fma(p, q, -8.018963039e-03f);
    p = fma(p, q, -5.422026664e-02f);
    p = fma(p, q, 1.640494615e-01f);
    p = fma(p, q, -1.660303771e-01f);
    p = fma(p, q, -9.276375920e-02f);
    p = fma(p, q, 2.769783735e-01f);
    float r = 1.0f/fma(2.0f, a, 1.0f);
    float g = fma(p, r, r);
    float e = fma(fma(g, -a, 0.5f), 2.0f, p-g);
    g = fma(e, r, g);
    float s = a*a;
    float t = g*exp(-s);
    float result = fma(t, fma(-a, a, s), t);
    result = (a > 10.0546875f ? 0.0f : result);
    return (x < 0.0f ? 2.0f-result : result);
}

inline float erf(float x) {
    // 1-erfc(x) has an absolute error near 1.5e-7, which is far too large when x is small.  PME exclusions
    // subtract nearly equal terms at small r, which happens with Drude particles, so use a series there.

    if (fabs(x) < 0.5f) {
        float x2 = x*x;
        return 1.1283791671f*x*(1.0f+x2*(-1.0f/3.0f+x2*(1.0f/10.0f+x2*(-1.0f/42.0f+x2*(1.0f/216.0f+x2*(-1.0f/1320.0f+x2*(1.0f/9360.0f)))))));
    }
    return 1.0f-erfc(x);
}

inline float remainder(float x, float y) {
    // Metal has fmod but not the IEEE remainder that CUDA and OpenCL provide.

    return x-y*rint(x/y);
}