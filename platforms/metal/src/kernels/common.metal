/**
 * This file contains Metal definitions for the macros and functions needed for the
 * common compute framework.
 */

uint3 _mmGlobalId [[thread_position_in_grid]];
uint3 _mmGlobalSize [[threads_per_grid]];
uint3 _mmLocalId [[thread_position_in_threadgroup]];
uint3 _mmLocalSize [[threads_per_threadgroup]];
uint3 _mmGroupId [[threadgroup_position_in_grid]];
uint3 _mmNumGroups [[threadgroups_per_grid]];

#define KERNEL kernel
#define DEVICE
#define LOCAL threadgroup
#define LOCAL_ARG threadgroup
#define GLOBAL device
#define RESTRICT
#define LOCAL_ID _mmLocalId.x
#define LOCAL_SIZE _mmLocalSize.x
#define GLOBAL_ID _mmGlobalId.x
#define GLOBAL_SIZE _mmGlobalSize.x
#define GROUP_ID _mmGroupId.x
#define NUM_GROUPS _mmNumGroups.x
#define SYNC_THREADS threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
#define SYNC_WARPS simdgroup_barrier(mem_flags::mem_threadgroup)
// MSL 3.1 has no fence without a barrier.  Common kernels only use MEM_FENCE after atomic
// force accumulation, which needs no ordering, so it expands to nothing.
#define MEM_FENCE

typedef long mm_long;
typedef unsigned long mm_ulong;

inline int ATOMIC_ADD(device int* dest, int value) {
    return atomic_fetch_add_explicit((device atomic_int*) dest, value, memory_order_relaxed);
}

inline unsigned int ATOMIC_ADD(device unsigned int* dest, unsigned int value) {
    return atomic_fetch_add_explicit((device atomic_uint*) dest, value, memory_order_relaxed);
}

inline float ATOMIC_ADD(device float* dest, float value) {
    return atomic_fetch_add_explicit((device atomic_float*) dest, value, memory_order_relaxed);
}

/**
 * Apple GPUs before the M3 have no 64 bit atomic add, so add the two 32 bit words separately
 * and carry into the upper one.  The final sum is exact, but other threads may observe a
 * partially updated value, so the result is only meaningful once the kernel has finished.
 */
inline unsigned long ATOMIC_ADD(device unsigned long* dest, unsigned long value) {
    device atomic_uint* word = (device atomic_uint*) dest;
    unsigned int lower = (unsigned int) value;
    unsigned int upper = (unsigned int) (value >> 32);
    unsigned int oldLower = atomic_fetch_add_explicit(&word[0], lower, memory_order_relaxed);
    if (oldLower+lower < oldLower)
        upper++;
    if (upper != 0)
        atomic_fetch_add_explicit(&word[1], upper, memory_order_relaxed);
    return 0;
}

#define make_short2 short2
#define make_short3 short3
#define make_short4 short4
#define make_int2 int2
#define make_int3 int3
#define make_int4 int4
#define make_uint2 uint2
#define make_uint3 uint3
#define make_uint4 uint4
#define make_float2 float2
#define make_float3 float3
#define make_float4 float4

// These are functions rather than a macro so another precision mode can overload them.

inline float3 trimTo3(float4 v) {
    return v.xyz;
}

inline float3 trimTo3(float3 v) {
    return v;
}

inline float4 cross(float4 a, float4 b) {
    return float4(cross(a.xyz, b.xyz), 0.0f);
}

// CUDA has separate functions for single precision math.  To allow them to be called
// consistently, we define the "single precision" functions to just be synonyms for
// the standard ones.

#define sqrtf sqrt
#define rsqrtf rsqrt
#define expf exp
#define logf log
#define powf pow
#define cosf cos
#define sinf sin
#define tanf tan
#define acosf acos
#define asinf asin
#define atanf atan
#define atan2f atan2

/**
 * MSL has no erfc.  This is a degree 7 rational approximation with a maximum relative error
 * of 1.5e-6 on [0, 4], computed directly rather than as 1-erf(x) to avoid cancellation.
 */
inline float erfc(float x) {
    float t = fabs(x);
    float u = 1.0f/(1.0f+0.47f*t);
    float poly = ((((((0.0966678f*u-0.4607003f)*u+0.6896449f)*u-0.21611532f)*u+0.3931878f)*u+0.22740916f)*u+0.2701903f)*u-0.00028434425f;
    float result = (t > 9.0f ? 0.0f : poly*exp(-t*t));
    return (x < 0.0f ? 2.0f-result : result);
}

/**
 * MSL has no erf.  Near zero 1-erfc(x) loses relative precision, so use the Taylor series there.
 */
inline float erf(float x) {
    if (fabs(x) >= 0.5f)
        return 1.0f-erfc(x);
    float x2 = x*x;
    return 1.1283791670955126f*x*(1.0f+x2*(-1.0f/3.0f+x2*(1.0f/10.0f+x2*(-1.0f/42.0f+x2*(1.0f/216.0f+x2*(-1.0f/1320.0f))))));
}

inline long realToFixedPoint(real x) {
    return (long) (x*0x100000000);
}

// Common kernels use "thread" as a variable name, which is an address space qualifier in MSL.
// This must come after everything that uses the qualifier.

#define thread _mmThread
