/**
 * This file contains METAL definitions for the macros and functions needed for the
 * common compute framework.
 */

// MSL 3.1 lets kernels read built-in values from program scope, so they can have the CUDA names.
uint3 threadIdx [[thread_position_in_threadgroup]];
uint3 blockIdx [[threadgroup_position_in_grid]];
uint3 blockDim [[threads_per_threadgroup]];
uint3 gridDim [[threadgroups_per_grid]];

#define __global__ kernel
#define __device__
#define __shared__ threadgroup
#define __syncthreads() threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#define __threadfence() atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)
#define __launch_bounds__(x)
#define __clz clz
#define __popc popcount

#define KERNEL extern "C" __global__
#define DEVICE __device__
#define LOCAL __shared__
#define LOCAL_ARG threadgroup
#define GLOBAL device
#define PRIVATE thread
#define RESTRICT __restrict__
#define LOCAL_ID threadIdx.x
#define LOCAL_SIZE blockDim.x
#define GLOBAL_ID (blockIdx.x*blockDim.x+threadIdx.x)
#define GLOBAL_SIZE (blockDim.x*gridDim.x)
#define GROUP_ID blockIdx.x
#define NUM_GROUPS gridDim.x
#define SYNC_THREADS __syncthreads();
#define SYNC_WARPS simdgroup_barrier(mem_flags::mem_threadgroup);
// MSL has no fence without a barrier.  Common kernels only use MEM_FENCE after atomic
// force accumulation, which needs no ordering, so it expands to nothing.
#define MEM_FENCE
#define ATOMIC_ADD(dest, value) atomicAdd(dest, value)

typedef long mm_long;
typedef unsigned long mm_ulong;

inline int atomicAdd(device int* dest, int value) {
    return atomic_fetch_add_explicit((device atomic_int*) dest, value, memory_order_relaxed);
}

inline unsigned int atomicAdd(device unsigned int* dest, unsigned int value) {
    return atomic_fetch_add_explicit((device atomic_uint*) dest, value, memory_order_relaxed);
}

inline float atomicAdd(device float* dest, float value) {
    return atomic_fetch_add_explicit((device atomic_float*) dest, value, memory_order_relaxed);
}

/**
 * Apple GPUs before the M3 have no 64 bit atomic add, so add the two 32 bit words separately
 * and carry into the upper one.  The final sum is exact, but other threads may observe a
 * partially updated value, so the result is only meaningful once the kernel has finished.
 */
inline unsigned long atomicAdd(device unsigned long* dest, unsigned long value) {
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

// MSL constructs vectors with constructors, not make_ functions.

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

// CUDA has separate functions for single precision math.  To allow them to be called
// consistently, we define the "single precision" functions to just be synonyms for
// the standard ones.

#define sqrtf sqrt
#define rsqrtf rsqrt
#define expf exp
#define logf log
#define cosf cos
#define sinf sin
#define tanf tan
#define acosf acos
#define asinf asin
#define atanf atan
#define atan2f atan2
#define powf pow
#define erff erf
#define erfcf erfc
#define fmaf fma
#define fabsf fabs
#define truncf trunc

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

__device__ inline long realToFixedPoint(float x) {
    // Faster way to calculate static_cast<long long>(x * 0x100000000) with exactly the same
    // results but less instructions.
    float integral = truncf(x);
    float fractional = (x - integral) * 0x100000000;
    unsigned int integral_u32 = static_cast<int>(integral);
    unsigned int fractional_u32 = static_cast<unsigned int>(fabsf(fractional));
    // A negative real number (with non-zero fractional) needs rounding-down x for integral and
    // changing fractional's sign. However, -1 is used as a threshold instead of 0 because, when
    // fractional is in (-1; 0], fractional_u32 is 0 and the number is considered an integer.
    bool isNegReal = fractional <= -1.0f;
    return (static_cast<unsigned long>(isNegReal ? integral_u32 - 1 : integral_u32) << 32) |
            static_cast<unsigned long>(isNegReal ? 0 - fractional_u32 : fractional_u32);
}
