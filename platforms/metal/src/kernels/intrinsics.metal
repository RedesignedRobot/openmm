/**
 * This file contains the device function for using cross-lane operations (ballot and shuffle)
 */

constant int warpSize = 32;

static inline long simd_shuffle(long var, ushort srcLane) {
    return as_type<long>(simd_shuffle(as_type<uint2>(var), srcLane));
}

static inline long simd_shuffle_rotate_down(long var, ushort delta) {
    return as_type<long>(simd_shuffle_rotate_down(as_type<uint2>(var), delta));
}

#define SHFL(var, srcLane) simd_shuffle(var, srcLane)
#define BALLOT(var) (unsigned int) (ulong) simd_ballot(var)
#define __shfl(var, srcLane) SHFL(var, srcLane)
#define __shfl_down(var, delta, width) simd_shuffle_down(var, delta)
#define __ballot(var) BALLOT(var)

template<class T>
static __inline__ __device__
T warpRotateLeft(const thread T& input) {
    return simd_shuffle_rotate_down(input, 1);
}
