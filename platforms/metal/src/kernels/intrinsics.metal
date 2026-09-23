/**
 * This file contains the device function for using cross-lane operations (ballot and shuffle)
 */

// Metal: SIMD group size = tile size = 32, as on RDNA

#define SHFL(var, srcLane) simd_shuffle(var, srcLane)
#define BALLOT(var) (unsigned int) (ulong) simd_ballot(var)

template<class T>
static __inline__ __device__
T warpRotateLeft(const thread T& input) {
    return simd_shuffle_rotate_down(input, 1);
}
