/**
 * This file defines vector operations to simplify code elsewhere.
 */

// Cross product

inline __device__ float4 cross(float4 a, float4 b) {
    return make_float4(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x, 0.0f);
}

// Strip off the fourth component of a vector.

inline __device__ short3 trimTo3(short4 v) {
    return make_short3(v.x, v.y, v.z);
}

inline __device__ int3 trimTo3(int4 v) {
    return make_int3(v.x, v.y, v.z);
}

inline __device__ float3 trimTo3(float4 v) {
    return make_float3(v.x, v.y, v.z);
}
