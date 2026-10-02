/** Rebuild reorder bins in queue order after the box changes. */
KERNEL void updateBoxReorderBins(GLOBAL uchar* RESTRICT boxState, float invZigzagWidth,
        GLOBAL int* RESTRICT error, double minAllowedSize) {
    GLOBAL const double4* vectors = (GLOBAL const double4*) (boxState+208);
    if (vectors[0].x < minAllowedSize || vectors[1].y < minAllowedSize || vectors[2].z < minAllowedSize)
        error[0] = 1;
    float4 size = *((GLOBAL float4*) boxState);
    float width = (float) ((double) max(max(size.x, size.y), size.z)/255.0);
    *((GLOBAL float*) (boxState+192)) = (float) (1.0/(double) width);
    float3 bins = floor(size.xyz*invZigzagWidth)+1.0f;
    // Check before integer conversion or multiplication, including the extra z row used by zigzag.
    // A negative x count selects the bounded 24-bit Hilbert key for every molecule group.
    bool overflow = !all(isfinite(bins)) || any(bins >= 268435456.0f);
    if (!overflow) {
        ulong x = (ulong) bins.x, y = (ulong) bins.y, z = (ulong) bins.z;
        overflow = x*y >= 268435456ul || x*y*(z+1) >= 268435456ul;
    }
    *((GLOBAL int*) (boxState+196)) = overflow ? -1 : (int) bins.x;
    *((GLOBAL int*) (boxState+200)) = overflow ? -1 : (int) bins.y;
}

/** Energy terms that depend on the authoritative periodic volume. */
KERNEL void addBoxEnergy(GLOBAL const uchar* RESTRICT boxState, GLOBAL mixed* RESTRICT energyBuffer, double coefficient) {
    // The host volume and coefficient keep their double precision in single mode too.
    double volume = *((GLOBAL const double*) (boxState+304));
    energyBuffer[0] += (mixed) (coefficient/volume);
}

/** Scale the authoritative IEEE vectors and rebuild every force consumer's derived fields. */
KERNEL void scaleBox(GLOBAL uchar* RESTRICT boxState, GLOBAL const double* RESTRICT scaleFactor) {
    GLOBAL double4* vectors = (GLOBAL double4*) (boxState+208);
    double s = scaleFactor[0];
    double4 a = vectors[0]*s, b = vectors[1]*s, c = vectors[2]*s;
    vectors[0] = a; vectors[1] = b; vectors[2] = c;
    GLOBAL float4* fields = (GLOBAL float4*) boxState;
    fields[0] = make_float4((float) a.x, (float) b.y, (float) c.z, 0);
    fields[1] = make_float4(1.0f/(float) a.x, 1.0f/(float) b.y, 1.0f/(float) c.z, 0);
    fields[2] = make_float4((float) a.x, (float) a.y, (float) a.z, 0);
    fields[3] = make_float4((float) b.x, (float) b.y, (float) b.z, 0);
    fields[4] = make_float4((float) c.x, (float) c.y, (float) c.z, 0);
    double volume = a.x*b.y*c.z;
    double invVolume = 1.0/volume;
    fields[5] = make_float4((float) (b.y*c.z*invVolume), 0, 0, 0);
    fields[6] = make_float4((float) (-b.x*c.z*invVolume), (float) (a.x*c.z*invVolume), 0, 0);
    fields[7] = make_float4((float) ((b.x*c.y-b.y*c.x)*invVolume), (float) (-a.x*c.y*invVolume), (float) (a.x*b.y*invVolume), 0);
    fields[8] = make_float4((float) (a.x-(double) (float) a.x), (float) (a.y-(double) (float) a.y), (float) (a.z-(double) (float) a.z), 0);
    fields[9] = make_float4((float) (b.x-(double) (float) b.x), (float) (b.y-(double) (float) b.y), (float) (b.z-(double) (float) b.z), 0);
    fields[10] = make_float4((float) (c.x-(double) (float) c.x), (float) (c.y-(double) (float) c.y), (float) (c.z-(double) (float) c.z), 0);
    fields[11] = make_float4((float) (1.0/a.x), (float) (1.0/b.y), (float) (1.0/c.z), 0);
    *((GLOBAL float*) (boxState+204)) = (float) volume;
    *((GLOBAL double*) (boxState+304)) = volume;
}
