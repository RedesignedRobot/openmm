/**
 * Undo the whole-box shifts a reorder applied to wrapped molecules, so each position is the one getPositions()
 * reports and no longer depends on the box: x - ox*a - oy*b - oz*c, with the box before it changes.  This is what the
 * common setPeriodicBoxVectors() does on the host through a download and an upload of every position.
 */
KERNEL void unwrapPositions(int numAtoms, GLOBAL real4* RESTRICT posq, GLOBAL real4* RESTRICT posqCorrection,
        GLOBAL const int4* RESTRICT cellOffsets, GLOBAL const mixed4* RESTRICT box) {
    const mixed4 a = box[0], b = box[1], c = box[2];
    for (int atom = GLOBAL_ID; atom < numAtoms; atom += GLOBAL_SIZE) {
        const int4 offset = cellOffsets[atom];
        if (offset.x == 0 && offset.y == 0 && offset.z == 0)
            continue;
        const mixed ox = (mixed) offset.x, oy = (mixed) offset.y, oz = (mixed) offset.z;
        const mixed shiftX = ox*a.x+oy*b.x+oz*c.x;
        const mixed shiftY = ox*a.y+oy*b.y+oz*c.y;
        const mixed shiftZ = ox*a.z+oy*b.z+oz*c.z;
        const real4 pos1 = posq[atom];
#ifdef USE_MIXED_PRECISION
        const real4 pos2 = posqCorrection[atom];
        const mixed x = pos1.x+(mixed) pos2.x-shiftX;
        const mixed y = pos1.y+(mixed) pos2.y-shiftY;
        const mixed z = pos1.z+(mixed) pos2.z-shiftZ;
        posq[atom] = make_real4((real) x, (real) y, (real) z, pos1.w);
        posqCorrection[atom] = make_real4(x-(real) x, y-(real) y, z-(real) z, 0);
#else
        posq[atom] = make_real4(pos1.x-shiftX, pos1.y-shiftY, pos1.z-shiftZ, pos1.w);
#endif
    }
}
