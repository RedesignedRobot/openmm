/**
 * Kernels for MetalAtomReorder, which reorders atoms on the device from current positions the way
 * ComputeContext::findAtomOrder() does on the host.  computeMoleculeKeys gives each molecule a key (group,
 * bin on a Hilbert or zigzag curve, molecule index), MetalSort sorts the keys, and gatherAtoms and storeAtoms
 * move every atom.  The molecules of a group are identical, so the molecule sorted into rank r of its group
 * takes the atom slots of the group's r-th molecule.  The molecule index in the key breaks ties the way the
 * host's sort of (bin, index) pairs does, and makes the order deterministic.
 *
 * A permutation cannot be applied in place, so gatherAtoms copies each atom's shifted position, its
 * correction, its raw velocity words, its atom index and its cell offsets into scratch arrays, and storeAtoms
 * writes them back.  Velocities are copied as stored (float pairs in mixed), so they come through bit for
 * bit.  Padding atoms get zero positions and velocities and keep their index and offsets, as in
 * ComputeContext::reorderAtomsImpl().
 */
#ifdef USE_MIXED_PRECISION
#define VEL_WORDS 2
#define WORDS 4
#else
#define VEL_WORDS 1
#define WORDS 2
#endif
#define GROUP_SHIFT 52
#define BIN_SHIFT 24
#define BIN_MASK 0xFFFFFFF
#define MOLECULE_MASK 0xFFFFFF

/**
 * The index of a point on a 256x256x256 Hilbert curve, equal to hilbert_c2i(3, 8, {x, y, z}) in
 * libraries/hilbert (checked on all 2^24 points).
 */
uint hilbertIndex(uint x, uint y, uint z) {
    uint coords = 0;
    for (int i = 7; i >= 0; i--)
        coords = (coords << 3) | (((z >> i) & 1) << 2) | (((y >> i) & 1) << 1) | ((x >> i) & 1);
    coords ^= coords >> 3;
    uint index = 0, rotation = 0, flipBit = 0;
    for (int b = 21; b >= 0; b -= 3) {
        uint bits = ((coords >> b) & 7) ^ flipBit;
        bits = ((bits >> rotation) | (bits << (3-rotation))) & 7;
        index = (index << 3) | bits;
        flipBit = 1u << rotation;
        bits &= (0u-bits) & 3;
        while (bits != 0) {
            bits >>= 1;
            rotation++;
        }
        if (++rotation >= 3)
            rotation -= 3;
    }
    index ^= 0x249249u >> 1;
    for (int d = 1; d < 24; d *= 2)
        index ^= index >> d;
    return index;
}

/**
 * Wraps molecule m's center into the box, records the cell shift and writes its sort key.  molecules[m] is
 * {first atom offset, group*2 + (1 if the group uses the Hilbert curve), start of the group's template in
 * templateAtoms, template size}.
 */
void writeMoleculeKey(int m, int4 mol, float cx, float cy, float cz, float4 boxX, float4 boxY, float4 boxZ, float4 invBoxSize,
        float invHilbertWidth, float invZigzagWidth, int zigzagXBins, int zigzagYBins, GLOBAL ulong* RESTRICT keys,
        GLOBAL int4* RESTRICT moleculeCells, GLOBAL int* RESTRICT nanFlag) {
    if (cx != cx)
        nanFlag[0] = 1;
    int zcell = (int) floor(cz*invBoxSize.z);
    cx -= zcell*boxZ.x;
    cy -= zcell*boxZ.y;
    cz -= zcell*boxZ.z;
    int ycell = (int) floor(cy*invBoxSize.y);
    cx -= ycell*boxY.x;
    cy -= ycell*boxY.y;
    int xcell = (int) floor(cx*invBoxSize.x);
    cx -= xcell*boxX.x;
    moleculeCells[m] = make_int4(xcell, ycell, zcell, 0);
    uint bin;
    if ((mol.y & 1) || zigzagXBins < 0) {
        uint x = (uint) clamp((int) (cx*invHilbertWidth), 0, 255);
        uint y = (uint) clamp((int) (cy*invHilbertWidth), 0, 255);
        uint z = (uint) clamp((int) (cz*invHilbertWidth), 0, 255);
        bin = hilbertIndex(x, y, z);
    }
    else {
        int x = (int) (cx*invZigzagWidth);
        int y = (int) (cy*invZigzagWidth);
        int z = (int) (cz*invZigzagWidth);
        int yodd = y&1;
        int zodd = z&1;
        int zigzag = z*zigzagXBins*zigzagYBins;
        zigzag += (zodd ? zigzagYBins-y : y)*zigzagXBins;
        zigzag += (yodd ? zigzagXBins-x : x);
        bin = (uint) zigzag;
    }
    keys[m] = (((ulong) (mol.y >> 1)) << GROUP_SHIFT) | (((ulong) (bin & BIN_MASK)) << BIN_SHIFT) | (ulong) m;
}

/**
 * Keys for molecules of at most LARGE_MOLECULE atoms, one thread each, summing in template order as the host does.
 */
KERNEL void computeMoleculeKeys(int numMolecules, GLOBAL const int4* RESTRICT molecules, GLOBAL const int* RESTRICT templateAtoms,
        GLOBAL const float4* RESTRICT posq, float4 boxX, float4 boxY, float4 boxZ, float4 invBoxSize, float invHilbertWidth,
        float invZigzagWidth, int zigzagXBins, int zigzagYBins, GLOBAL ulong* RESTRICT keys, GLOBAL int4* RESTRICT moleculeCells,
        GLOBAL int* RESTRICT nanFlag) {
    for (int m = GLOBAL_ID; m < numMolecules; m += GLOBAL_SIZE) {
        int4 mol = molecules[m];
        if (mol.w > LARGE_MOLECULE)
            continue;
        float cx = 0, cy = 0, cz = 0;
        for (int j = 0; j < mol.w; j++) {
            float4 pos = posq[mol.x+templateAtoms[mol.z+j]];
            cx += pos.x;
            cy += pos.y;
            cz += pos.z;
        }
        float invNumAtoms = 1.0f/mol.w;
        writeMoleculeKey(m, mol, cx*invNumAtoms, cy*invNumAtoms, cz*invNumAtoms, boxX, boxY, boxZ, invBoxSize, invHilbertWidth,
                invZigzagWidth, zigzagXBins, zigzagYBins, keys, moleculeCells, nanFlag);
    }
}

/**
 * Keys for molecules of more than LARGE_MOLECULE atoms (proteins, lipids), one threadgroup of LARGE_BLOCK_SIZE
 * threads each.  A single thread summing a 7,000 atom protein took 0.8 ms on apoa1.  The fixed reduction tree
 * keeps the result deterministic.
 */
KERNEL void computeLargeMoleculeKeys(int numLarge, GLOBAL const int* RESTRICT largeMolecules, GLOBAL const int4* RESTRICT molecules,
        GLOBAL const int* RESTRICT templateAtoms, GLOBAL const float4* RESTRICT posq, float4 boxX, float4 boxY, float4 boxZ,
        float4 invBoxSize, float invHilbertWidth, float invZigzagWidth, int zigzagXBins, int zigzagYBins, GLOBAL ulong* RESTRICT keys,
        GLOBAL int4* RESTRICT moleculeCells, GLOBAL int* RESTRICT nanFlag) {
    LOCAL float sumX[LARGE_BLOCK_SIZE];
    LOCAL float sumY[LARGE_BLOCK_SIZE];
    LOCAL float sumZ[LARGE_BLOCK_SIZE];
    for (int b = GROUP_ID; b < numLarge; b += NUM_GROUPS) {
        int m = largeMolecules[b];
        int4 mol = molecules[m];
        float cx = 0, cy = 0, cz = 0;
        for (int j = LOCAL_ID; j < mol.w; j += LARGE_BLOCK_SIZE) {
            float4 pos = posq[mol.x+templateAtoms[mol.z+j]];
            cx += pos.x;
            cy += pos.y;
            cz += pos.z;
        }
        sumX[LOCAL_ID] = cx;
        sumY[LOCAL_ID] = cy;
        sumZ[LOCAL_ID] = cz;
        SYNC_THREADS;
        for (int step = LARGE_BLOCK_SIZE/2; step > 0; step /= 2) {
            if (LOCAL_ID < step) {
                sumX[LOCAL_ID] += sumX[LOCAL_ID+step];
                sumY[LOCAL_ID] += sumY[LOCAL_ID+step];
                sumZ[LOCAL_ID] += sumZ[LOCAL_ID+step];
            }
            SYNC_THREADS;
        }
        if (LOCAL_ID == 0) {
            float invNumAtoms = 1.0f/mol.w;
            writeMoleculeKey(m, mol, sumX[0]*invNumAtoms, sumY[0]*invNumAtoms, sumZ[0]*invNumAtoms, boxX, boxY, boxZ, invBoxSize,
                    invHilbertWidth, invZigzagWidth, zigzagXBins, zigzagYBins, keys, moleculeCells, nanFlag);
        }
        SYNC_THREADS;
    }
}

/**
 * slotMolecule[i] is the molecule whose atoms occupy slot i.  sortedKeys[m] names the molecule that moves into
 * the slots of molecule m.
 */
KERNEL void gatherAtoms(int numAtoms, int paddedNumAtoms, GLOBAL const int* RESTRICT slotMolecule, GLOBAL const int4* RESTRICT molecules,
        GLOBAL const ulong* RESTRICT sortedKeys, GLOBAL const int4* RESTRICT moleculeCells, GLOBAL const float4* RESTRICT posq,
#ifdef USE_MIXED_PRECISION
        GLOBAL const float4* RESTRICT posqCorrection,
#endif
        GLOBAL const mixed4s* RESTRICT velm, GLOBAL const int* RESTRICT atomIndex, GLOBAL const int4* RESTRICT cellOffsets,
        GLOBAL float4* RESTRICT scratch, GLOBAL int* RESTRICT scratchIndex, GLOBAL int4* RESTRICT scratchOffsets,
        float4 boxXHi, float4 boxYHi, float4 boxZHi, float4 boxXLo, float4 boxYLo, float4 boxZLo) {
    GLOBAL const float4* velocityWords = (GLOBAL const float4*) velm;
    for (int i = GLOBAL_ID; i < paddedNumAtoms; i += GLOBAL_SIZE) {
        if (i >= numAtoms) {
            for (int k = 0; k < WORDS; k++)
                scratch[WORDS*i+k] = make_float4(0);
            continue;
        }
        int slotMol = slotMolecule[i];
        int sourceMol = (int) (sortedKeys[slotMol] & MOLECULE_MASK);
        int old = i-molecules[slotMol].x+molecules[sourceMol].x;
        int4 cell = moleculeCells[sourceMol];
        float4 pos = posq[old];
#ifdef USE_MIXED_PRECISION
        float4 corr = posqCorrection[old];
#endif
        if (cell.x != 0 || cell.y != 0 || cell.z != 0) {
            float cx = (float) cell.x;
            float cy = (float) cell.y;
            float cz = (float) cell.z;
#ifdef USE_MIXED_PRECISION
            // Shift in df64 and split again, as the host path does in double.
            df64 x = df64(pos.x, corr.x) - (df64(boxXHi.x, boxXLo.x)*cx + df64(boxYHi.x, boxYLo.x)*cy + df64(boxZHi.x, boxZLo.x)*cz);
            df64 y = df64(pos.y, corr.y) - (df64(boxYHi.y, boxYLo.y)*cy + df64(boxZHi.y, boxZLo.y)*cz);
            df64 z = df64(pos.z, corr.z) - df64(boxZHi.z, boxZLo.z)*cz;
            pos.x = x.hi;
            pos.y = y.hi;
            pos.z = z.hi;
            corr.x = x.lo;
            corr.y = y.lo;
            corr.z = z.lo;
#else
            pos.x -= boxXHi.x*cx + boxYHi.x*cy + boxZHi.x*cz;
            pos.y -= boxYHi.y*cy + boxZHi.y*cz;
            pos.z -= boxZHi.z*cz;
#endif
        }
        scratch[WORDS*i] = pos;
#ifdef USE_MIXED_PRECISION
        scratch[WORDS*i+1] = corr;
#endif
        for (int k = 0; k < VEL_WORDS; k++)
            scratch[WORDS*i+WORDS-VEL_WORDS+k] = velocityWords[VEL_WORDS*old+k];
        int4 offset = cellOffsets[old];
        scratchIndex[i] = atomIndex[old];
        scratchOffsets[i] = make_int4(offset.x-cell.x, offset.y-cell.y, offset.z-cell.z, offset.w);
    }
}

KERNEL void storeAtoms(int numAtoms, int paddedNumAtoms, GLOBAL const float4* RESTRICT scratch, GLOBAL const int* RESTRICT scratchIndex,
        GLOBAL const int4* RESTRICT scratchOffsets, GLOBAL float4* RESTRICT posq,
#ifdef USE_MIXED_PRECISION
        GLOBAL float4* RESTRICT posqCorrection,
#endif
        GLOBAL mixed4s* RESTRICT velm, GLOBAL int* RESTRICT atomIndex, GLOBAL int4* RESTRICT cellOffsets) {
    GLOBAL float4* velocityWords = (GLOBAL float4*) velm;
    for (int i = GLOBAL_ID; i < paddedNumAtoms; i += GLOBAL_SIZE) {
        posq[i] = scratch[WORDS*i];
#ifdef USE_MIXED_PRECISION
        posqCorrection[i] = scratch[WORDS*i+1];
#endif
        for (int k = 0; k < VEL_WORDS; k++)
            velocityWords[VEL_WORDS*i+k] = scratch[WORDS*i+WORDS-VEL_WORDS+k];
        if (i < numAtoms) {
            atomIndex[i] = scratchIndex[i];
            cellOffsets[i] = scratchOffsets[i];
        }
    }
}

#ifdef USE_DEVICE_BOX
/** Match the host barostat's rigid translation of molecule centers, preserving mixed corrections. */
KERNEL void scaleMoleculeCenters(int numMolecules, GLOBAL const int4* RESTRICT molecules,
        GLOBAL const int* RESTRICT templateAtoms, GLOBAL float4* RESTRICT posq,
        GLOBAL const double* RESTRICT scaleFactor) {
    float scale = (float) scaleFactor[0];
    for (int m = GLOBAL_ID; m < numMolecules; m += GLOBAL_SIZE) {
        int4 mol = molecules[m];
        float3 center = make_float3(0, 0, 0);
        for (int j = 0; j < mol.w; j++)
            center += posq[mol.x+templateAtoms[mol.z+j]].xyz;
        center *= RECIP((float) mol.w);
        float3 delta = center*(scale-1.0f);
        for (int j = 0; j < mol.w; j++) {
            int atom = mol.x+templateAtoms[mol.z+j];
            float4 pos = posq[atom];
            pos.xyz += delta;
            posq[atom] = pos;
        }
    }
}
#endif
