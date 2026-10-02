KERNEL void findAtomGridIndex(GLOBAL const real4* RESTRICT posq, GLOBAL int2* RESTRICT pmeAtomGridIndex,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ
    ) {
    // Compute the index of the grid point each atom is associated with.

    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real4 pos = posq[atom];
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);
        pmeAtomGridIndex[atom] = make_int2(atom, gridIndex.x*GRID_SIZE_Y*GRID_SIZE_Z+gridIndex.y*GRID_SIZE_Z+gridIndex.z);
    }
}

KERNEL void gridSpreadCharge(GLOBAL const real4* RESTRICT posq,
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        GLOBAL mm_ulong* RESTRICT pmeGrid,
#else
        GLOBAL real* RESTRICT pmeGrid,
#endif
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, GLOBAL const int2* RESTRICT pmeAtomGridIndex,
#ifdef CHARGE_FROM_SIGEPS
        GLOBAL const float2* RESTRICT sigmaEpsilon
#else
        GLOBAL const real* RESTRICT charges
#endif
        ) {
#if defined(USE_METAL) && !defined(USE_FIXED_POINT_CHARGE_SPREADING)
    // Metal with float atomics (Apple9).  Atoms arrive in a spatially coherent order, so a few
    // consecutive atoms share most of their stencil points, and the device atomics, not the math,
    // bound the one-atom-per-thread kernel below.  Each SIMD group computes the splines of 32
    // atoms, one per lane, then sums spans of SPREAD_SPAN consecutive atoms into a private slab of
    // threadgroup memory covering the span's stencil bounding box, and adds each nonzero slab point
    // to the grid with one atomic.  Atom data stays in the owning lane's registers and is broadcast
    // one atom at a time; lane l < 25 adds the atom's footprint column (dy, dz) = (l/5, l%5), so no
    // two lanes write one slab address in an atom step and the slab needs no atomics.  A span whose
    // box exceeds SPREAD_SLAB points halves down to 4 atoms, and a span that still doesn't fit
    // spreads its atoms directly, one atom at a time with the same lane columns.

    LOCAL real slabs[SPREAD_GROUPS][SPREAD_SLAB];
    const int lane = LOCAL_ID%32;
    const int group = LOCAL_ID/32;
    const int groups = min((int) (LOCAL_SIZE/32), SPREAD_GROUPS);
    if (group >= groups)
        return;
    LOCAL real* slab = slabs[group];
    for (int v = lane; v < SPREAD_SLAB; v += 32)
        slab[v] = 0;
    SYNC_WARPS; // The first atom can add to entries initialized by another lane.
    const real scale = RECIP((real) (PME_ORDER-1));
    const int dy = lane/PME_ORDER;
    const int dz = lane-PME_ORDER*dy;
    const int numBlocks = (NUM_ATOMS+31)/32;
    for (int block = GROUP_ID*groups+group; block < numBlocks; block += NUM_GROUPS*groups) {
        const int first = block*32;
        const int count = min(32, NUM_ATOMS-first);
        real3 data[PME_ORDER];
        int3 gridIndex = make_int3(0, 0, 0);
        if (lane < count) {
            int atom = pmeAtomGridIndex[first+lane].x;
            real4 pos = posq[atom];
#ifdef CHARGE_FROM_SIGEPS
            const float2 sigEps = sigmaEpsilon[atom];
            const real charge = 8*sigEps.x*sigEps.x*sigEps.x*sigEps.y;
#else
            const real charge = (CHARGE)*EPSILON_FACTOR;
#endif
            APPLY_PERIODIC_TO_POS(pos)
            real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                                 pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                                 pos.z*recipBoxVecZ.z);
            t.x = (t.x-floor(t.x))*GRID_SIZE_X;
            t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
            t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
            gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                  ((int) t.y) % GRID_SIZE_Y,
                                  ((int) t.z) % GRID_SIZE_Z);
            real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
            data[PME_ORDER-1] = make_real3(0);
            data[1] = dr;
            data[0] = make_real3(1)-dr;
            for (int j = 3; j < PME_ORDER; j++) {
                real div = RECIP((real) (j-1));
                data[j-1] = div*dr*data[j-2];
                for (int k = 1; k < (j-1); k++)
                    data[j-k-1] = div*((dr+make_real3(k))*data[j-k-2] + (make_real3(j-k)-dr)*data[j-k-1]);
                data[0] = div*(make_real3(1)-dr)*data[0];
            }
            data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
            for (int j = 1; j < (PME_ORDER-1); j++)
                data[PME_ORDER-j-1] = scale*((dr+make_real3(j))*data[PME_ORDER-j-2] + (make_real3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
            data[0] = scale*(make_real3(1)-dr)*data[0];
            for (int k = 0; k < PME_ORDER; k++)
                data[k].z *= charge;
        }
        int start = 0, span = SPREAD_SPAN;
        while (start < count) {
            const int end = min(count, start+span);
            const bool inSpan = (lane >= start && lane < end);

            // The span's bounding box, in grid points relative to its first atom, wrapped to the
            // nearest periodic image.  lo <= 0, so the box origin is in [-GRID_SIZE/2, GRID_SIZE),
            // and with every extent at most the grid size one wrap maps a slab point onto the grid.

            const int3 origin = simd_broadcast(gridIndex, (ushort) start);
            int3 d = gridIndex-origin;
            d.x += (d.x > GRID_SIZE_X/2 ? -GRID_SIZE_X : (d.x < -GRID_SIZE_X/2 ? GRID_SIZE_X : 0));
            d.y += (d.y > GRID_SIZE_Y/2 ? -GRID_SIZE_Y : (d.y < -GRID_SIZE_Y/2 ? GRID_SIZE_Y : 0));
            d.z += (d.z > GRID_SIZE_Z/2 ? -GRID_SIZE_Z : (d.z < -GRID_SIZE_Z/2 ? GRID_SIZE_Z : 0));
            const int3 lo = make_int3(simd_min(inSpan ? d.x : GRID_SIZE_X), simd_min(inSpan ? d.y : GRID_SIZE_Y), simd_min(inSpan ? d.z : GRID_SIZE_Z));
            const int3 hi = make_int3(simd_max(inSpan ? d.x : -GRID_SIZE_X), simd_max(inSpan ? d.y : -GRID_SIZE_Y), simd_max(inSpan ? d.z : -GRID_SIZE_Z));
            const int3 extent = hi-lo+make_int3(PME_ORDER);
            const int volume = extent.x*extent.y*extent.z;
            const bool fits = (volume <= SPREAD_SLAB && extent.x <= GRID_SIZE_X && extent.y <= GRID_SIZE_Y && extent.z <= GRID_SIZE_Z);
            if (!fits && span > 4) {
                span /= 2;
                continue;
            }
            const int strideX = extent.y*extent.z;
            const int3 corner = d-lo;
            const int packed = corner.x*strideX+corner.y*extent.z+corner.z;
            const int laneOffset = dy*extent.z+dz;
            for (int a = start; a < end; a++) {
                const ushort src = (ushort) a;
                real wy = 0, wz = 0;
                for (int k = 0; k < PME_ORDER; k++) {
                    real y = simd_broadcast(data[k].y, src);
                    real z = simd_broadcast(data[k].z, src);
                    wy = (k == dy ? y : wy);
                    wz = (k == dz ? z : wz);
                }
                const real w = wy*wz;
                real wx[PME_ORDER];
                for (int k = 0; k < PME_ORDER; k++)
                    wx[k] = w*simd_broadcast(data[k].x, src);
                if (!fits) {
                    const int3 atomIndex = simd_broadcast(gridIndex, src);
                    if (dy < PME_ORDER) {
                        int y = atomIndex.y+dy;
                        y -= (y >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                        int z = atomIndex.z+dz;
                        z -= (z >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                        for (int k = 0; k < PME_ORDER; k++) {
                            int x = atomIndex.x+k;
                            x -= (x >= GRID_SIZE_X ? GRID_SIZE_X : 0);
                            if (fabs(wx[k]) > 2.3e-10f)
                                ATOMIC_ADD(&pmeGrid[(x*GRID_SIZE_Y+y)*GRID_SIZE_Z+z], wx[k]);
                        }
                    }
                    continue;
                }
                const int index = simd_broadcast(packed, src)+laneOffset;
                if (dy < PME_ORDER)
                    for (int k = 0; k < PME_ORDER; k++)
                        slab[index+k*strideX] += wx[k];
                SYNC_WARPS;
            }
            start = end;
            if (!fits)
                continue;

            // Add each nonzero point to the grid and leave the slab zeroed for the next span.
            // Float reciprocals recover the coordinates exactly for volumes this small.

            const int3 base = origin+lo;
            const float invStrideX = 1.0f/strideX;
            const float invExtentZ = 1.0f/extent.z;
            for (int v = lane; v < volume; v += 32) {
                real value = slab[v];
                if (value == 0)
                    continue;
                slab[v] = 0;
                if (fabs(value) <= 2.3e-10f)
                    continue;
                int x = (int) ((v+0.5f)*invStrideX);
                int yz = v-x*strideX;
                int y = (int) ((yz+0.5f)*invExtentZ);
                int z = yz-y*extent.z;
                x += base.x;
                x += (x < 0 ? GRID_SIZE_X : (x >= GRID_SIZE_X ? -GRID_SIZE_X : 0));
                y += base.y;
                y += (y < 0 ? GRID_SIZE_Y : (y >= GRID_SIZE_Y ? -GRID_SIZE_Y : 0));
                z += base.z;
                z += (z < 0 ? GRID_SIZE_Z : (z >= GRID_SIZE_Z ? -GRID_SIZE_Z : 0));
                ATOMIC_ADD(&pmeGrid[(x*GRID_SIZE_Y+y)*GRID_SIZE_Z+z], value);
            }
            SYNC_WARPS;
        }
    }
#else
    // Process the atoms in spatially sorted order.  This improves efficiency when writing
    // the grid values.  PME_ORDER threads process one atom.

    real3 data[PME_ORDER];
    const real scale = RECIP((real) (PME_ORDER-1));
    for (int i = GLOBAL_ID; i < NUM_ATOMS*PME_ORDER; i += GLOBAL_SIZE) {
        int atom = pmeAtomGridIndex[i/PME_ORDER].x;
        real4 pos = posq[atom];
#ifdef CHARGE_FROM_SIGEPS
        const float2 sigEps = sigmaEpsilon[atom];
        const real charge = 8*sigEps.x*sigEps.x*sigEps.x*sigEps.y;
#else
        const real charge = (CHARGE)*EPSILON_FACTOR;
#endif
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);
        if (charge == 0)
            continue;

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.

        real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
        data[PME_ORDER-1] = make_real3(0);
        data[1] = dr;
        data[0] = make_real3(1)-dr;
        for (int j = 3; j < PME_ORDER; j++) {
            real div = RECIP((real) (j-1));
            data[j-1] = div*dr*data[j-2];
            for (int k = 1; k < (j-1); k++)
                data[j-k-1] = div*((dr+make_real3(k))*data[j-k-2] + (make_real3(j-k)-dr)*data[j-k-1]);
            data[0] = div*(make_real3(1)-dr)*data[0];
        }
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < (PME_ORDER-1); j++)
            data[PME_ORDER-j-1] = scale*((dr+make_real3(j))*data[PME_ORDER-j-2] + (make_real3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(make_real3(1)-dr)*data[0];

        // Spread the charge from this atom onto each grid point.  PME_ORDER threads access
        // consecutive addresses.

        int iz = i%PME_ORDER;
        int zindex = gridIndex.z+iz;
        zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
        real dz = 0;
        for (int i = 0; i < PME_ORDER; i++) {
            dz = i == iz ? data[i].z : dz;
        }
        dz *= charge;
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real dzdx = dz*data[ix].x;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = ybase*GRID_SIZE_Z;
                int index = xbase + ybase + zindex;
                real add = dzdx*data[iy].y;
                if (fabs(add) > 2.3e-10f) { // Smallest value representable in 64 bit fixed point
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
                    ATOMIC_ADD(&pmeGrid[index], (mm_ulong) realToFixedPoint(add));
#if defined(__GFX12__) && defined(USE_HIP)
                    // Workaround for rare cases when few values of pmeGrid are very large and
                    // incorrect. The cause is unknown. Why this workaround or other irrelevant
                    // changes like printf help is also unknown.
                    asm volatile("s_wait_storecnt 0x0");
#endif
#else
                    ATOMIC_ADD(&pmeGrid[index], add);
#endif
                }
            }
        }
    }
#endif
}

#ifdef USE_FIXED_POINT_CHARGE_SPREADING

KERNEL void finishSpreadCharge(
        GLOBAL const mm_long* RESTRICT grid1,
        GLOBAL real* RESTRICT grid2) {
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
    real scale = 1/(real) 0x100000000;
    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
        grid2[index] = scale*grid1[index];
    }
}

#endif

KERNEL void reciprocalConvolution(GLOBAL real2* RESTRICT pmeGrid, GLOBAL const real* RESTRICT pmeBsplineModuliX,
        GLOBAL const real* RESTRICT pmeBsplineModuliY, GLOBAL const real* RESTRICT pmeBsplineModuliZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    // R2C stores into a half complex matrix where the last dimension is cut by half
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*(GRID_SIZE_Z/2+1);
#ifdef USE_LJPME
    const real recipScaleFactor = -(2*M_PI/6)*SQRT(M_PI)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
    real bfac = M_PI / EWALD_ALPHA;
    real fac1 = 2*M_PI*M_PI*M_PI*SQRT(M_PI);
    real fac2 = EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA;
    real fac3 = -2*EWALD_ALPHA*M_PI*M_PI;
#else
    const real recipScaleFactor = RECIP(M_PI)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
#endif

    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
        // real indices
        int kx = index/(GRID_SIZE_Y*(GRID_SIZE_Z/2+1));
        int remainder = index-kx*GRID_SIZE_Y*(GRID_SIZE_Z/2+1);
        int ky = remainder/(GRID_SIZE_Z/2+1);
        int kz = remainder-ky*(GRID_SIZE_Z/2+1);
        int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx-GRID_SIZE_X);
        int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky-GRID_SIZE_Y);
        int mz = (kz < (GRID_SIZE_Z+1)/2) ? kz : (kz-GRID_SIZE_Z);
        real mhx = mx*recipBoxVecX.x;
        real mhy = mx*recipBoxVecY.x+my*recipBoxVecY.y;
        real mhz = mx*recipBoxVecZ.x+my*recipBoxVecZ.y+mz*recipBoxVecZ.z;
        real bx = pmeBsplineModuliX[kx];
        real by = pmeBsplineModuliY[ky];
        real bz = pmeBsplineModuliZ[kz];
        real2 grid = pmeGrid[index];
        real m2 = mhx*mhx+mhy*mhy+mhz*mhz;
#ifdef USE_LJPME
        real denom = recipScaleFactor/(bx*by*bz);
        real m = SQRT(m2);
        real m3 = m*m2;
        real b = bfac*m;
        real expfac = -b*b;
        real expterm = EXP(expfac);
        real erfcterm = ERFC(b);
        real eterm = (fac1*erfcterm*m3 + expterm*(fac2 + fac3*m2)) * denom;
        pmeGrid[index] = make_real2(grid.x*eterm, grid.y*eterm);
#else
        real denom = m2*bx*by*bz;
        real eterm = recipScaleFactor*EXP(-RECIP_EXP_FACTOR*m2)/denom;
        if (kx != 0 || ky != 0 || kz != 0) {
            pmeGrid[index] = make_real2(grid.x*eterm, grid.y*eterm);
        } else {
            pmeGrid[index] = make_real2(0);
        }
#endif
    }
}

KERNEL void gridEvaluateEnergy(GLOBAL real2* RESTRICT pmeGrid, GLOBAL mixed* RESTRICT energyBuffer,
                      GLOBAL const real* RESTRICT pmeBsplineModuliX, GLOBAL const real* RESTRICT pmeBsplineModuliY, GLOBAL const real* RESTRICT pmeBsplineModuliZ,
                      real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    // R2C stores into a half complex matrix where the last dimension is cut by half
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
 #ifdef USE_LJPME
    const real recipScaleFactor = -(2*M_PI/6)*SQRT(M_PI)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
    real bfac = M_PI / EWALD_ALPHA;
    real fac1 = 2*M_PI*M_PI*M_PI*SQRT(M_PI);
    real fac2 = EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA;
    real fac3 = -2*EWALD_ALPHA*M_PI*M_PI;
#else
    const real recipScaleFactor = RECIP(M_PI)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
#endif

    mixed energy = 0;
    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
        // real indices
        int kx = index/(GRID_SIZE_Y*(GRID_SIZE_Z));
        int remainder = index-kx*GRID_SIZE_Y*(GRID_SIZE_Z);
        int ky = remainder/(GRID_SIZE_Z);
        int kz = remainder-ky*(GRID_SIZE_Z);
        int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx-GRID_SIZE_X);
        int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky-GRID_SIZE_Y);
        int mz = (kz < (GRID_SIZE_Z+1)/2) ? kz : (kz-GRID_SIZE_Z);
        real mhx = mx*recipBoxVecX.x;
        real mhy = mx*recipBoxVecY.x+my*recipBoxVecY.y;
        real mhz = mx*recipBoxVecZ.x+my*recipBoxVecZ.y+mz*recipBoxVecZ.z;
        real m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        real bx = pmeBsplineModuliX[kx];
        real by = pmeBsplineModuliY[ky];
        real bz = pmeBsplineModuliZ[kz];
#ifdef USE_LJPME
        real denom = recipScaleFactor/(bx*by*bz);
        real m = SQRT(m2);
        real m3 = m*m2;
        real b = bfac*m;
        real expfac = -b*b;
        real expterm = EXP(expfac);
        real erfcterm = ERFC(b);
        real eterm = (fac1*erfcterm*m3 + expterm*(fac2 + fac3*m2)) * denom;
#else
        real denom = m2*bx*by*bz;
        real eterm = recipScaleFactor*EXP(-RECIP_EXP_FACTOR*m2)/denom;
#endif
        if (kz >= (GRID_SIZE_Z/2+1)) {
            kx = ((kx == 0) ? kx : GRID_SIZE_X-kx);
            ky = ((ky == 0) ? ky : GRID_SIZE_Y-ky);
            kz = GRID_SIZE_Z-kz;
        }
        int indexInHalfComplexGrid = kz + ky*(GRID_SIZE_Z/2+1)+kx*(GRID_SIZE_Y*(GRID_SIZE_Z/2+1));
        real2 grid = pmeGrid[indexInHalfComplexGrid];
#ifndef USE_LJPME
        if (kx != 0 || ky != 0 || kz != 0)
#endif
            energy += eterm*(grid.x*grid.x + grid.y*grid.y);
    }
#if defined(USE_PME_STREAM) && !defined(USE_LJPME)
    energyBuffer[GLOBAL_ID] = 0.5f*energy;
#else
    energyBuffer[GLOBAL_ID] += 0.5f*energy;
#endif
}

KERNEL void gridInterpolateForce(GLOBAL const real4* RESTRICT posq, GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL const real* RESTRICT pmeGrid,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, GLOBAL const int2* RESTRICT pmeAtomGridIndex,
#ifdef CHARGE_FROM_SIGEPS
        GLOBAL const float2* RESTRICT sigmaEpsilon
#else
        GLOBAL const real* RESTRICT charges
#endif
        ) {
    real3 data[PME_ORDER];
    real3 ddata[PME_ORDER];
    const real scale = RECIP((real) (PME_ORDER-1));

    // Process the atoms in spatially sorted order.  This improves cache performance when loading
    // the grid values.

    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        int atom = pmeAtomGridIndex[i].x;
        real3 force = make_real3(0);
        real4 pos = posq[atom];
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);
#ifdef CHARGE_FROM_SIGEPS
        const float2 sigEps = sigmaEpsilon[atom];
        real q = 8*sigEps.x*sigEps.x*sigEps.x*sigEps.y;
#else
        real q = CHARGE*EPSILON_FACTOR;
#endif
        if (q == 0)
            continue;

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.

        real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
        data[PME_ORDER-1] = make_real3(0);
        data[1] = dr;
        data[0] = make_real3(1)-dr;
        for (int j = 3; j < PME_ORDER; j++) {
            real div = RECIP((real) (j-1));
            data[j-1] = div*dr*data[j-2];
            for (int k = 1; k < (j-1); k++)
                data[j-k-1] = div*((dr+make_real3(k))*data[j-k-2] + (make_real3(j-k)-dr)*data[j-k-1]);
            data[0] = div*(make_real3(1)-dr)*data[0];
        }
        ddata[0] = -data[0];
        for (int j = 1; j < PME_ORDER; j++)
            ddata[j] = data[j-1]-data[j];
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < (PME_ORDER-1); j++)
            data[PME_ORDER-j-1] = scale*((dr+make_real3(j))*data[PME_ORDER-j-2] + (make_real3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(make_real3(1)-dr)*data[0];

        // Compute the force on this atom.

        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real dx = data[ix].x;
            real ddx = ddata[ix].x;

            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real dy = data[iy].y;
                real ddy = ddata[iy].y;

                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = gridIndex.z+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int index = ybase + zindex;
                    real gridvalue = pmeGrid[index];
                    force.x += ddx*dy*data[iz].z*gridvalue;
                    force.y += dx*ddy*data[iz].z*gridvalue;
                    force.z += dx*dy*ddata[iz].z*gridvalue;
                }
            }
        }
        real forceX = -q*(force.x*GRID_SIZE_X*recipBoxVecX.x);
        real forceY = -q*(force.x*GRID_SIZE_X*recipBoxVecY.x+force.y*GRID_SIZE_Y*recipBoxVecY.y);
        real forceZ = -q*(force.x*GRID_SIZE_X*recipBoxVecZ.x+force.y*GRID_SIZE_Y*recipBoxVecZ.y+force.z*GRID_SIZE_Z*recipBoxVecZ.z);
#if defined(USE_PME_STREAM) || defined(ATOMIC_PME_FORCES)
        ATOMIC_ADD(&forceBuffers[atom], (mm_ulong) realToFixedPoint(forceX));
        ATOMIC_ADD(&forceBuffers[atom+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceY));
        ATOMIC_ADD(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceZ));
#else
        forceBuffers[atom] += (mm_ulong) realToFixedPoint(forceX);
        forceBuffers[atom+PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(forceY);
        forceBuffers[atom+2*PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(forceZ);
#endif
    }
}

KERNEL void gridInterpolateChargeDerivatives(GLOBAL const real4* RESTRICT posq, GLOBAL mm_ulong* RESTRICT derivatives, GLOBAL const real* RESTRICT pmeGrid,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, GLOBAL const int* RESTRICT atomIndices
        ) {
    real3 data[PME_ORDER];
    const real scale = RECIP((real) (PME_ORDER-1));

    for (int i = GLOBAL_ID; i < NUM_INDICES; i += GLOBAL_SIZE) {
        int atom = atomIndices[i];
        real derivative = 0;
        real4 pos = posq[atom];
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.

        real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
        data[PME_ORDER-1] = make_real3(0);
        data[1] = dr;
        data[0] = make_real3(1)-dr;
        for (int j = 3; j < PME_ORDER; j++) {
            real div = RECIP((real) (j-1));
            data[j-1] = div*dr*data[j-2];
            for (int k = 1; k < (j-1); k++)
                data[j-k-1] = div*((dr+make_real3(k))*data[j-k-2] + (make_real3(j-k)-dr)*data[j-k-1]);
            data[0] = div*(make_real3(1)-dr)*data[0];
        }
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < (PME_ORDER-1); j++)
            data[PME_ORDER-j-1] = scale*((dr+make_real3(j))*data[PME_ORDER-j-2] + (make_real3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(make_real3(1)-dr)*data[0];

        // Compute the charge derivative on this atom.

        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real dx = data[ix].x;

            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real dy = data[iy].y;

                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = gridIndex.z+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    derivative += dx*dy*data[iz].z*pmeGrid[ybase + zindex];
                }
            }
        }
        derivative *= EPSILON_FACTOR;
#ifdef USE_PME_STREAM
        ATOMIC_ADD(&derivatives[i], (mm_ulong) realToFixedPoint(derivative));
#else
        derivatives[i] += (mm_ulong) realToFixedPoint(derivative);
#endif
    }
}

KERNEL void addForces(GLOBAL const real4* RESTRICT forces, GLOBAL mm_long* RESTRICT forceBuffers) {
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real4 f = forces[atom];
        forceBuffers[atom] += realToFixedPoint(f.x);
        forceBuffers[atom+PADDED_NUM_ATOMS] += realToFixedPoint(f.y);
        forceBuffers[atom+2*PADDED_NUM_ATOMS] += realToFixedPoint(f.z);
    }
}

KERNEL void addEnergy(GLOBAL const mixed* RESTRICT pmeEnergyBuffer, GLOBAL mixed* RESTRICT energyBuffer, int bufferSize) {
    for (int i = GLOBAL_ID; i < bufferSize; i += GLOBAL_SIZE)
        energyBuffer[i] += pmeEnergyBuffer[i];
}
