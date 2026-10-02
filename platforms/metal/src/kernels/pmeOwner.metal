// A bin holds stencil origins in a 4x4x4 tile.  The only atomic is a 32-bit
// linked-list insertion, never an update to the charge grid.  Dispatch completion
// publishes both the links and spline data before the gather dispatch reads them.
#define OWNER_BINS_X ((GRID_SIZE_X+3)/4)
#define OWNER_BINS_Y ((GRID_SIZE_Y+3)/4)
#define OWNER_BINS_Z ((GRID_SIZE_Z+3)/4)
#define OWNER_TILES (OWNER_BINS_X*OWNER_BINS_Y*OWNER_BINS_Z)

KERNEL void binPmeOwner(GLOBAL const real4* RESTRICT posq,
        GLOBAL int* RESTRICT heads, GLOBAL int4* RESTRICT atoms,
        GLOBAL real4* RESTRICT splines, GLOBAL const real* RESTRICT charges,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real4 pos = posq[atom];
        const real charge = (OWNER_CHARGE)*EPSILON_FACTOR;
        if (charge == 0)
            continue;
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 base = make_int3(((int) t.x)%GRID_SIZE_X, ((int) t.y)%GRID_SIZE_Y, ((int) t.z)%GRID_SIZE_Z);
#ifndef OWNER_PME_KILL
        real3 data[PME_ORDER];
        const real scale = RECIP((real) (PME_ORDER-1));
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

        for (int j = 0; j < PME_ORDER; j++)
            splines[atom*PME_ORDER+j] = make_real4(data[j].x, data[j].y, data[j].z, charge);
#endif
        int bin = ((base.x/4)*OWNER_BINS_Y+base.y/4)*OWNER_BINS_Z+base.z/4;
        int next = atomic_exchange_explicit((device atomic_int*) &heads[bin], atom+1, memory_order_relaxed);
        atoms[atom] = make_int4(base.x, base.y, base.z, next);
    }
}

KERNEL void gatherPmeOwner(GLOBAL const int* RESTRICT heads,
        GLOBAL const int4* RESTRICT atoms, GLOBAL const real4* RESTRICT splines,
        GLOBAL real* RESTRICT grid) {
    // Exactly 64 lanes, one lane per grid point.  The group-stride loop handles
    // backends that cap dispatch size.  Edge tiles retain their own partial extent.
    for (int tile = GROUP_ID; tile < OWNER_TILES; tile += NUM_GROUPS) {
        int3 owner = make_int3(tile/(OWNER_BINS_Y*OWNER_BINS_Z), (tile/OWNER_BINS_Z)%OWNER_BINS_Y, tile%OWNER_BINS_Z);
        int3 origin = owner*4;
        int3 point = origin+make_int3(LOCAL_ID/16, (LOCAL_ID/4)%4, LOCAL_ID%4);
        if (point.x >= GRID_SIZE_X || point.y >= GRID_SIZE_Y || point.z >= GRID_SIZE_Z)
            continue;
        // The four preceding grid coordinates are the halo.  Walk cyclic bin
        // indices through the owner, including short final bins at periodic seams.
        int3 first = make_int3(((origin.x+GRID_SIZE_X-4)%GRID_SIZE_X)/4,
                              ((origin.y+GRID_SIZE_Y-4)%GRID_SIZE_Y)/4,
                              ((origin.z+GRID_SIZE_Z-4)%GRID_SIZE_Z)/4);
        // With fewer than eight points the halo can begin inside the owner
        // after wrapping.  Visit every bin once in that dimension.
        if (GRID_SIZE_X < 8) first.x = (owner.x+1)%OWNER_BINS_X;
        if (GRID_SIZE_Y < 8) first.y = (owner.y+1)%OWNER_BINS_Y;
        if (GRID_SIZE_Z < 8) first.z = (owner.z+1)%OWNER_BINS_Z;
#ifdef OWNER_PME_INTERIOR_ONLY
        first = owner;
#endif
        mm_ulong sum = 0;
#ifndef OWNER_PME_WRITE_ONLY
        int bx = first.x;
        while (true) {
            int by = first.y;
            while (true) {
                int bz = first.z;
                while (true) {
                    int link = heads[(bx*OWNER_BINS_Y+by)*OWNER_BINS_Z+bz];
                    while (link != 0) {
                        int atom = link-1;
                        int4 entry = atoms[atom];
                        int3 d = point-make_int3(entry.x, entry.y, entry.z);
                        d.x += (d.x < 0 ? GRID_SIZE_X : 0);
                        d.y += (d.y < 0 ? GRID_SIZE_Y : 0);
                        d.z += (d.z < 0 ? GRID_SIZE_Z : 0);
                        if (d.x < PME_ORDER && d.y < PME_ORDER && d.z < PME_ORDER) {
#ifdef OWNER_PME_KILL
                            // Optimistic membership/gather/write floor, no spline
                            // computation, spline loads or charge multiplication.
                            sum += 1;
#else
                            real4 z = splines[atom*PME_ORDER+d.z];
                            real dz = z.z*z.w;
                            real dzdx = dz*splines[atom*PME_ORDER+d.x].x;
                            real add = dzdx*splines[atom*PME_ORDER+d.y].y;
                            if (fabs(add) > 2.3e-10f)
                                sum += (mm_ulong) realToFixedPoint(add);
#endif
                        }
                        link = entry.w;
                    }
                    if (bz == owner.z) break;
                    bz = (bz+1)%OWNER_BINS_Z;
                }
                if (by == owner.y) break;
                by = (by+1)%OWNER_BINS_Y;
            }
            if (bx == owner.x) break;
            bx = (bx+1)%OWNER_BINS_X;
        }
#endif
        grid[(point.x*GRID_SIZE_Y+point.y)*GRID_SIZE_Z+point.z] = (real) ((mm_long) sum)*(1/(real) 0x100000000);
    }
}
