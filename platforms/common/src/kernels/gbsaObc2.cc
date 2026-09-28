{
    real rScaledRadiusJ = r+OBC_PARAMS2.y;
    real rScaledRadiusI = r+OBC_PARAMS1.y;
    real term1, term2;
#ifndef USE_DOUBLE_PRECISION
    // Beyond r = R+S each term is invR^3*x^3*Q(x^2), x = S/r, with Q(y) = sum over m >= 1 of 2m(m+1)/(2m+1) y^(m-1),
    // fitted to float precision (1.6e-7 relative) for x <= 0.6.  It needs no log or divide and avoids the closed
    // form's cancellation at small x.
    if (r >= OBC_PARAMS1.x+OBC_PARAMS2.y && r >= OBC_PARAMS2.x+OBC_PARAMS1.y && max(OBC_PARAMS1.y, OBC_PARAMS2.y) <= 0.6f*r) {
        real invR3 = invR*invR*invR;
        real xJ = OBC_PARAMS2.y*invR;
        real xI = OBC_PARAMS1.y*invR;
        real yJ = xJ*xJ;
        real yI = xI*xI;
        term1 = invR3*xJ*yJ*(1.33333325f+yJ*(2.40000772f+yJ*(3.42801404f+yJ*(4.46175146f+yJ*(5.17209673f+yJ*(9.1373148f+
                yJ*(-7.88075542f+yJ*(61.6550179f+yJ*(-95.4890366f+yJ*107.52388f)))))))));
        term2 = invR3*xI*yI*(1.33333325f+yI*(2.40000772f+yI*(3.42801404f+yI*(4.46175146f+yI*(5.17209673f+yI*(9.1373148f+
                yI*(-7.88075542f+yI*(61.6550179f+yI*(-95.4890366f+yI*107.52388f)))))))));
    }
    else
#endif
    {
        real invRSquaredOver4 = 0.25f*invR*invR;
        real lowerLimitJ = max((real) OBC_PARAMS1.x, fabs(r-OBC_PARAMS2.y));
        real lowerLimitI = max((real) OBC_PARAMS2.x, fabs(r-OBC_PARAMS1.y));
        real l_ijJ = RECIP(lowerLimitJ);
        real l_ijI = RECIP(lowerLimitI);
        real u_ijJ = RECIP(rScaledRadiusJ);
        real u_ijI = RECIP(rScaledRadiusI);
        real l_ij2J = l_ijJ*l_ijJ;
        real l_ij2I = l_ijI*l_ijI;
        real u_ij2J = u_ijJ*u_ijJ;
        real u_ij2I = u_ijI*u_ijI;
        real t1J = LOG(u_ijJ*lowerLimitJ);
        real t1I = LOG(u_ijI*lowerLimitI);
        real t2J = (l_ij2J-u_ij2J);
        real t2I = (l_ij2I-u_ij2I);
        term1 = (0.5f*(0.25f+OBC_PARAMS2.y*OBC_PARAMS2.y*invRSquaredOver4)*t2J + t1J*invRSquaredOver4)*invR;
        term2 = (0.5f*(0.25f+OBC_PARAMS1.y*OBC_PARAMS1.y*invRSquaredOver4)*t2I + t1I*invRSquaredOver4)*invR;
    }
    real tempdEdR = (OBC_PARAMS1.x < rScaledRadiusJ ? BORN_FORCE1*term1/0x100000000 : 0);
    tempdEdR += (OBC_PARAMS2.x < rScaledRadiusI ? BORN_FORCE2*term2/0x100000000 : 0);
#ifdef USE_CUTOFF
    unsigned int includeInteraction = (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 != atom2 && r2 < CUTOFF_SQUARED);
#else
    unsigned int includeInteraction = (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 != atom2);
#endif
    dEdR += (includeInteraction ? tempdEdR : (real) 0);
}
