{
    real invRSquaredOver4 = 0.25f*invR*invR;
    real rScaledRadiusJ = r+OBC_PARAMS2.y;
    real rScaledRadiusI = r+OBC_PARAMS1.y;
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
    real term1 = (0.5f*(0.25f+OBC_PARAMS2.y*OBC_PARAMS2.y*invRSquaredOver4)*t2J + t1J*invRSquaredOver4)*invR;
    real term2 = (0.5f*(0.25f+OBC_PARAMS1.y*OBC_PARAMS1.y*invRSquaredOver4)*t2I + t1I*invRSquaredOver4)*invR;
#ifdef CUSTOM_OBC2
    // term1/2 are -dI/dr/r for the integral including engulfment.  The app omits
    // 1/or-1/(sr-r), whose derivative is -1/(sr-r)^2.  At equality max(or, D)
    // selects or, so use the same strict inequality as the omitted correction.
    term1 -= (OBC_PARAMS1.x < OBC_PARAMS2.y-r ? invR*l_ij2J : 0);
    term2 -= (OBC_PARAMS2.x < OBC_PARAMS1.y-r ? invR*l_ij2I : 0);
#endif
    real tempdEdR = (OBC_PARAMS1.x < rScaledRadiusJ ? BORN_FORCE1*term1/0x100000000 : 0);
    tempdEdR += (OBC_PARAMS2.x < rScaledRadiusI ? BORN_FORCE2*term2/0x100000000 : 0);
#ifdef USE_CUTOFF
    unsigned int includeInteraction = (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 != atom2 && r2 < CUTOFF_SQUARED);
#else
    unsigned int includeInteraction = (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 != atom2);
#endif
    dEdR += (includeInteraction ? tempdEdR : (real) 0);
}
