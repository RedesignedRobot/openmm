/** Energy terms that depend on the authoritative periodic volume. */
KERNEL void addBoxEnergy(GLOBAL const uchar* RESTRICT boxState, GLOBAL mixed* RESTRICT energyBuffer, double coefficient) {
    // The host volume and coefficient keep their double precision in single mode too.
    double volume = *((GLOBAL const double*) (boxState+304));
    energyBuffer[0] += (mixed) (coefficient/volume);
}
