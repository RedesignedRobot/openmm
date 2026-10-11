/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2017 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "MetalTests.h"
#include "TestDispersionPME.h"
#include "ReferencePlatform.h"

void testReciprocalForceAccuracy() {
    // The dispersion reciprocal space convolution subtracts nearly equal terms when erfc(b) is small,
    // so it needs erfc with a small relative error.  Compare the reciprocal space forces of a
    // Lennard-Jones fluid to the Reference platform's on the same grid.

    const int gridSide = 16;
    const double spacing = 0.37;
    const double boxSize = gridSide*spacing;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    NonbondedForce* force = new NonbondedForce();
    force->setNonbondedMethod(NonbondedForce::LJPME);
    force->setCutoffDistance(1.0);
    force->setEwaldErrorTolerance(5e-4);
    force->setReciprocalSpaceForceGroup(1);
    system.addForce(force);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions;
    for (int i = 0; i < gridSide; i++)
        for (int j = 0; j < gridSide; j++)
            for (int k = 0; k < gridSide; k++) {
                system.addParticle(39.9);
                if (positions.size()%2 == 0)
                    force->addParticle(0.0, 0.34, 0.996);
                else
                    force->addParticle(0.0, 0.30, 0.5);
                Vec3 jitter(genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5);
                positions.push_back(Vec3(i+0.5, j+0.5, k+0.5)*spacing + jitter*0.06);
            }
    VerletIntegrator integrator1(0.001), integrator2(0.001);
    Context context(system, integrator1, platform);
    ReferencePlatform reference;
    Context referenceContext(system, integrator2, reference);
    context.setPositions(positions);
    referenceContext.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy, false, 1<<1);
    State referenceState = referenceContext.getState(State::Forces | State::Energy, false, 1<<1);
    double diff = 0, norm = 0;
    for (int i = 0; i < system.getNumParticles(); i++) {
        Vec3 delta = state.getForces()[i]-referenceState.getForces()[i];
        diff += delta.dot(delta);
        norm += referenceState.getForces()[i].dot(referenceState.getForces()[i]);
    }
    ASSERT(sqrt(diff/norm) < 1e-4);
    ASSERT_EQUAL_TOL(referenceState.getPotentialEnergy(), state.getPotentialEnergy(), 1e-5);
}

void runPlatformTests() {
    testReciprocalForceAccuracy();
}
