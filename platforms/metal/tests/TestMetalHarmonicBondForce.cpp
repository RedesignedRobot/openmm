/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2024 Stanford University and the Authors.      *
 * Portions copyright (c) 2026 the Authors.                                   *
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
#include "TestHarmonicBondForce.h"
#include "openmm/HarmonicAngleForce.h"

void testBondChunks() {
    // The bonded kernel sums each threadgroup's chunk of terms in threadgroup memory (MetalBondedUtilities).
    // A threadgroup takes more than one chunk of a force once the force has more terms than the grid has
    // threads, 46,080 on a 60-core GPU.  The angles are in another force group, so each group is checked
    // alone and with the other.  The chain snakes through a 50x50x41 grid 0.1 nm apart: along a straight
    // line, float positions near 10,000 nm would be off by 5e-4 nm and move the bond energy by 0.5%.
    const int numParticles = 100001;
    const int side = 50;
    System system;
    HarmonicBondForce* bonds = new HarmonicBondForce();
    HarmonicAngleForce* angles = new HarmonicAngleForce();
    angles->setForceGroup(1);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        int row = i/side, column = i%side, plane = row/side, rowInPlane = row%side;
        if (row%2 == 1)
            column = side-1-column;
        if (plane%2 == 1)
            rowInPlane = side-1-rowInPlane;
        positions[i] = Vec3(0.1*column, 0.1*rowInPlane+0.02*(i%3), 0.1*plane+0.03*(i%2));
    }
    for (int i = 1; i < numParticles; i++)
        bonds->addBond(i-1, i, 0.11, 1000.0);
    for (int i = 2; i < numParticles; i++)
        angles->addAngle(i-2, i-1, i, 2.0, 100.0);
    system.addForce(bonds);
    system.addForce(angles);
    VerletIntegrator integrator(0.001), referenceIntegrator(0.001);
    Context context(system, integrator, platform);
    Context referenceContext(system, referenceIntegrator, Platform::getPlatformByName("Reference"));
    context.setPositions(positions);
    referenceContext.setPositions(positions);
    for (int groups = 1; groups <= 3; groups++) {
        State state = context.getState(State::Forces | State::Energy, false, groups);
        State expected = referenceContext.getState(State::Forces | State::Energy, false, groups);
        ASSERT_EQUAL_TOL(expected.getPotentialEnergy(), state.getPotentialEnergy(), 1e-4);
        for (int i = 0; i < numParticles; i++)
            ASSERT_EQUAL_VEC(expected.getForces()[i], state.getForces()[i], 1e-4);
    }
}

void runPlatformTests() {
    testBondChunks();
}
