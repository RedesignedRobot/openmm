/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2021 Stanford University and the Authors.           *
 * Portions copyright (c) 2021 Advanced Micro Devices, Inc.                   *
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

#include "MetalAmoebaTests.h"
#include "TestHippoNonbondedForce.h"
#include "MetalDeterministicForces.h"

void testDeterministicPmeForces() {
    System system;
    HippoNonbondedForce* hippo = new HippoNonbondedForce();
    const int gridSize = 4;
    const double spacing = 0.31;
    const double boxSize = gridSize*spacing;
    buildWaterSystem(system, gridSize*gridSize*gridSize, hippo);
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    hippo->setNonbondedMethod(HippoNonbondedForce::PME);
    hippo->setCutoffDistance(0.5);
    hippo->setSwitchingDistance(0.4);
    hippo->setPMEParameters(3.85037, 24, 24, 24);
    hippo->setDPMEParameters(3.85037, 20, 20, 20);
    vector<Vec3> positions;
    for (int x = 0; x < gridSize; x++)
        for (int y = 0; y < gridSize; y++)
            for (int z = 0; z < gridSize; z++) {
                Vec3 oxygen = Vec3(x, y, z)*spacing;
                positions.push_back(oxygen);
                positions.push_back(oxygen+Vec3(0.09572, 0, 0));
                positions.push_back(oxygen+Vec3(-0.02399872, 0.09266272, 0));
            }
    assertDeterministicForces(system, platform, [&](Context& context) {
        context.setPositions(positions);
        return context.getState(State::Forces).getForces();
    });
}

void runPlatformTests() {
    testDeterministicPmeForces();
}
