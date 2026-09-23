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
#include "TestCustomNonbondedForce.h"

void testTooManyArguments() {
    // Each per-particle parameter is a separate kernel argument, and a Metal kernel can have
    // at most 31 of them.

    const int numParameters = 32;
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    string energy = "r*(0";
    for (int i = 0; i < numParameters; i++)
        energy += "+p"+to_string(i)+"1*p"+to_string(i)+"2";
    CustomNonbondedForce* force = new CustomNonbondedForce(energy+")");
    for (int i = 0; i < numParameters; i++)
        force->addPerParticleParameter("p"+to_string(i));
    vector<double> params(numParameters, 1.0);
    force->addParticle(params);
    force->addParticle(params);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    bool thrown = false;
    try {
        Context context(system, integrator, platform);
        context.setPositions({Vec3(0, 0, 0), Vec3(1, 0, 0)});
        context.getState(State::Energy);
    }
    catch (const OpenMMException& e) {
        thrown = (string(e.what()).find("at most 31") != string::npos);
    }
    ASSERT(thrown);
}

void runPlatformTests() {
    testTooManyArguments();
}
