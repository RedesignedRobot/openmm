/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2015 Stanford University and the Authors.      *
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
#include "TestCustomExternalForce.h"
#include "sfmt/SFMT.h"

void testErfAccuracy() {
    // Metal has no erf or erfc, so the platform supplies its own.  Check erfc's relative error over the
    // whole range, including negative arguments and arguments where erfc is tiny, and erf's absolute
    // error.  Each particle's y force is -scale*f(x), with scale chosen to make it close to -1 for erfc,
    // so fixed point force accumulation keeps its relative precision.

    for (bool isErfc : {true, false}) {
        System system;
        CustomExternalForce* force = new CustomExternalForce(isErfc ? "y*scale*erfc(x)" : "y*scale*erf(x)");
        force->addPerParticleParameter("scale");
        system.addForce(force);
        vector<Vec3> positions;
        vector<double> expected;
        for (int i = -640; i <= 640; i++) {
            float x = i/64.0f;
            double value = (isErfc ? erfc((double) x) : erf((double) x));
            if (isErfc && value < 1e-30)
                continue;
            double scale = (isErfc ? 1.0/value : 1.0);
            system.addParticle(1.0);
            force->addParticle(positions.size(), {scale});
            positions.push_back(Vec3(x, 0.5, 0));
            expected.push_back(-scale*value);
        }
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
        context.setPositions(positions);
        State state = context.getState(State::Forces);
        for (int i = 0; i < (int) positions.size(); i++)
            ASSERT_EQUAL_TOL(expected[i], state.getForces()[i][1], isErfc ? 1e-6 : 3e-7);
    }
}

void runPlatformTests() {
    testErfAccuracy();
}
