/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2023 Stanford University and the Authors.           *
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
#include "TestCustomCPPForce.h"

/**
 * A harmonic restraint to the origin, computed on the host.
 */
class RestraintForce : public Force {
protected:
    ForceImpl* createImpl() const;
};

class RestraintForceImpl : public CustomCPPForceImpl {
public:
    const RestraintForce& owner;
    RestraintForceImpl(const RestraintForce& owner) : CustomCPPForceImpl(owner), owner(owner) {
    }
    double computeForce(ContextImpl& context, const vector<Vec3>& positions, vector<Vec3>& forces) {
        double energy = 0;
        for (int i = 0; i < positions.size(); i++) {
            forces[i] = -positions[i];
            energy += 0.5*positions[i].dot(positions[i]);
        }
        return energy;
    }
    const RestraintForce& getOwner() const {
        return owner;
    }
};

ForceImpl* RestraintForce::createImpl() const {
    return new RestraintForceImpl(*this);
}

void testWorkerThreadUploads() {
    // CustomCPPForce uploads its forces from a worker thread while the main thread encodes the
    // PME kernels, so every step exercises concurrent use of the command queue.

    const int gridSize = 15;
    const double spacing = 0.4;
    System system;
    double boxSize = gridSize*spacing;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->setCutoffDistance(1.0);
    RestraintForce* restraint = new RestraintForce();
    restraint->setForceGroup(1);
    vector<Vec3> positions;
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            for (int k = 0; k < gridSize; k++) {
                int index = system.addParticle(10.0);
                nonbonded->addParticle(index%2 == 0 ? 0.2 : -0.2, 0.3, 0.5);
                positions.push_back(Vec3(i, j, k)*spacing);
            }
    system.addForce(nonbonded);
    system.addForce(restraint);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    for (int i = 0; i < 50; i++) {
        integrator.step(10);
        State all = context.getState(State::Positions | State::Forces);
        State group1 = context.getState(State::Forces, false, 1<<1);
        State group0 = context.getState(State::Forces, false, 1<<0);
        for (int j = 0; j < system.getNumParticles(); j++) {
            ASSERT_EQUAL_VEC(-all.getPositions()[j], group1.getForces()[j], 1e-5);
            ASSERT_EQUAL_VEC(all.getForces()[j], group0.getForces()[j]+group1.getForces()[j], 1e-3);
        }
    }
}

void runPlatformTests() {
    testWorkerThreadUploads();
}

