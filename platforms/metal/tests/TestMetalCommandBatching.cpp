/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 the Authors.                                   *
 * Authors:                                                                   *
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

/**
 * This tests that the Metal platform batches kernels into a few command buffers per step,
 * instead of submitting each kernel separately, and that work left open when step() returns
 * still reaches the GPU.
 */

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "MetalContext.h"
#include "MetalPlatform.h"
#include "MetalQueue.h"
#include "openmm/Context.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include <chrono>
#include <iostream>
#include <thread>

using namespace OpenMM;
using namespace std;

/**
 * Exposes the command queue of a Context.
 */
class QueuePlatform : public MetalPlatform {
public:
    MetalQueue& getQueue(Context& context) const {
        PlatformData* data = reinterpret_cast<PlatformData*>(getContextImpl(context).getPlatformData());
        return data->contexts[0]->getCurrentStream();
    }
};

static QueuePlatform platform;

void createSystem(System& system, vector<Vec3>& positions, NonbondedForce::NonbondedMethod method) {
    // NonbondedForce only uses a neighbor list for more than 3000 particles.

    const int gridSize = 15;
    const double spacing = 0.4;
    double boxSize = gridSize*spacing;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(method);
    nonbonded->setCutoffDistance(1.0);
    HarmonicBondForce* bonds = new HarmonicBondForce();
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            for (int k = 0; k < gridSize; k++) {
                int index = system.addParticle(10.0);
                nonbonded->addParticle(index%2 == 0 ? 0.2 : -0.2, 0.3, 0.5);
                positions.push_back(Vec3(i, j, k)*spacing);
                if (k%2 == 1) {
                    bonds->addBond(index-1, index, spacing, 1000.0);
                    nonbonded->addException(index-1, index, 0.0, 1.0, 0.0);
                }
            }
    system.addForce(nonbonded);
    system.addForce(bonds);
}

double measureCommitsPerStep(NonbondedForce::NonbondedMethod method) {
    System system;
    vector<Vec3> positions;
    createSystem(system, positions, method);
    LangevinMiddleIntegrator integrator(300.0, 1.0, 0.002);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    integrator.step(10);
    MetalQueue& queue = platform.getQueue(context);
    const int numSteps = 100;
    long long start = queue.getCommitCount();
    integrator.step(numSteps);
    context.getState(State::Positions);
    return (queue.getCommitCount()-start)/(double) numSteps;
}

void testCommitsPerStep() {
    // Each step is submitted once.  With a neighbor list, the buffer holds the neighbor list and the
    // force kernels, and is submitted before the host waits for the interaction count, if it has to.
    // The rest of the step goes with the next step's neighbor list.

    double noCutoff = measureCommitsPerStep(NonbondedForce::NoCutoff);
    double pme = measureCommitsPerStep(NonbondedForce::PME);
    cout << "Commits per step: NoCutoff " << noCutoff << ", PME " << pme << endl;
    ASSERT(noCutoff <= 1.1);
    ASSERT(pme <= 1.1);
}

void testWorkLeftOpenAfterStep() {
    // Work left open when step() returns must reach the GPU while the host does something else, and
    // committing it from another thread must not change the trajectory.

    System system;
    vector<Vec3> positions;
    createSystem(system, positions, NonbondedForce::PME);
    map<string, string> properties;
    properties[MetalPlatform::MetalDeterministicForces()] = "true";
    LangevinMiddleIntegrator integrator1(300.0, 1.0, 0.002);
    LangevinMiddleIntegrator integrator2(300.0, 1.0, 0.002);
    integrator1.setRandomNumberSeed(5);
    integrator2.setRandomNumberSeed(5);
    Context context1(system, integrator1, platform, properties);
    Context context2(system, integrator2, platform, properties);
    context1.setPositions(positions);
    context2.setPositions(positions);

    // After 25 steps the neighbor list may ask for an early reorder, based on an interaction count whose
    // age depends on how far ahead the host runs, so the two contexts could reorder at different steps.

    const int numSteps = 25;
    MetalQueue& queue = platform.getQueue(context1);
    long long start = queue.getCommitCount();
    integrator1.step(numSteps);

    // The host leaves the queue alone for 50 ms, and a slow GPU gets up to 10 s more.

    this_thread::sleep_for(chrono::milliseconds(50));
    auto deadline = chrono::steady_clock::now()+chrono::seconds(10);
    while (!queue.isDrained() && chrono::steady_clock::now() < deadline)
        this_thread::sleep_for(chrono::milliseconds(1));
    ASSERT(queue.isDrained());
    cout << "Commits per step without a sync: " << (queue.getCommitCount()-start)/(double) numSteps << endl;

    // context2 synchronizes after every step, so no command buffer holds work from two steps.

    for (int i = 0; i < numSteps; i++) {
        integrator2.step(1);
        context2.getState(State::Positions);
    }
    vector<Vec3> positions1 = context1.getState(State::Positions).getPositions();
    vector<Vec3> positions2 = context2.getState(State::Positions).getPositions();
    for (int i = 0; i < system.getNumParticles(); i++)
        ASSERT(positions1[i] == positions2[i]);
}

int main(int argc, char* argv[]) {
    try {
        testCommitsPerStep();
        testWorkLeftOpenAfterStep();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
