/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2012-2015 Stanford University and the Authors.      *
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
#include "TestCheckpoints.h"
#include "openmm/CMMotionRemover.h"
#include "openmm/LangevinMiddleIntegrator.h"

/**
 * Build a periodic Lennard-Jones fluid with LangevinMiddleIntegrator, run it past an atom reorder and return a
 * checkpoint, along with the state 200 steps later.
 */
State runToCheckpoint(Context& context, LangevinMiddleIntegrator& integrator, string& checkpoint, double& time) {
    integrator.step(300);
    time = context.getState(0).getTime();
    stringstream stream(ios_base::out | ios_base::in | ios_base::binary);
    context.createCheckpoint(stream);
    checkpoint = stream.str();
    integrator.step(200);
    return context.getState(State::Positions | State::Velocities);
}

void buildFluid(System& system, vector<Vec3>& positions) {
    const int gridSize = 5;
    const double spacing = 0.4, boxSize = gridSize*spacing;
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    nonbonded->setCutoffDistance(0.9);
    system.addForce(nonbonded);
    system.addForce(new CMMotionRemover(1));
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            for (int k = 0; k < gridSize; k++) {
                system.addParticle(40.0);
                nonbonded->addParticle(0.0, 0.34, 1.0);
                positions.push_back(Vec3(i*spacing, j*spacing, k*spacing));
            }
}

/**
 * In mixed precision velm stores float pairs, which a double can't always hold.  A resumed run must still follow
 * the uninterrupted one bit for bit.
 */
void testResumeIsBitwise() {
    System system;
    vector<Vec3> positions;
    buildFluid(system, positions);
    LangevinMiddleIntegrator integrator(300.0, 1.0, 0.002);
    integrator.setRandomNumberSeed(5);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocitiesToTemperature(300.0, 5);
    string checkpoint;
    double time;
    State s1 = runToCheckpoint(context, integrator, checkpoint, time);
    stringstream stream(checkpoint, ios_base::out | ios_base::in | ios_base::binary);
    context.loadCheckpoint(stream);
    integrator.step(200);
    State s2 = context.getState(State::Positions | State::Velocities);
    for (int i = 0; i < system.getNumParticles(); i++)
        for (int j = 0; j < 3; j++) {
            ASSERT(s1.getPositions()[i][j] == s2.getPositions()[i][j]);
            ASSERT(s1.getVelocities()[i][j] == s2.getVelocities()[i][j]);
        }
}

/**
 * A mixed precision checkpoint written before the block of velm pairs existed still loads.
 */
void testCheckpointWithoutPairs() {
    System system;
    vector<Vec3> positions;
    buildFluid(system, positions);
    LangevinMiddleIntegrator integrator(300.0, 1.0, 0.002);
    integrator.setRandomNumberSeed(5);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocitiesToTemperature(300.0, 5);
    string checkpoint;
    double time;
    State s1 = runToCheckpoint(context, integrator, checkpoint, time);

    // Cut the block out: its marker, the velm words and the atom order.  The common checkpoint follows it and
    // starts with its version 3, mixed precision 1 and the time.

    const int marker = 0x4d505231, version = 3, mixed = 1;
    size_t start = checkpoint.find(string((char*) &marker, sizeof(int)));
    ASSERT(start != string::npos);
    string header = string((char*) &version, sizeof(int))+string((char*) &mixed, sizeof(int))+string((char*) &time, sizeof(double));
    size_t end = checkpoint.find(header, start+sizeof(int));
    ASSERT(end != string::npos);
    ASSERT_EQUAL(sizeof(int)+36*128, end-start);
    string old = checkpoint.substr(0, start)+checkpoint.substr(end);
    stringstream stream(old, ios_base::out | ios_base::in | ios_base::binary);
    context.loadCheckpoint(stream);
    integrator.step(200);
    State s2 = context.getState(State::Positions | State::Velocities);
    double maxDelta = 0;
    for (int i = 0; i < system.getNumParticles(); i++)
        maxDelta = max(maxDelta, sqrt((s1.getPositions()[i]-s2.getPositions()[i]).dot(s1.getPositions()[i]-s2.getPositions()[i])));
    ASSERT(maxDelta < 1e-3);
}

void runPlatformTests() {
    if (platform.getPropertyDefaultValue("Precision") != "mixed")
        return;
    testResumeIsBitwise();
    testCheckpointWithoutPairs();
}
