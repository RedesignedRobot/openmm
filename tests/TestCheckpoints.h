/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2012-2024 Stanford University and the Authors.      *
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

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/AndersenThermostat.h"
#include "openmm/Context.h"
#include "openmm/CustomIntegrator.h"
#include "openmm/LangevinIntegrator.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "sfmt/SFMT.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;

void compareStates(State& s1, State& s2) {
    ASSERT_EQUAL_TOL(s1.getTime(), s2.getTime(), TOL);
    int numParticles = s1.getPositions().size();
    for (int i = 0; i < numParticles; i++) {
        ASSERT_EQUAL_VEC(s1.getPositions()[i], s2.getPositions()[i], TOL);
        ASSERT_EQUAL_VEC(s1.getVelocities()[i], s2.getVelocities()[i], TOL);
        Vec3 a1, b1, c1, a2, b2, c2;
        s1.getPeriodicBoxVectors(a1, b1, c1);
        s2.getPeriodicBoxVectors(a2, b2, c2);
        ASSERT_EQUAL_VEC(a1, a2, TOL);
        ASSERT_EQUAL_VEC(b1, b2, TOL);
        ASSERT_EQUAL_VEC(c1, c2, TOL);
        for (map<string, double>::const_iterator iter = s1.getParameters().begin(); iter != s1.getParameters().end(); ++iter)
            ASSERT_EQUAL(iter->second, (*s2.getParameters().find(iter->first)).second);
    }
}

void testSetState() {
    const int numParticles = 10;
    const double boxSize = 3.0;
    const double temperature = 200.0;
    System system;
    system.addForce(new AndersenThermostat(0.0, 100.0));
    NonbondedForce* nonbonded = new NonbondedForce();
    system.addForce(nonbonded);
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(i%2 == 0 ? 0.1 : -0.1, 0.2, 0.1);
        positions[i] = Vec3(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    context.setParameter(AndersenThermostat::Temperature(), temperature);
    
    // Run for a little while.
    
    integrator.step(100);
    
    // Record the current state.
    
    State s1 = context.getState(State::Positions | State::Velocities | State::Parameters);
    
    // Continue the simulation for a few more steps and record a partial state.
    
    integrator.step(10);
    State s2 = context.getState(State::Positions);
    
    // Restore the original state and see if everything gets restored correctly.
    
    context.setPeriodicBoxVectors(Vec3(2*boxSize, 0, 0), Vec3(0, 2*boxSize, 0), Vec3(0, 0, 2*boxSize));
    context.setParameter(AndersenThermostat::Temperature(), temperature+10);
    context.setState(s1);
    State s3 = context.getState(State::Positions | State::Velocities | State::Parameters);
    compareStates(s1, s3);
    
    // Set the partial state and see if the correct things were set.
    
    context.setState(s2);
    State s4 = context.getState(State::Positions | State::Velocities | State::Parameters);
    for (int i = 0; i < numParticles; i++) {
        ASSERT_EQUAL_VEC(s2.getPositions()[i], s4.getPositions()[i], TOL);
        ASSERT_EQUAL_VEC(s1.getVelocities()[i], s4.getVelocities()[i], TOL);
    }
}

void testMultipleDevices() {
    const int numParticles = 100;
    const double boxSize = 5.0;
    const double temperature = 200.0;
    System system;
    system.addForce(new AndersenThermostat(0.0, 100.0));
    NonbondedForce* nonbonded = new NonbondedForce();
    system.addForce(nonbonded);
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(i%2 == 0 ? 0.1 : -0.1, 0.2, 0.1);
        bool clash;
        do {
            clash = false;
            positions[i] = Vec3(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
            for (int j = 0; j < i; j++) {
                Vec3 delta = positions[i]-positions[j];
                if (sqrt(delta.dot(delta)) < 0.1)
                    clash = true;
            }
        } while (clash);
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    context.setParameter(AndersenThermostat::Temperature(), temperature);
    
    // Run for a little while.
    
    integrator.step(100);
    
    // Record the current state and make a checkpoint.
    
    State s1 = context.getState(State::Positions | State::Velocities | State::Parameters);
    stringstream stream1(ios_base::out | ios_base::in | ios_base::binary);
    context.createCheckpoint(stream1);
    
    // Continue the simulation for a few more steps and record the state again.
    
    integrator.step(10);
    State s2 = context.getState(State::Positions | State::Velocities | State::Parameters);
    
    // Restore from the checkpoint and see if everything gets restored correctly.
    
    context.setPeriodicBoxVectors(Vec3(2*boxSize, 0, 0), Vec3(0, 2*boxSize, 0), Vec3(0, 0, 2*boxSize));
    context.setParameter(AndersenThermostat::Temperature(), temperature+10);
    context.loadCheckpoint(stream1);
    State s3 = context.getState(State::Positions | State::Velocities | State::Parameters);
    compareStates(s1, s3);
    
    // Now simulate from there and see if the trajectory is identical.
    
    integrator.step(10);
    State s4 = context.getState(State::Positions | State::Velocities | State::Parameters);
    compareStates(s2, s4);
    
    // Create a new Context that uses multiple devices.

    map<string, string> props;
    try {
        string deviceIndex = platform.getPropertyValue(context, "DeviceIndex");
        props["DeviceIndex"] = deviceIndex+","+deviceIndex;
    }
    catch (OpenMMException& ex) {
        // This platform doesn't have a DeviceIndex property.
    }
    VerletIntegrator integrator2(0.001);
    Context context2(system, integrator2, platform, props);
    context2.setPositions(positions);
    context2.setPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    context2.setParameter(AndersenThermostat::Temperature(), temperature);
    
    // Now repeat all of the above tests with it.

    integrator2.step(100);
    State s5 = context2.getState(State::Positions | State::Velocities | State::Parameters);
    stringstream stream2(ios_base::out | ios_base::in | ios_base::binary);
    context2.createCheckpoint(stream2);
    integrator2.step(10);
    State s6 = context2.getState(State::Positions | State::Velocities | State::Parameters);
    context2.setPeriodicBoxVectors(Vec3(2*boxSize, 0, 0), Vec3(0, 2*boxSize, 0), Vec3(0, 0, 2*boxSize));
    context2.setParameter(AndersenThermostat::Temperature(), temperature+10);
    context2.loadCheckpoint(stream2);
    State s7 = context2.getState(State::Positions | State::Velocities | State::Parameters);
    compareStates(s5, s7);
    integrator2.step(10);
    State s8 = context2.getState(State::Positions | State::Velocities | State::Parameters);
    compareStates(s6, s8);
    
    // See if a checkpoint created from one Context can be loaded into a different one.
    
    VerletIntegrator integrator3(0.001);
    Context context3(system, integrator3, platform);
    stream1.seekg(0, stream1.beg);
    context3.loadCheckpoint(stream1);
    State s9 = context3.getState(State::Positions | State::Velocities | State::Parameters | State::Energy);
    compareStates(s1, s9);
}

void testLangevin() {
    const int numParticles = 10;
    const double boxSize = 3.0;
    System system;
    NonbondedForce* nonbonded = new NonbondedForce();
    system.addForce(nonbonded);
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(i%2 == 0 ? 0.1 : -0.1, 0.2, 0.1);
        positions[i] = Vec3(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
    }
    LangevinIntegrator integrator(300.0, 1.0, 0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));

    // Run for a little while.

    integrator.step(100);

    // Record the current state and make a checkpoint.

    State s1 = context.getState(State::Positions | State::Velocities | State::Parameters);
    stringstream stream1(ios_base::out | ios_base::in | ios_base::binary);
    context.createCheckpoint(stream1);

    // Continue the simulation for a few more steps and record the state again.

    integrator.step(10);
    State s2 = context.getState(State::Positions | State::Velocities | State::Parameters);

    // Restore from the checkpoint and see if everything gets restored correctly.

    context.setPeriodicBoxVectors(Vec3(2*boxSize, 0, 0), Vec3(0, 2*boxSize, 0), Vec3(0, 0, 2*boxSize));
    context.loadCheckpoint(stream1);
    State s3 = context.getState(State::Positions | State::Velocities | State::Parameters);
    compareStates(s1, s3);

    // Now simulate from there and see if the trajectory is identical.

    integrator.step(10);
    State s4 = context.getState(State::Positions | State::Velocities | State::Parameters);
    compareStates(s2, s4);
}

void assertBitwiseEqual(const vector<Vec3>& expected, const vector<Vec3>& found, const string& quantity, int step) {
    double maxDiff = 0;
    for (int i = 0; i < expected.size(); i++)
        for (int j = 0; j < 3; j++)
            maxDiff = max(maxDiff, fabs(expected[i][j]-found[i][j]));
    if (maxDiff != 0) {
        stringstream message;
        message << quantity << " differ at step " << step << ", max difference " << maxDiff;
        throw OpenMMException(message.str());
    }
}

Integrator* createResumeIntegrator(bool custom) {
    if (custom) {
        // Velocity Verlet, which reuses the forces from the end of the previous step.

        CustomIntegrator* integrator = new CustomIntegrator(0.002);
        integrator->addPerDofVariable("x1", 0);
        integrator->addUpdateContextState();
        integrator->addComputePerDof("v", "v+0.5*dt*f/m");
        integrator->addComputePerDof("x", "x+dt*v");
        integrator->addComputePerDof("x1", "x");
        integrator->addConstrainPositions();
        integrator->addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt");
        integrator->addConstrainVelocities();
        return integrator;
    }
    LangevinMiddleIntegrator* integrator = new LangevinMiddleIntegrator(300.0, 1.0, 0.002);
    integrator->setRandomNumberSeed(5);
    return integrator;
}

/**
 * A Context that loads a checkpoint should compute exactly the same forces as the Context
 * that wrote it, and both should then follow the same trajectory.  This needs a platform
 * whose forces are deterministic.  The box must be large enough that a neighbor list doesn't
 * hold every atom, or lists built at different steps are always the same.
 */
void testResumeIsExact(bool customIntegrator) {
    const int gridSize = 13;
    const double boxSize = 4.0;
    const double spacing = boxSize/gridSize;
    const double angle = 104.52*M_PI/180;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    NonbondedForce* nonbonded = new NonbondedForce();
    system.addForce(nonbonded);
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->setCutoffDistance(0.9);
    vector<Vec3> positions;
    vector<pair<int, int> > bonds;
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            for (int k = 0; k < gridSize; k++) {
                int oxygen = system.addParticle(15.999);
                system.addParticle(1.008);
                system.addParticle(1.008);
                nonbonded->addParticle(-0.834, 0.31507524, 0.635968);
                nonbonded->addParticle(0.417, 1.0, 0.0);
                nonbonded->addParticle(0.417, 1.0, 0.0);
                Vec3 pos(i*spacing, j*spacing, k*spacing);
                positions.push_back(pos);
                positions.push_back(pos+Vec3(0.09572, 0, 0));
                positions.push_back(pos+Vec3(0.09572*cos(angle), 0.09572*sin(angle), 0));
                system.addConstraint(oxygen, oxygen+1, 0.09572);
                system.addConstraint(oxygen, oxygen+2, 0.09572);
                system.addConstraint(oxygen+1, oxygen+2, 0.15139);
                bonds.push_back(make_pair(oxygen, oxygen+1));
                bonds.push_back(make_pair(oxygen, oxygen+2));
            }
    nonbonded->createExceptionsFromBonds(bonds, 0.0, 0.0);
    map<string, string> properties;
    vector<string> names = platform.getPropertyNames();
    if (find(names.begin(), names.end(), "DeterministicForces") != names.end())
        properties["DeterministicForces"] = "true";
    unique_ptr<Integrator> integrator(createResumeIntegrator(customIntegrator));
    Context context(system, *integrator, platform, properties);
    context.setPositions(positions);
    context.setVelocitiesToTemperature(300.0, 5);
    integrator->step(50);

    // The Context only rebuilds its neighbor list when atoms have moved far enough, so most steps
    // use a list built at an earlier step.  Write a checkpoint after each of several steps, load
    // it into a new Context, and compare the forces.

    for (int step = 1; step <= 20; step++) {
        integrator->step(1);
        stringstream checkpoint(ios_base::out | ios_base::in | ios_base::binary);
        context.createCheckpoint(checkpoint);
        unique_ptr<Integrator> integrator2(createResumeIntegrator(customIntegrator));
        Context context2(system, *integrator2, platform, properties);
        context2.loadCheckpoint(checkpoint);
        State s1 = context.getState(State::Positions | State::Forces);
        State s2 = context2.getState(State::Positions | State::Forces);
        assertBitwiseEqual(s1.getPositions(), s2.getPositions(), "Positions", step);
        assertBitwiseEqual(s1.getForces(), s2.getForces(), "Forces", step);
    }

    // Write one more checkpoint, continue both simulations without querying either Context,
    // and see if the trajectories are identical.

    integrator->step(1);
    stringstream checkpoint(ios_base::out | ios_base::in | ios_base::binary);
    context.createCheckpoint(checkpoint);
    unique_ptr<Integrator> integrator2(createResumeIntegrator(customIntegrator));
    Context context2(system, *integrator2, platform, properties);
    context2.loadCheckpoint(checkpoint);
    integrator->step(100);
    integrator2->step(100);
    State s1 = context.getState(State::Positions | State::Velocities | State::Forces);
    State s2 = context2.getState(State::Positions | State::Velocities | State::Forces);
    assertBitwiseEqual(s1.getPositions(), s2.getPositions(), "Positions", 121);
    assertBitwiseEqual(s1.getVelocities(), s2.getVelocities(), "Velocities", 121);
    assertBitwiseEqual(s1.getForces(), s2.getForces(), "Forces", 121);
}

void runPlatformTests();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testSetState();
        testMultipleDevices();
        testLangevin();
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
