/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2011-2014 Stanford University and the Authors.      *
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

#include "MetalTests.h"
#include "TestRpmd.h"
#include "openmm/CustomExternalForce.h"
#include <cmath>

extern "C" void registerRPMDMetalKernelFactories();

using namespace OpenMM;

void testOddBeadThermostat() {
    const int numParticles = 100;
    const int samples = 300;
    const double temperature = 300;
    const double mass = 1;
    for (int copies : {3, 5, 4}) {
        System system;
        for (int i = 0; i < numParticles; i++)
            system.addParticle(mass);
        RPMDIntegrator integrator(copies, temperature, 10, 0.002);
        integrator.setRandomNumberSeed(1234);
        Context context(system, integrator, platform);
        for (int bead = 0; bead < copies; bead++) {
            integrator.setPositions(bead, vector<Vec3>(numParticles, Vec3()));
            integrator.setVelocities(bead, vector<Vec3>(numParticles, Vec3()));
        }
        integrator.step(300);
        vector<double> cosine2(copies/2+1, 0.0), sine2(copies/2+1, 0.0);
        for (int sample = 0; sample < samples; sample++) {
            integrator.step(1);
            vector<State> states;
            for (int bead = 0; bead < copies; bead++)
                states.push_back(integrator.getState(bead, State::Velocities));
            for (int k = 1; k <= (copies-1)/2; k++) {
                for (int particle = 0; particle < numParticles; particle++) {
                    Vec3 cosine, sine;
                    for (int bead = 0; bead < copies; bead++) {
                        double angle = 2*M_PI*k*bead/copies;
                        cosine += states[bead].getVelocities()[particle]*cos(angle);
                        sine += states[bead].getVelocities()[particle]*sin(angle);
                    }
                    // Orthonormal real modes each have variance copies*kT/m.
                    cosine2[k] += (2.0/copies)*cosine.dot(cosine);
                    sine2[k] += (2.0/copies)*sine.dot(sine);
                }
            }
        }
        double expected = samples*3.0*numParticles*copies*BOLTZ*temperature/mass;
        for (int k = 1; k <= (copies-1)/2; k++) {
            ASSERT_USUALLY_EQUAL_TOL(1.0, cosine2[k]/expected, 0.05);
            ASSERT_USUALLY_EQUAL_TOL(1.0, sine2[k]/expected, 0.05);
        }
    }
}

void testMixedPositionRoundTripAndDrift() {
    const double dt = 0.001;
    const Vec3 velocity(1e-6, -2e-6, 3e-6);
    for (int copies : {1, 3, 4}) {
        System system;
        system.addParticle(1);
        // Also exercise contracted transfers and periodic cell translations.
        map<int, int> contractions;
        if (copies == 4) {
            system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
            NonbondedForce* force = new NonbondedForce();
            force->addParticle(0, 1, 0);
            force->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
            force->setCutoffDistance(1);
            system.addForce(force);
            contractions[0] = 1;
        }
        RPMDIntegrator integrator(copies, 300, 1, dt, contractions);
        integrator.setApplyThermostat(false);
        map<string, string> properties;
        properties["Precision"] = "mixed";
        Context context(system, integrator, platform, properties);
        Vec3 initial((copies == 4 ? 9.0 : 1.0)+ldexp(1.0, -26), 1.0+ldexp(1.0, -25), 1.0-ldexp(1.0, -26));
        // Different residuals on each bead expose stale Context corrections.
        for (int bead = 0; bead < copies; bead++) {
            integrator.setPositions(bead, vector<Vec3>(1, initial+Vec3(bead*1e-8, 0, 0)));
            integrator.setVelocities(bead, vector<Vec3>(1, velocity));
        }
        for (int bead = copies-1; bead >= 0; bead--) {
            Vec3 expected = initial+Vec3(bead*1e-8, 0, 0);
            State state = integrator.getState(bead, State::Positions);
            for (int component = 0; component < 3; component++)
                ASSERT_EQUAL_TOL(expected[component], state.getPositions()[0][component], 2e-12);
        }
        // Equal bead positions and velocities have only a centroid mode, so
        // the exact trajectory is free drift, even with ring-polymer springs.
        for (int bead = 0; bead < copies; bead++)
            integrator.setPositions(bead, vector<Vec3>(1, initial));
        integrator.step(100);
        for (int bead = 0; bead < copies; bead++) {
            State state = integrator.getState(bead, State::Positions | State::Velocities);
            Vec3 expected = initial+velocity*(100*dt);
            for (int component = 0; component < 3; component++) {
                ASSERT_EQUAL_TOL(expected[component], state.getPositions()[0][component], 2e-11);
                ASSERT_EQUAL_TOL(velocity[component], state.getVelocities()[0][component], 2e-11);
            }
        }
    }
}

void testContractedForceGroups() {
    // Two groups share a contraction.  Test exclusion of the whole contraction
    // and exclusion of only one of its groups, with an uncontracted control.
    for (int mask : {0, 1, 2, 3, 4, 7}) {
        System system;
        system.addParticle(2);
        for (int group = 0; group < 3; group++) {
            CustomExternalForce* force = new CustomExternalForce("-f*x");
            force->addPerParticleParameter("f");
            force->addParticle(0, {double(1<<group)});
            force->setForceGroup(group);
            system.addForce(force);
        }
        map<int, int> contractions = {{0, 1}, {1, 1}};
        RPMDIntegrator integrator(4, 300, 1, 0.001, contractions);
        integrator.setApplyThermostat(false);
        integrator.setIntegrationForceGroups(mask);
        Context context(system, integrator, platform);
        for (int bead = 0; bead < 4; bead++) {
            integrator.setPositions(bead, vector<Vec3>(1, Vec3()));
            integrator.setVelocities(bead, vector<Vec3>(1, Vec3(0.1, 0, 0)));
        }
        integrator.step(10);
        double time = 10*integrator.getStepSize();
        double acceleration = mask/2.0;
        for (int bead = 0; bead < 4; bead++) {
            State state = integrator.getState(bead, State::Positions | State::Velocities);
            Vec3 expectedVelocity(0.1+acceleration*time, 0, 0);
            Vec3 expectedPosition(0.1*time+0.5*acceleration*time*time, 0, 0);
            for (int component = 0; component < 3; component++) {
                ASSERT_EQUAL_TOL(expectedVelocity[component], state.getVelocities()[0][component], 1e-5);
                ASSERT_EQUAL_TOL(expectedPosition[component], state.getPositions()[0][component], 1e-6);
            }
        }
    }
}

void runPlatformTests() {
    testOddBeadThermostat();
    testMixedPositionRoundTripAndDrift();
    testContractedForceGroups();
    testParaHydrogen();
}

void setupKernels (int argc, char* argv[]) {
    registerRPMDMetalKernelFactories();
    platform = dynamic_cast<MetalPlatform&>(Platform::getPlatformByName("Metal"));
    initializeTests(argc, argv);
}
