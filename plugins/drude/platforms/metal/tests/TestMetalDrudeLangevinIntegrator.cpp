/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2013 Stanford University and the Authors.           *
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

#include "MetalDrudeTests.h"
#include "TestDrudeLangevinIntegrator.h"
#include "openmm/CustomExternalForce.h"
#include <cmath>

void testIndependentThermostatNoise() {
    // Move the normal particle through all three slots.  Slots 1 and 2 used to
    // share noise with the pair's center of mass and relative mode, respectively.
    for (int normal = 0; normal < 3; normal++) {
        int parent = (normal+1)%3;
        int drude = (normal+2)%3;
        System system;
        for (int i = 0; i < 3; i++)
            system.addParticle(1.0);
        DrudeForce* force = new DrudeForce();
        force->addParticle(drude, parent, -1, -1, -1, 0.1, 1.0, 1.0, 1.0);
        system.addForce(force);
        DrudeLangevinIntegrator integrator(300, 10, 10, 20, 0.001);
        integrator.setRandomNumberSeed(1234);
        integrator.setMaxDrudeDistance(0);
        Context context(system, integrator, platform);
        vector<Vec3> zero(3, Vec3());
        double normal2 = 0, cm2 = 0, relative2 = 0, normalCM = 0, normalRelative = 0;
        for (int sample = 0; sample < 1000; sample++) {
            // Each sample starts with zero spring force and zero velocity.
            context.setPositions(zero);
            context.setVelocities(zero);
            integrator.step(1);
            State state = context.getState(State::Velocities);
            const vector<Vec3>& v = state.getVelocities();
            Vec3 cm = (v[parent]+v[drude])*0.5;
            Vec3 relative = v[drude]-v[parent];
            normal2 += v[normal].dot(v[normal]);
            cm2 += cm.dot(cm);
            relative2 += relative.dot(relative);
            normalCM += v[normal].dot(cm);
            normalRelative += v[normal].dot(relative);
        }
        ASSERT(normal2 > 0 && cm2 > 0 && relative2 > 0);
        ASSERT_USUALLY_EQUAL_TOL(0.0, normalCM/sqrt(normal2*cm2), 0.1);
        ASSERT_USUALLY_EQUAL_TOL(0.0, normalRelative/sqrt(normal2*relative2), 0.1);
    }
}

void testZeroFriction() {
    // Zero friction in either thermostat, both together, and the near-zero limit.
    const double frictions[][2] = {{0, 5}, {5, 0}, {0, 0}, {1e-14, 1e-14}};
    const double dt = 0.002;
    for (auto& friction : frictions) {
        System system;
        system.addParticle(2.0);
        system.addParticle(1.0);
        system.addParticle(3.0);
        DrudeForce* drude = new DrudeForce();
        drude->addParticle(1, 0, -1, -1, -1, 0.1, 1.0, 1.0, 1.0);
        system.addForce(drude);
        CustomExternalForce* external = new CustomExternalForce("-fx*x-fy*y-fz*z");
        external->addPerParticleParameter("fx");
        external->addPerParticleParameter("fy");
        external->addPerParticleParameter("fz");
        vector<Vec3> forces = {Vec3(1, 2, -1), Vec3(-2, 1, 3), Vec3(3, -2, 1)};
        for (int i = 0; i < 3; i++)
            external->addParticle(i, {forces[i][0], forces[i][1], forces[i][2]});
        system.addForce(external);
        DrudeLangevinIntegrator integrator(friction[0] == 0 ? 300 : 0, friction[0], friction[1] == 0 ? 10 : 0, friction[1], dt);
        integrator.setRandomNumberSeed(1234);
        integrator.setMaxDrudeDistance(0);
        Context context(system, integrator, platform);
        vector<Vec3> initial = {Vec3(0.1, 0.2, 0.3), Vec3(0.3, -0.1, 0.2), Vec3(0.2, 0.1, -0.2)};
        context.setPositions(vector<Vec3>(3, Vec3()));
        context.setVelocities(initial);
        integrator.step(1);
        State state = context.getState(State::Positions | State::Velocities);
        double damping = exp(-friction[0]*dt);
        double drudeDamping = exp(-friction[1]*dt);
        double response = (friction[0] == 0 ? dt : -expm1(-friction[0]*dt)/friction[0]);
        double drudeResponse = (friction[1] == 0 ? dt : -expm1(-friction[1]*dt)/friction[1]);
        Vec3 cm = (initial[0]*2+initial[1])*(damping/3)+(forces[0]+forces[1])*(response/3);
        Vec3 relative = (initial[1]-initial[0])*drudeDamping+(forces[1]-forces[0]*0.5)*drudeResponse;
        vector<Vec3> expected = {cm-relative/3, cm+relative*(2.0/3), initial[2]*damping+forces[2]*(response/3)};
        for (int i = 0; i < 3; i++) {
            for (int component = 0; component < 3; component++) {
                ASSERT(std::isfinite(state.getPositions()[i][component]));
                ASSERT(std::isfinite(state.getVelocities()[i][component]));
            }
            ASSERT_EQUAL_VEC(expected[i], state.getVelocities()[i], 1e-5);
            ASSERT_EQUAL_VEC(expected[i]*dt, state.getPositions()[i], 1e-6);
        }
    }
}

void runPlatformTests() {
    testIndependentThermostatNoise();
    testZeroFriction();
}
