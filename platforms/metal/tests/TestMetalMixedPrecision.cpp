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
 * This tests that mixed precision on Metal, which has no double precision, computes mixed values
 * as df64 (a sum of two floats) instead of silently falling back to float.
 */

#include "openmm/internal/AssertionUtilities.h"
#include "MetalArray.h"
#include "MetalContext.h"
#include "MetalPlatform.h"
#include "openmm/Context.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/CustomIntegrator.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "sfmt/SFMT.h"
#include <cmath>
#include <iostream>

using namespace OpenMM;
using namespace std;

static MetalPlatform platform;

/**
 * Check that SQRT, RSQRT, RECIP, EXP, LOG, and the other math functions kernels apply to mixed values
 * return mixed values, for every choice of fast and precise versions the accuracy probe can make.  In
 * mixed precision, also check that they are accurate to double-float precision.
 */
void testMathFunctions(const string& precision) {
    System system;
    system.addParticle(1.0);
    MetalPlatform::PlatformData platformData(NULL, system, precision, "false", platform.getPropertyDefaultValue(MetalPlatform::MetalTempDirectory()), 1, NULL);
    MetalContext& context = *platformData.contexts[0];
    const string source = R"(
#define ASSERT_TYPE(expression, type) static_assert(is_same<decltype(expression), type>::value, #expression " does not return " #type);
ASSERT_TYPE(SQRT(mixed()), mixed)
ASSERT_TYPE(RSQRT(mixed()), mixed)
ASSERT_TYPE(RECIP(mixed()), mixed)
ASSERT_TYPE(RECIP(mixed3()), mixed3)
ASSERT_TYPE(EXP(mixed()), mixed)
ASSERT_TYPE(LOG(mixed()), mixed)
ASSERT_TYPE(sqrt(mixed()), mixed)
ASSERT_TYPE(rsqrt(mixed()), mixed)
ASSERT_TYPE(exp(mixed()), mixed)
ASSERT_TYPE(log(mixed()), mixed)
ASSERT_TYPE(fabs(mixed()), mixed)
ASSERT_TYPE(floor(mixed()), mixed)
ASSERT_TYPE(ceil(mixed()), mixed)
ASSERT_TYPE(min(mixed(), mixed()), mixed)
ASSERT_TYPE(max(mixed(), mixed()), mixed)

KERNEL void evaluate(GLOBAL mixed* values) {
    mixed x = values[0];
    values[1] = SQRT(x);
    values[2] = RSQRT(x);
    values[3] = RECIP(x);
    values[4] = EXP(x);
    values[5] = LOG(x);
}
)";
    const double x = 1.2345678901234567;
    for (int flags = 0; flags < 16; flags++) {
        map<string, string> defines = MetalContext::getMathFunctionDefines(flags&1, flags&2, flags&4, flags&8);
        ComputeKernel kernel = context.compileProgram(source, defines)->createKernel("evaluate");
        if (!context.getUseMixedPrecision())
            continue;
        MetalArray values;
        values.initialize<double>(context, 6, "values");
        vector<double> results(6, 0.0);
        results[0] = x;
        values.upload(results);
        kernel->addArg(values);
        kernel->execute(1);
        values.download(results);
        ASSERT_EQUAL_TOL(sqrt(x), results[1], 1e-12);
        ASSERT_EQUAL_TOL(1.0/sqrt(x), results[2], 1e-12);
        ASSERT_EQUAL_TOL(1.0/x, results[3], 1e-12);
        ASSERT_EQUAL_TOL(exp(x), results[4], 1e-12);
        ASSERT_EQUAL_TOL(log(x), results[5], 1e-12);
    }
}

/**
 * In mixed precision, a function with no df64 version must fail to compile on a mixed value, not run in
 * single precision.  In single precision, the same program compiles.
 */
void testUnsupportedFunctions(const string& precision) {
    System system;
    system.addParticle(1.0);
    MetalPlatform::PlatformData platformData(NULL, system, precision, "false", platform.getPropertyDefaultValue(MetalPlatform::MetalTempDirectory()), 1, NULL);
    MetalContext& context = *platformData.contexts[0];
    for (string call : {"SIN(x)", "COS(x)", "TAN(x)", "ASIN(x)", "ACOS(x)", "ATAN(x)", "atan2(x, x)", "sinh(x)", "cosh(x)", "tanh(x)", "ERF(x)", "ERFC(x)", "POW(x, x)"}) {
        bool threwException = false;
        try {
            context.compileProgram("KERNEL void evaluate(GLOBAL mixed* values) {\nmixed x = values[0];\nvalues[1] = "+call+";\n}\n");
        }
        catch (const OpenMMException& ex) {
            ASSERT(string(ex.what()).find("Metal can only evaluate in single precision") != string::npos);
            threwException = true;
        }
        ASSERT_EQUAL(context.getUseMixedPrecision(), threwException);
    }
}

/**
 * Take one step with Verlet and LangevinMiddle (at zero temperature, so without noise) and compare the
 * result to the exact step computed in double on the host.  The forces are constant and exactly
 * representable in fixed point, so all of the difference comes from the integration arithmetic, which
 * in mixed precision is about 1e-15 and in single precision about 1e-7.
 */
void testIntegrationStep() {
    const int numParticles = 20;
    const double dt = 0.002;
    const double friction = 1.0;
    System system;
    CustomExternalForce* force = new CustomExternalForce("-fx*x-fy*y-fz*z");
    force->addPerParticleParameter("fx");
    force->addPerParticleParameter("fy");
    force->addPerParticleParameter("fz");
    system.addForce(force);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions, velocities, forces;
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0+0.37*i);
        forces.push_back(Vec3((i%5)-2.25, 0.5*(i%3)-0.75, 1.5-0.125*i));
        force->addParticle(i, {forces[i][0], forces[i][1], forces[i][2]});
        positions.push_back(Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*2);
        velocities.push_back(Vec3(genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5)*4);
    }
    VerletIntegrator verlet(dt);
    LangevinMiddleIntegrator langevin(0.0, friction, dt);
    for (Integrator* integrator : {(Integrator*) &verlet, (Integrator*) &langevin}) {
        Context context(system, *integrator, platform);
        context.setPositions(positions);
        context.setVelocities(velocities);
        integrator->step(1);
        State state = context.getState(State::Positions | State::Velocities | State::Energy);
        double vscale = (integrator == &verlet ? 1.0 : exp(-friction*dt));
        double ke = 0.0;
        for (int i = 0; i < numParticles; i++) {
            double mass = system.getParticleMass(i);
            Vec3 v = velocities[i]+forces[i]*(dt/mass);
            Vec3 expectedVelocity = v*vscale;
            Vec3 expectedPosition = positions[i]+(integrator == &verlet ? v*dt : (v+expectedVelocity)*(0.5*dt));
            ASSERT_EQUAL_VEC(expectedPosition, state.getPositions()[i], 1e-12);
            ASSERT_EQUAL_VEC(expectedVelocity, state.getVelocities()[i], 1e-12);
            ke += 0.5*mass*expectedVelocity.dot(expectedVelocity);
        }

        // Verlet reports the kinetic energy at a half step, shifted with a float time step on every platform.

        if (integrator == &langevin)
            ASSERT_EQUAL_TOL(ke, state.getKineticEnergy(), 1e-12);
    }
}

/**
 * Constants in custom expressions that a float cannot hold exactly must keep double-float precision.
 */
void testExpressionConstants() {
    System system;
    system.addParticle(1.0);
    CustomIntegrator integrator(0.001);
    integrator.addGlobalVariable("g", 1.3);
    integrator.addComputeGlobal("g", "0.1*g+0.7");
    integrator.addComputePerDof("v", "0.1*v+0.3");
    Context context(system, integrator, platform);
    context.setPositions({Vec3()});
    Vec3 velocity(1.1, -0.9, 0.35);
    context.setVelocities({velocity});
    integrator.step(1);
    ASSERT_EQUAL_TOL(0.1*1.3+0.7, integrator.getGlobalVariable(0), 1e-12);
    ASSERT_EQUAL_VEC(velocity*0.1+Vec3(0.3, 0.3, 0.3), context.getState(State::Velocities).getVelocities()[0], 1e-12);
}

int main(int argc, char* argv[]) {
    try {
        string precision = (argc > 1 ? argv[1] : "single");
        platform.setPropertyDefaultValue("Precision", precision);
        testMathFunctions(precision);
        testUnsupportedFunctions(precision);
        if (precision == "mixed") {
            testIntegrationStep();
            testExpressionConstants();
        }
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
