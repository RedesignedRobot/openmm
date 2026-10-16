/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
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
 * This tests the double-float arithmetic behind mixed precision on Metal.  Mixed values must carry
 * about 48 bits of mantissa, so results are checked against double to 1e-12.
 */

#include "openmm/internal/AssertionUtilities.h"
#include "MetalArray.h"
#include "MetalContext.h"
#include "openmm/Context.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/CustomIntegrator.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VariableVerletIntegrator.h"
#include "openmm/VerletIntegrator.h"
#include "sfmt/SFMT.h"
#include <climits>
#include <cmath>
#include <iostream>

using namespace OpenMM;
using namespace std;

MetalPlatform platform;

static long long saturateToLong(double x) {
    if (isnan(x))
        return 0;
    if (x >= 0x1p63)
        return LLONG_MAX;
    if (x <= -0x1p63)
        return LLONG_MIN;
    return (long long) x;
}

void testMathFunctions() {
    System system;
    system.addParticle(1.0);
    MetalPlatform::PlatformData platformData(NULL, system, "mixed", "false", "false", 1, NULL);
    MetalContext& context = *platformData.contexts[0];
    context.initialize();
    context.setAsCurrent();
    const string source = R"(
KERNEL void evaluate(GLOBAL mixed* values) {
    mixed x = values[0];
    values[1] = SQRT(x);
    values[2] = RSQRT(x);
    values[3] = RECIP(x);
    values[4] = EXP(x);
    values[5] = LOG(x);
    values[6] = SIN(x);
    values[7] = COS(x);
    values[8] = SIN(-3*x);
    values[9] = COS(-3*x);
    values[10] = SIN(100*x);
    values[11] = COS(100*x);
}
)";
    const double x = 1.2345678901234567;
    MetalArray values;
    values.initialize<double>(context, 12, "values");
    vector<double> results(12, 0.0);
    results[0] = x;
    values.upload(results);
    ComputeKernel kernel = context.compileProgram(source)->createKernel("evaluate");
    kernel->addArg(values);
    kernel->execute(1);
    values.download(results);
    ASSERT_EQUAL_TOL(sqrt(x), results[1], 1e-12);
    ASSERT_EQUAL_TOL(1.0/sqrt(x), results[2], 1e-12);
    ASSERT_EQUAL_TOL(1.0/x, results[3], 1e-12);
    ASSERT_EQUAL_TOL(exp(x), results[4], 1e-12);
    ASSERT_EQUAL_TOL(log(x), results[5], 1e-12);
    ASSERT_EQUAL_TOL(sin(x), results[6], 1e-12);
    ASSERT_EQUAL_TOL(cos(x), results[7], 1e-12);
    ASSERT_EQUAL_TOL(sin(-3*x), results[8], 1e-12);
    ASSERT_EQUAL_TOL(cos(-3*x), results[9], 1e-12);
    ASSERT_EQUAL_TOL(sin(100*x), results[10], 1e-12);
    ASSERT_EQUAL_TOL(cos(100*x), results[11], 1e-12);
}

void testLongConversion() {
    // Values from 2^63 up are out of range and must saturate on every chip.  The M3 wraps a plain
    // cast modulo 2^64.

    System system;
    system.addParticle(1.0);
    MetalPlatform::PlatformData platformData(NULL, system, "mixed", "false", "false", 1, NULL);
    MetalContext& context = *platformData.contexts[0];
    context.initialize();
    context.setAsCurrent();
    const string source = R"(
KERNEL void convert(GLOBAL mixed* values, GLOBAL mm_long* results, int numValues) {
    for (int i = GLOBAL_ID; i < numValues; i += GLOBAL_SIZE) {
        mixed x = values[i];
        values[i] = x;
        results[i] = (mm_long) x;
    }
}
)";
    vector<double> exact = {0.0, 0.75, 1.5, 3.0-pow(2.0, -30), pow(2.0, 24)-0.5, pow(2.0, 31)+0.25, pow(2.0, 40)+0.75,
            1234.5*pow(2.0, 32), pow(2.0, 52)-0.5, pow(2.0, 53)-1, pow(2.0, 53)+2, pow(2.0, 62)+pow(2.0, 40),
            pow(2.0, 63)-1024, pow(2.0, 63)-pow(2.0, 40), pow(2.0, 63), pow(2.0, 63)+pow(2.0, 41), pow(2.0, 64), pow(2.0, 100)};
    vector<double> values;
    for (double x : exact) {
        values.push_back(x);
        values.push_back(-x);
    }
    int numExact = values.size();
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < 1000; i++)
        values.push_back((genrand_real2(sfmt) < 0.5 ? -1 : 1)*(1+genrand_real2(sfmt))*pow(2.0, 61*genrand_real2(sfmt)));
    for (double x : {1e30, -1e30, (double) INFINITY, (double) -INFINITY, (double) NAN})
        values.push_back(x);
    int numValues = values.size();
    MetalArray valuesArray, resultsArray;
    valuesArray.initialize<double>(context, numValues, "values");
    resultsArray.initialize<long long>(context, numValues, "results");
    valuesArray.upload(values);
    ComputeKernel kernel = context.compileProgram(source)->createKernel("convert");
    kernel->addArg(valuesArray);
    kernel->addArg(resultsArray);
    kernel->addArg(numValues);
    kernel->execute(numValues);
    vector<double> stored;
    vector<long long> results;
    valuesArray.download(stored);
    resultsArray.download(results);
    for (int i = 0; i < numValues; i++) {
        if (i < numExact) {
            ASSERT_EQUAL(values[i], stored[i]);
        }
        else if (isfinite(values[i])) {
            ASSERT_EQUAL_TOL(values[i], stored[i], 1e-14);
        }
        ASSERT_EQUAL(saturateToLong(stored[i]), results[i]);
    }
}

void testUnsupportedFunctions() {
    // These have no double-float version, so a mixed kernel that uses one must fail to compile
    // instead of silently running in float.

    System system;
    system.addParticle(1.0);
    MetalPlatform::PlatformData platformData(NULL, system, "mixed", "false", "false", 1, NULL);
    MetalContext& context = *platformData.contexts[0];
    context.initialize();
    context.setAsCurrent();
    for (string call : {"TAN(x)", "ASIN(x)", "ACOS(x)", "ATAN(x)", "atan2(x, x)", "sinh(x)", "cosh(x)", "tanh(x)", "ERF(x)", "ERFC(x)", "POW(x, x)"}) {
        bool threwException = false;
        try {
            context.compileProgram("KERNEL void evaluate(GLOBAL mixed* values) {\nmixed x = values[0];\nvalues[1] = "+call+";\n}\n");
        }
        catch (const OpenMMException& ex) {
            ASSERT(string(ex.what()).find("Error compiling program: ") == 0);
            threwException = true;
        }
        ASSERT(threwException);
    }
}

void createFreeParticles(System& system, vector<Vec3>& positions, vector<Vec3>& velocities, vector<Vec3>& forces) {
    const int numParticles = 20;
    CustomExternalForce* force = new CustomExternalForce("-fx*x-fy*y-fz*z");
    force->addPerParticleParameter("fx");
    force->addPerParticleParameter("fy");
    force->addPerParticleParameter("fz");
    system.addForce(force);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0+0.37*i);
        forces.push_back(Vec3((i%5)-2.25, 0.5*(i%3)-0.75, 1.5-0.125*i));
        force->addParticle(i, {forces[i][0], forces[i][1], forces[i][2]});
        positions.push_back(Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*2);
        velocities.push_back(Vec3(genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5)*4);
    }
}

void testIntegrationStep() {
    // Constant forces make one step exact, so positions, velocities and kinetic energy must match
    // the analytic step to 1e-12.

    const double dt = 0.002;
    const double friction = 1.0;
    System system;
    vector<Vec3> positions, velocities, forces;
    createFreeParticles(system, positions, velocities, forces);
    const int numParticles = system.getNumParticles();
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
        if (integrator == &langevin)
            ASSERT_EQUAL_TOL(ke, state.getKineticEnergy(), 1e-12);
    }
}

void testVariableStep() {
    const double errorTol = 1e-6;
    System system;
    vector<Vec3> positions, velocities, forces;
    createFreeParticles(system, positions, velocities, forces);
    const int numParticles = system.getNumParticles();
    VariableVerletIntegrator integrator(errorTol);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocities(velocities);
    integrator.step(1);
    State state = context.getState(State::Positions | State::Velocities);
    double error = 0.0;
    for (int i = 0; i < numParticles; i++) {
        Vec3 acceleration = forces[i]/system.getParticleMass(i);
        error += acceleration.dot(acceleration);
    }
    double dt = sqrt(errorTol/sqrt(error/(3*numParticles)));
    ASSERT_EQUAL_TOL(1.0, integrator.getStepSize()/dt, 1e-12);
    for (int i = 0; i < numParticles; i++) {
        Vec3 velocity = velocities[i]+forces[i]*(0.5*dt/system.getParticleMass(i));
        ASSERT_EQUAL_VEC(positions[i]+velocity*dt, state.getPositions()[i], 1e-12);
        ASSERT_EQUAL_VEC(velocity, state.getVelocities()[i], 1e-12);
    }
}

void testExpressionConstants() {
    System system;
    system.addParticle(1.0);
    CustomIntegrator integrator(0.001);
    integrator.addGlobalVariable("g", 1.3);
    integrator.addComputePerDof("v", "0.1*v+g");
    Context context(system, integrator, platform);
    context.setPositions({Vec3()});
    Vec3 velocity(1.1, -0.9, 0.35);
    context.setVelocities({velocity});
    integrator.step(1);
    ASSERT_EQUAL_VEC(velocity*0.1+Vec3(1.3, 1.3, 1.3), context.getState(State::Velocities).getVelocities()[0], 1e-12);
}

void testConstantMatchesUploadedValue() {
    // This value rounds to an odd float hi with value-hi rounding to exactly half an ulp.  A constant in an
    // expression must split it the same way as an uploaded value.

    const double value = 1+pow(2.0, -23)+pow(2.0, -24)-pow(2.0, -52);
    ASSERT_EQUAL(value, 1.000000178813934);
    System system;
    system.addParticle(1.0);
    CustomIntegrator integrator(0.001);
    integrator.addPerDofVariable("v0", 0);
    integrator.addPerDofVariable("diff", 1);
    integrator.addComputePerDof("diff", "1.000000178813934-v0");
    Context context(system, integrator, platform);
    context.setPositions({Vec3()});
    integrator.setPerDofVariableByName("v0", {Vec3(value, value, value)});
    integrator.step(1);
    vector<Vec3> diff;
    integrator.getPerDofVariableByName("diff", diff);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), diff[0], 0);
}

void testReorderWrapKeepsPositions() {
    // Reordering moves each molecule whose center is outside the periodic box back inside.  In mixed
    // precision that shift must keep the bits stored in posqCorrection.

    const int numParticles = 64;
    const double width = 7.3;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(width, 0, 0), Vec3(0, width, 0), Vec3(0, 0, width));
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    nonbonded->setCutoffDistance(1.0);
    system.addForce(nonbonded);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions;
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(0.0, 0.3, 0.0);
        positions.push_back(Vec3(-0.5*genrand_real2(sfmt), width+0.5*genrand_real2(sfmt), width*genrand_real2(sfmt)));
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    vector<Vec3> found = context.getState(State::Positions).getPositions();
    for (int i = 0; i < numParticles; i++)
        ASSERT_EQUAL_VEC(positions[i], found[i], 1e-12);
}

void testSupportedIntegratorExpressions() {
    // openmmtools-style integrators: BAOAB Langevin splitting, and a GHMC step that sums the kinetic
    // energy and uses integer powers, min, abs, step and select per DOF.

    System system;
    CustomExternalForce* force = new CustomExternalForce("x^2+y^2+z^2");
    system.addForce(force);
    for (int i = 0; i < 4; i++) {
        system.addParticle(1.0+i);
        force->addParticle(i);
    }
    vector<Vec3> positions = {Vec3(0.1, 0.2, 0.3), Vec3(-0.3, 0.1, 0.2), Vec3(0.2, -0.1, 0.4), Vec3(0.5, 0.3, -0.2)};
    CustomIntegrator baoab(0.002);
    baoab.addGlobalVariable("a", exp(-0.002));
    baoab.addGlobalVariable("b", sqrt(1-exp(-0.004)));
    baoab.addGlobalVariable("kT", 2.49);
    baoab.addPerDofVariable("x1", 0);
    baoab.addUpdateContextState();
    baoab.addComputePerDof("v", "v+0.5*dt*f/m");
    baoab.addComputePerDof("x", "x+0.5*dt*v");
    baoab.addComputePerDof("v", "a*v+b*sqrt(kT/m)*gaussian");
    baoab.addComputePerDof("x", "x+0.5*dt*v");
    baoab.addComputePerDof("x1", "x");
    baoab.addConstrainPositions();
    baoab.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt");
    CustomIntegrator ghmc(0.002);
    ghmc.addGlobalVariable("kT", 2.49);
    ghmc.addGlobalVariable("ke", 0);
    ghmc.addGlobalVariable("Eold", 0);
    ghmc.addGlobalVariable("Enew", 0);
    ghmc.addGlobalVariable("accept", 0);
    ghmc.addPerDofVariable("xold", 0);
    ghmc.addComputePerDof("v", "sqrt(kT/m)*gaussian");
    ghmc.addComputeSum("ke", "0.5*m*v*v");
    ghmc.addComputeGlobal("Eold", "ke+energy");
    ghmc.addComputePerDof("xold", "x");
    ghmc.addComputePerDof("v", "v+0.5*dt*f/m");
    ghmc.addComputePerDof("x", "x+dt*v*(1+0.1*min(abs(x), 1)^2*step(x)/(1+x^4))");
    ghmc.addComputePerDof("v", "v+0.5*dt*f/m");
    ghmc.addComputeSum("ke", "0.5*m*v^2");
    ghmc.addComputeGlobal("Enew", "ke+energy");
    ghmc.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT)-uniform)");
    ghmc.addComputePerDof("x", "select(accept, x, xold)");
    for (CustomIntegrator* integrator : {&baoab, &ghmc}) {
        Context context(system, *integrator, platform);
        context.setPositions(positions);
        integrator->step(10);
        State state = context.getState(State::Positions | State::Energy);
        ASSERT(isfinite(state.getPotentialEnergy()));
        for (const Vec3& position : state.getPositions())
            ASSERT(isfinite(position[0]) && isfinite(position[1]) && isfinite(position[2]));
    }
}

int main(int argc, char* argv[]) {
    try {
        string precision = (argc > 1 ? argv[1] : "single");
        if (precision != "mixed") {
            cout << "Done" << endl;
            return 0;
        }
        platform.setPropertyDefaultValue("Precision", precision);
        testMathFunctions();
        testLongConversion();
        testUnsupportedFunctions();
        testIntegrationStep();
        testVariableStep();
        testExpressionConstants();
        testConstantMatchesUploadedValue();
        testReorderWrapKeepsPositions();
        testSupportedIntegratorExpressions();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
