/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * Authors: Amir Ayub                                                         *
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

#include "openmm/internal/AssertionUtilities.h"
#include "MetalArray.h"
#include "MetalContext.h"
#include "MetalPlatform.h"
#include "openmm/Context.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/CustomIntegrator.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VariableVerletIntegrator.h"
#include "openmm/VerletIntegrator.h"
#include "sfmt/SFMT.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>

using namespace OpenMM;
using namespace std;

static MetalPlatform platform;

static long long saturateToLong(double x) {
    if (isnan(x))
        return 0;
    if (x >= 0x1p63)
        return LLONG_MAX;
    if (x <= -0x1p63)
        return LLONG_MIN;
    return (long long) x;
}

void testMathFunctions(const string& precision) {
    System system;
    system.addParticle(1.0);
    MetalPlatform::PlatformData platformData(NULL, system, "", "true", precision, "false", platform.getPropertyDefaultValue(MetalPlatform::MetalTempDirectory()), "false", "false", 1, NULL);
    MetalContext& context = *platformData.contexts[0];
    const string source = R"(
#define ASSERT_TYPE(expression, type) static_assert(is_same<decltype(expression), type>::value, #expression " does not return " #type);
ASSERT_TYPE(SQRT(mixed()), mixed)
ASSERT_TYPE(RSQRT(mixed()), mixed)
ASSERT_TYPE(RECIP(mixed()), mixed)
ASSERT_TYPE(RECIP(mixed3()), mixed3)
ASSERT_TYPE(EXP(mixed()), mixed)
ASSERT_TYPE(LOG(mixed()), mixed)
ASSERT_TYPE(SIN(mixed()), mixed)
ASSERT_TYPE(COS(mixed()), mixed)
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
    values[6] = SIN(x);
    values[7] = COS(x);
    values[8] = SIN(-3*x);
    values[9] = COS(-3*x);
    values[10] = SIN(100*x);
    values[11] = COS(100*x);
}
)";
    const double x = 1.2345678901234567;
    ComputeKernel kernel = context.compileProgram(source)->createKernel("evaluate");
    if (context.getUseMixedPrecision()) {
        MetalArray values;
        values.initialize<double>(context, 12, "values");
        vector<double> results(12, 0.0);
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
        ASSERT_EQUAL_TOL(sin(x), results[6], 1e-12);
        ASSERT_EQUAL_TOL(cos(x), results[7], 1e-12);
        ASSERT_EQUAL_TOL(sin(-3*x), results[8], 1e-12);
        ASSERT_EQUAL_TOL(cos(-3*x), results[9], 1e-12);
        ASSERT_EQUAL_TOL(sin(100*x), results[10], 1e-12);
        ASSERT_EQUAL_TOL(cos(100*x), results[11], 1e-12);
    }
}

void testLongConversion() {
    System system;
    system.addParticle(1.0);
    MetalPlatform::PlatformData platformData(NULL, system, "", "true", "mixed", "false", platform.getPropertyDefaultValue(MetalPlatform::MetalTempDirectory()), "false", "false", 1, NULL);
    MetalContext& context = *platformData.contexts[0];
    const string source = R"(
KERNEL void convert(GLOBAL mixed* values, GLOBAL mm_long* results, int numValues) {
    for (int i = GLOBAL_ID; i < numValues; i += GLOBAL_SIZE) {
        mixed x = values[i];
        values[i] = x;
        results[i] = (mm_long) x;
    }
}
)";

    // Values from 2^63 up are out of range and must saturate, as they do on the M2.  The M3 wraps a plain
    // cast modulo 2^64 (#5434).
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

void testFixedPointConversion(const string& precision) {
    // The minimizer finds forces too large for fixed point by looking for values beyond 2^62, so
    // realToFixedPoint() has to saturate on every chip.

    System system;
    system.addParticle(1.0);
    MetalPlatform::PlatformData platformData(NULL, system, "", "true", precision, "false", platform.getPropertyDefaultValue(MetalPlatform::MetalTempDirectory()), "false", "false", 1, NULL);
    MetalContext& context = *platformData.contexts[0];
    const string source = R"(
KERNEL void convert(GLOBAL const float* values, GLOBAL mm_long* results, int numValues) {
    for (int i = GLOBAL_ID; i < numValues; i += GLOBAL_SIZE)
        results[i] = realToFixedPoint(values[i]);
}
)";
    vector<float> values;
    for (float x : {0.0f, 0.75f, 1.5f, 1234.5678f, 0x1p31f-128, 0x1p31f, 0x1p40f, 1e30f, INFINITY}) {
        values.push_back(x);
        values.push_back(-x);
    }
    int numValues = values.size();
    MetalArray valuesArray, resultsArray;
    valuesArray.initialize<float>(context, numValues, "values");
    resultsArray.initialize<long long>(context, numValues, "results");
    valuesArray.upload(values);
    ComputeKernel kernel = context.compileProgram(source)->createKernel("convert");
    kernel->addArg(valuesArray);
    kernel->addArg(resultsArray);
    kernel->addArg(numValues);
    kernel->execute(numValues);
    vector<long long> results;
    resultsArray.download(results);
    const long long limit = 0x4000000000000000LL;
    for (int i = 0; i < numValues; i++) {
        double scaled = values[i]*0x1p32;
        if (fabs(scaled) < 0x1p63) {
            ASSERT_EQUAL((long long) scaled, results[i]);
        }
        else if (scaled > 0) {
            ASSERT(results[i] > limit);
        }
        else {
            ASSERT(results[i] < -limit);
        }
    }
}

void testUnsupportedFunctions(const string& precision) {
    System system;
    system.addParticle(1.0);
    MetalPlatform::PlatformData platformData(NULL, system, "", "true", precision, "false", platform.getPropertyDefaultValue(MetalPlatform::MetalTempDirectory()), "false", "false", 1, NULL);
    MetalContext& context = *platformData.contexts[0];
    for (string call : {"TAN(x)", "ASIN(x)", "ACOS(x)", "ATAN(x)", "atan2(x, x)", "sinh(x)", "cosh(x)", "tanh(x)", "ERF(x)", "ERFC(x)", "POW(x, x)"}) {
        string function = call.substr(0, call.find('('));
        transform(function.begin(), function.end(), function.begin(), ::tolower);
        bool threwException = false;
        try {
            context.compileProgram("KERNEL void evaluate(GLOBAL mixed* values) {\nmixed x = values[0];\nvalues[1] = "+call+";\n}\n");
        }
        catch (const OpenMMException& ex) {
            string message = ex.what();
            ASSERT(message.find("Error compiling program: ") == 0);
            threwException = true;
        }
        ASSERT_EQUAL(context.getUseMixedPrecision(), threwException);
    }

    try {
        context.compileProgram("KERNEL void evaluate(GLOBAL mixed* values) {\nmixed x = values[0];\nvalues[1] = SIN(x)+undefinedValue;\n}\n");
        ASSERT(false);
    }
    catch (const OpenMMException& ex) {
        string message = ex.what();
        ASSERT(message.find("Error compiling program: program_source:") == 0);
        ASSERT(message.find("undefinedValue") != string::npos);
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

void testDoublePrecisionRejected() {
    ASSERT(!platform.supportsDoublePrecision());
    System system;
    system.addParticle(1.0);
    VerletIntegrator integrator(0.001);
    map<string, string> properties = {{"Precision", "double"}};
    try {
        Context context(system, integrator, platform, properties);
    }
    catch (const OpenMMException& ex) {
        ASSERT(string(ex.what()).find("Precision=double") != string::npos);
        return;
    }
    throw OpenMMException("Precision=double was accepted");
}

void testReorderWrapKeepsPositions() {
    // Reordering moves each molecule whose center is outside the periodic box back inside.  In mixed
    // precision that shift must keep the bits stored in posqCorrection, or every wrapped atom moves by
    // up to half a float ulp of the box size and constraints break by about 1e-5 until the next step.

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

void testUnsupportedIntegratorExpressions(const string& precision) {
    // Mixed precision has no double-single tanh or non-integer pow, so a per-DOF expression that needs one must be
    // refused when the Context is created, naming the operation and the expression.  Globals run on the host
    // in double and may use them.

    System system;
    system.addParticle(1.0);
    for (string expression : {"v+tanh(x)", "v+abs(x)^1.5"}) {
        CustomIntegrator integrator(0.001);
        integrator.addComputePerDof("v", expression);
        bool threwException = false;
        try {
            Context context(system, integrator, platform);
            context.setPositions({Vec3(0.1, 0.2, 0.3)});
            integrator.step(1);
        }
        catch (const OpenMMException& ex) {
            string message = ex.what();
            ASSERT(message.find(expression.find("tanh") != string::npos ? "evaluate tanh" : "evaluate ^1.5") != string::npos);
            ASSERT(message.find("'"+expression+"'") != string::npos);
            threwException = true;
        }
        ASSERT_EQUAL(precision == "mixed", threwException);
    }
}

void testSupportedIntegratorExpressions() {
    // openmmtools-style integrators: BAOAB Langevin splitting, and a GHMC step that sums the kinetic energy, draws
    // an acceptance in globals that use tanh, erf and a non-integer power, and uses integer powers per DOF.

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
    ghmc.addGlobalVariable("lambda", 0);
    ghmc.addPerDofVariable("xold", 0);
    ghmc.addComputeGlobal("lambda", "0.5*(1+tanh(2*uniform-1))*erf(0.7)^1.5");
    ghmc.addComputePerDof("v", "sqrt(kT/m)*gaussian");
    ghmc.addComputeSum("ke", "0.5*m*v*v");
    ghmc.addComputeGlobal("Eold", "ke+energy");
    ghmc.addComputePerDof("xold", "x");
    ghmc.addComputePerDof("v", "v+0.5*dt*f/m");
    ghmc.addComputePerDof("x", "x+dt*v*(1+lambda*min(abs(x), 1)^2*step(x)/(1+x^4))");
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
        ASSERT(std::isfinite(state.getPotentialEnergy()));
        for (const Vec3& position : state.getPositions())
            ASSERT(std::isfinite(position[0]) && std::isfinite(position[1]) && std::isfinite(position[2]));
    }
}

int main(int argc, char* argv[]) {
    try {
        string precision = (argc > 1 ? argv[1] : "single");
        platform.setPropertyDefaultValue("Precision", precision);
        testMathFunctions(precision);
        testUnsupportedFunctions(precision);
        testFixedPointConversion(precision);
        testDoublePrecisionRejected();
        testUnsupportedIntegratorExpressions(precision);
        if (precision == "mixed") {
            testLongConversion();
            testIntegrationStep();
            testVariableStep();
            testExpressionConstants();
            testReorderWrapKeepsPositions();
            testSupportedIntegratorExpressions();
        }
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
