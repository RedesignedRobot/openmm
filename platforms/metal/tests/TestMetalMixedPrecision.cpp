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

#include "openmm/internal/AssertionUtilities.h"
#include "MetalArray.h"
#include "MetalContext.h"
#include "MetalPlatform.h"
#include "openmm/Context.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/CustomIntegrator.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/System.h"
#include "openmm/VariableVerletIntegrator.h"
#include "openmm/VerletIntegrator.h"
#include "sfmt/SFMT.h"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace OpenMM;
using namespace std;

static MetalPlatform platform;

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

    vector<double> exact = {0.0, 0.75, 1.5, 3.0-pow(2.0, -30), pow(2.0, 24)-0.5, pow(2.0, 31)+0.25, pow(2.0, 40)+0.75,
            1234.5*pow(2.0, 32), pow(2.0, 52)-0.5, pow(2.0, 53)-1, pow(2.0, 53)+2, pow(2.0, 62)+pow(2.0, 40)};
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
        else {
            ASSERT_EQUAL_TOL(values[i], stored[i], 1e-14);
        }
        ASSERT_EQUAL((long long) stored[i], results[i]);
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

int main(int argc, char* argv[]) {
    try {
        string precision = (argc > 1 ? argv[1] : "single");
        platform.setPropertyDefaultValue("Precision", precision);
        testMathFunctions(precision);
        testUnsupportedFunctions(precision);
        testDoublePrecisionRejected();
        if (precision == "mixed") {
            testLongConversion();
            testIntegrationStep();
            testVariableStep();
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
