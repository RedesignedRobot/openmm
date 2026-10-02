/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2015 Stanford University and the Authors.           *
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
#include "TestCustomGBForce.h"
#include <cstdio>

// The OBC2 CustomGBForce openmm.app writes (customgbforces.GBSAOBC2Force, SA 'ACE' or None, kappa 0), which the Metal
// platform runs through its GBSAOBCForce kernels.  tanhCoefficient 4.85 is the template; any other value is a
// different model that must take the generic CustomGB path.
CustomGBForce* createAppOBC2(CustomGBForce::NonbondedMethod method, double cutoff, bool ace, double tanhCoefficient) {
    char number[40];
    string params = "; solventDielectric=78.5; soluteDielectric=1; kappa=0; offset=";
    snprintf(number, sizeof(number), "%.16g", 0.009);
    params += number;
    if (method != CustomGBForce::NoCutoff) {
        snprintf(number, sizeof(number), "%.16g", cutoff);
        params += string("; cutoff=")+number;
    }
    snprintf(number, sizeof(number), "%g", tanhCoefficient);
    CustomGBForce* force = new CustomGBForce();
    force->addPerParticleParameter("charge");
    force->addPerParticleParameter("or");
    force->addPerParticleParameter("sr");
    force->addComputedValue("I", "select(step(r+sr2-or1), 0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r), 0);"
                                 "U=r+sr2;L=max(or1, D);D=abs(r-sr2)", CustomGBForce::ParticlePairNoExclusions);
    force->addComputedValue("B", string("1/(1/or-tanh(psi-0.8*psi^2+")+number+"*psi^3)/radius);psi=I*or; radius=or+offset; offset=0.009",
                            CustomGBForce::SingleParticle);
    force->addEnergyTerm("-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*charge^2/B"+params, CustomGBForce::SingleParticle);
    if (ace)
        force->addEnergyTerm("28.3919551*(radius+0.14)^2*(radius/B)^6; radius=or+offset"+params, CustomGBForce::SingleParticle);
    string pair = "-138.935485*(1/soluteDielectric-1/solventDielectric)*charge1*charge2";
    string f = ";f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))";
    if (method == CustomGBForce::NoCutoff)
        force->addEnergyTerm(pair+"/f"+f+params, CustomGBForce::ParticlePairNoExclusions);
    else
        force->addEnergyTerm(pair+"*(1/f-0.5)"+f+params, CustomGBForce::ParticlePairNoExclusions);  // Python's str(1/2.0)
    force->setNonbondedMethod(method);
    force->setCutoffDistance(cutoff);
    return force;
}

State computeState(const System& system, const vector<Vec3>& positions, Platform& statePlatform) {
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, statePlatform);
    context.setPositions(positions);
    return context.getState(State::Forces | State::Energy);
}

/**
 * Check the Metal route for openmm.app's OBC2 CustomGBForce: the template must give bitwise the forces and energy of
 * the equivalent GBSAOBCForce on this platform (the route ran), agree with Reference running the CustomGBForce, and
 * keep both true after updateParametersInContext.  A changed expression must take the generic path and still agree.
 */
void testAppOBC2Route(CustomGBForce::NonbondedMethod method, bool ace) {
    const int gridSize = 5;
    const int numParticles = gridSize*gridSize*gridSize;
    const double spacing = 0.3, boxSize = 4.0, cutoff = 2.0;
    Platform& reference = Platform::getPlatformByName("Reference");
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    System customSystem, obcSystem, variantSystem;
    CustomGBForce* custom = createAppOBC2(method, cutoff, ace, 4.85);
    CustomGBForce* variant = createAppOBC2(method, cutoff, ace, 4.9);
    GBSAOBCForce* obc = new GBSAOBCForce();
    obc->setNonbondedMethod((GBSAOBCForce::NonbondedMethod) method);
    obc->setCutoffDistance(cutoff);
    obc->setSurfaceAreaEnergy(ace ? 28.3919551/(4*M_PI) : 0.0);
    obc->setSolventDielectric(78.5);  // the template's; GBSAOBCForce defaults to 78.3
    obc->setSoluteDielectric(1.0);
    vector<Vec3> positions;
    for (int i = 0; i < numParticles; i++) {
        for (System* system : {&customSystem, &obcSystem, &variantSystem}) {
            system->addParticle(1.0);
            system->setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
        }
        double charge = (i%2 == 0 ? 0.5 : -0.5)*(0.5+genrand_real2(sfmt));
        double offsetRadius = 0.111+0.05*genrand_real2(sfmt);
        double scaledRadius = (0.7+0.25*genrand_real2(sfmt))*offsetRadius;
        custom->addParticle({charge, offsetRadius, scaledRadius});
        variant->addParticle({charge, offsetRadius, scaledRadius});
        obc->addParticle(charge, offsetRadius+0.009, scaledRadius/offsetRadius);
        int x = i/(gridSize*gridSize), y = (i/gridSize)%gridSize, z = i%gridSize;
        positions.push_back(Vec3(x, y, z)*spacing + Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*0.1);
    }
    customSystem.addForce(custom);
    variantSystem.addForce(variant);
    obcSystem.addForce(obc);

    for (int update = 0; update < 2; update++) {
        State customState = computeState(customSystem, positions, platform);
        State obcState = computeState(obcSystem, positions, platform);
        State referenceState = computeState(customSystem, positions, reference);
        ASSERT(customState.getPotentialEnergy() == obcState.getPotentialEnergy());
        ASSERT_EQUAL_TOL(referenceState.getPotentialEnergy(), customState.getPotentialEnergy(), 1e-5);
        for (int i = 0; i < numParticles; i++) {
            ASSERT(customState.getForces()[i] == obcState.getForces()[i]);
            ASSERT_EQUAL_VEC(referenceState.getForces()[i], customState.getForces()[i], 1e-4);
        }
        State variantState = computeState(variantSystem, positions, platform);
        State variantReference = computeState(variantSystem, positions, reference);
        ASSERT_EQUAL_TOL(variantReference.getPotentialEnergy(), variantState.getPotentialEnergy(), 1e-5);
        for (int i = 0; i < numParticles; i++)
            ASSERT_EQUAL_VEC(variantReference.getForces()[i], variantState.getForces()[i], 1e-4);

        // Scale the charges and radii, and check the parameter update path the same way.
        VerletIntegrator integrator(0.001);
        Context context(customSystem, integrator, platform);
        context.setPositions(positions);
        for (int i = 0; i < numParticles; i++) {
            vector<double> params;
            custom->getParticleParameters(i, params);
            params[0] *= 0.8;
            params[1] *= 1.05;
            params[2] *= 1.05;
            custom->setParticleParameters(i, params);
            obc->setParticleParameters(i, params[0], params[1]+0.009, params[2]/params[1]);
        }
        custom->updateParametersInContext(context);
        State updated = context.getState(State::Forces | State::Energy);
        State updatedObc = computeState(obcSystem, positions, platform);
        ASSERT(updated.getPotentialEnergy() == updatedObc.getPotentialEnergy());
        for (int i = 0; i < numParticles; i++)
            ASSERT(updated.getForces()[i] == updatedObc.getForces()[i]);
    }
}

void runPlatformTests() {
    for (bool ace : {true, false}) {
        testAppOBC2Route(CustomGBForce::NoCutoff, ace);
        testAppOBC2Route(CustomGBForce::CutoffNonPeriodic, ace);
        testAppOBC2Route(CustomGBForce::CutoffPeriodic, ace);
    }
}
