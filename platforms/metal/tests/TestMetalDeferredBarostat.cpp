/* Portions copyright (c) 2026 the Authors. Licensed under the MIT license. */
#include "MetalTests.h"
#include "openmm/Context.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/MonteCarloBarostat.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VirtualSite.h"
#include "openmm/internal/AssertionUtilities.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

// Compare the object representation of each scalar, including the sign of zero.
static void assertBits(double expected, double found) {
    ASSERT(memcmp(&expected, &found, sizeof(double)) == 0);
}

static void compareVectors(const vector<Vec3>& expected, const vector<Vec3>& found) {
    ASSERT_EQUAL(expected.size(), found.size());
    for (int i = 0; i < expected.size(); i++)
        for (int j = 0; j < 3; j++)
            assertBits(expected[i][j], found[i][j]);
}

static void testTrajectory() {
    System system;
    const int side = 4, frequency = 2, steps = 400;
    const double spacing = 0.31, length = side*spacing;
    system.setDefaultPeriodicBoxVectors(Vec3(length, 0, 0), Vec3(0, length, 0), Vec3(0, 0, length));
    auto* nb = new NonbondedForce();
    nb->setNonbondedMethod(NonbondedForce::PME);
    nb->setCutoffDistance(0.5);
    nb->setEwaldErrorTolerance(1e-4);
    nb->setUseDispersionCorrection(true); // Exercise a nonzero host energy term.
    nb->setForceGroup(1);
    nb->setReciprocalSpaceForceGroup(2);
    const double oh = 0.09572, halfAngle = 104.52*acos(-1.0)/360;
    const Vec3 h1(oh*cos(halfAngle), oh*sin(halfAngle), 0);
    const Vec3 h2(oh*cos(halfAngle), -oh*sin(halfAngle), 0);
    vector<Vec3> positions;
    for (int x = 0; x < side; x++)
        for (int y = 0; y < side; y++)
            for (int z = 0; z < side; z++) {
                int first = system.getNumParticles();
                system.addParticle(15.99943);
                system.addParticle(1.007947);
                system.addParticle(1.007947);
                system.addParticle(0);
                system.setVirtualSite(first+3, new ThreeParticleAverageSite(first, first+1, first+2,
                        0.786646558, 0.106676721, 0.106676721));
                system.addConstraint(first, first+1, oh);
                system.addConstraint(first, first+2, oh);
                system.addConstraint(first+1, first+2, 2*oh*sin(halfAngle));
                nb->addParticle(0, 0.316435, 0.680946);
                nb->addParticle(0.52422, 1, 0);
                nb->addParticle(0.52422, 1, 0);
                nb->addParticle(-1.04844, 1, 0);
                for (int i = 0; i < 4; i++)
                    for (int j = i+1; j < 4; j++)
                        nb->addException(first+i, first+j, 0, 1, 0);
                Vec3 o((x+0.5)*spacing, (y+0.5)*spacing, (z+0.5)*spacing);
                positions.push_back(o);
                positions.push_back(o+h1);
                positions.push_back(o+h2);
                positions.push_back(o);
            }
    system.addForce(nb);
    auto* barostat = new MonteCarloBarostat(1.0, 300.0, frequency);
    barostat->setRandomNumberSeed(192837);
    system.addForce(barostat);
    vector<State> reference;
    // Construct contexts sequentially: the barostat's host RNG is seeded at initialization.
    for (int deferred = 0; deferred < 2; deferred++) {
        ASSERT_EQUAL(0, setenv("OPENMM_METAL_DEFER_BAROSTAT_ENERGIES", deferred ? "1" : "0", 1));
        LangevinMiddleIntegrator integrator(300, 1, 0.001);
        integrator.setRandomNumberSeed(918273);
        integrator.setConstraintTolerance(1e-6);
        integrator.setIntegrationForceGroups((1<<1) | (1<<2));
        Context context(system, integrator, platform);
        context.setPositions(positions);
        context.computeVirtualSites();
        context.setVelocitiesToTemperature(300, 564738);
        context.applyVelocityConstraints(1e-6);
        int accepted = 0, rejected = 0;
        double lastBox = length;
        for (int step = 1; step <= steps; step++) {
            integrator.step(1);
            State state = context.getState(State::Positions | State::Velocities | State::Forces | State::Energy,
                    false, integrator.getIntegrationForceGroups());
            Vec3 box[3];
            state.getPeriodicBoxVectors(box[0], box[1], box[2]);
            if (step%frequency == 0) {
                if (box[0][0] == lastBox)
                    rejected++;
                else
                    accepted++;
            }
            lastBox = box[0][0];
            if (!deferred)
                reference.push_back(state);
            else {
                const State& old = reference[step-1];
                compareVectors(old.getPositions(), state.getPositions());
                compareVectors(old.getVelocities(), state.getVelocities());
                compareVectors(old.getForces(), state.getForces());
                assertBits(old.getPotentialEnergy(), state.getPotentialEnergy());
                assertBits(old.getKineticEnergy(), state.getKineticEnergy());
                assertBits(old.getTime(), state.getTime());
                ASSERT_EQUAL(old.getStepCount(), state.getStepCount());
                Vec3 oldBox[3];
                old.getPeriodicBoxVectors(oldBox[0], oldBox[1], oldBox[2]);
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        assertBits(oldBox[i][j], box[i][j]);
            }
        }
        ASSERT(accepted > 0);
        ASSERT(rejected > 0);
    }
    unsetenv("OPENMM_METAL_DEFER_BAROSTAT_ENERGIES");
}

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        platform.setPropertyDefaultValue(MetalPlatform::MetalDeterministicForces(), "true");
        testTrajectory();
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
