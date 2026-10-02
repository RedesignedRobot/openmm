/* Portions copyright (c) 2026 the Authors. Licensed under the MIT license. */
#include "MetalContext.h"
#include "MetalNonbondedUtilities.h"
#include "MetalPlatform.h"
#include "openmm/Context.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/MonteCarloBarostat.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VirtualSite.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <vector>

using namespace OpenMM;
using namespace std;

class TestPlatform : public MetalPlatform {
public:
    MetalContext& metal(Context& context) const {
        return *static_cast<PlatformData*>(getContextImpl(context).getPlatformData())->contexts[0];
    }
};

static void createWater(System& system, vector<Vec3>& positions) {
    // 1000 rigid TIP4P-Ew molecules exceed the all-pairs threshold and activate role order.
    const int side = 10;
    const double spacing = 0.31, length = side*spacing;
    const double oh = 0.09572, hh = 0.15139, hx = sqrt(oh*oh-0.25*hh*hh);
    system.setDefaultPeriodicBoxVectors(Vec3(length, 0, 0), Vec3(0, length, 0), Vec3(0, 0, length));
    auto* force = new NonbondedForce();
    force->setNonbondedMethod(NonbondedForce::PME);
    force->setCutoffDistance(0.9);
    force->setEwaldErrorTolerance(1e-4);
    system.addForce(force);
    for (int x = 0; x < side; x++)
        for (int y = 0; y < side; y++)
            for (int z = 0; z < side; z++) {
                int o = system.addParticle(15.99943);
                system.addParticle(1.007947);
                system.addParticle(1.007947);
                system.addParticle(0);
                force->addParticle(0, 0.316435, 0.680946);
                force->addParticle(0.52422, 1, 0);
                force->addParticle(0.52422, 1, 0);
                force->addParticle(-1.04844, 1, 0);
                for (int a = 0; a < 4; a++)
                    for (int b = a+1; b < 4; b++)
                        force->addException(o+a, o+b, 0, 1, 0);
                system.setVirtualSite(o+3, new ThreeParticleAverageSite(o, o+1, o+2,
                        0.786646558, 0.106676721, 0.106676721));
                system.addConstraint(o, o+1, oh);
                system.addConstraint(o, o+2, oh);
                system.addConstraint(o+1, o+2, hh);
                Vec3 base(spacing*x, spacing*y, spacing*z);
                positions.push_back(base);
                positions.push_back(base+Vec3(hx, hh/2, 0));
                positions.push_back(base+Vec3(hx, -hh/2, 0));
                positions.push_back(base+Vec3(2*0.106676721*hx, 0, 0));
            }
}


static vector<unsigned char> rawVelocities(MetalContext& metal) {
    MetalArray& velm = metal.unwrap(metal.getVelm());
    bool pairs = velm.getPairStorage();
    velm.setPairStorage(false);
    vector<unsigned char> bytes(velm.getSize()*velm.getElementSize());
    velm.download(bytes.data());
    velm.setPairStorage(pairs);
    return bytes;
}

// A fresh Context must use the same arithmetic as a Context that has already stepped.
static void testCheckpoint(TestPlatform& platform, const string& precision) {
    System system;
    vector<Vec3> positions;
    createWater(system, positions);
    auto* barostat = new MonteCarloBarostat(1, 300, 25);
    barostat->setRandomNumberSeed(2027);
    system.addForce(barostat);
    map<string, string> properties = {{"Precision", precision}, {"DeterministicForces", "true"}};
    string blob;
    vector<vector<Vec3>> expected;
    vector<unsigned char> velocities;
    {
        LangevinMiddleIntegrator integrator(300, 1, 0.002);
        integrator.setRandomNumberSeed(2026);
        Context context(system, integrator, platform, properties);
        context.setPositions(positions);
        context.setVelocitiesToTemperature(300, 2026);
        context.applyConstraints(integrator.getConstraintTolerance());
        integrator.step(2500);
        stringstream checkpoint;
        context.createCheckpoint(checkpoint);
        blob = checkpoint.str();
        velocities = rawVelocities(platform.metal(context));
        for (int i = 0; i < 100; i++) {
            integrator.step(1);
            expected.push_back(context.getState(State::Positions).getPositions());
        }
    }
    LangevinMiddleIntegrator integrator(300, 1, 0.002);
    integrator.setRandomNumberSeed(2026);
    Context context(system, integrator, platform, properties);
    stringstream checkpoint(blob);
    context.loadCheckpoint(checkpoint);
    stringstream restored;
    context.createCheckpoint(restored);
    ASSERT(blob == restored.str());
    ASSERT(velocities == rawVelocities(platform.metal(context)));
    for (int i = 0; i < 100; i++) {
        integrator.step(1);
        const vector<Vec3> actual = context.getState(State::Positions).getPositions();
        double delta = 0;
        for (int j = 0; j < actual.size(); j++)
            for (int k = 0; k < 3; k++)
                delta = max(delta, abs(actual[j][k]-expected[i][j][k]));
        if (memcmp(actual.data(), expected[i].data(), actual.size()*sizeof(Vec3)) != 0) {
            cout << "checkpoint mismatch at step " << 2501+i << ", maximum position delta " << delta << endl;
            throw OpenMMException("Checkpoint continuation must be bitwise equal");
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        TestPlatform platform;
        testCheckpoint(platform, argc > 1 ? argv[1] : "mixed");
        cout << "Done" << endl;
        return 0;
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }
}
