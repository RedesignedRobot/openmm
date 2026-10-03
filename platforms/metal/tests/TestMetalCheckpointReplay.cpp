/* Portions copyright (c) 2026 the Authors. Licensed under the MIT license. */
#include "MetalContext.h"
#include "MetalNonbondedUtilities.h"
#include "MetalPlatform.h"
#include "MetalKernels.h"
#include "openmm/Context.h"
#include "openmm/KernelFactory.h"
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
    mutable MetalApplyMonteCarloBarostatKernel* trial = nullptr;
};

class TrialFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(string name, const Platform& platform, ContextImpl& context) const override {
        auto& p=static_cast<const TestPlatform&>(platform);
        auto& metal=*static_cast<MetalPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
        p.trial=new MetalApplyMonteCarloBarostatKernel(name, platform, metal);
        return p.trial;
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
static void advanceOwner(TestPlatform& platform, Context& context, LangevinMiddleIntegrator& integrator, int step) {
    auto& metal=platform.metal(context);
    if (step == 5) {
        auto before=metal.getAtomIndex();
        // Force a real reorder in both continuations without changing coordinates.
        // PME forces need not be invariant to translation relative to its mesh.
        metal.forceReorder();
        metal.reorderAtoms();
        ASSERT(metal.getAtomsWereReordered());
        auto reordered=context.getState(State::Forces).getForces();
        auto repeated=context.getState(State::Forces).getForces();
        ASSERT(memcmp(reordered.data(),repeated.data(),reordered.size()*sizeof(Vec3)) == 0);
        cout << "owner continuation: forced reorder, permutation changed=" << (before != metal.getAtomIndex()) << endl;
    }
    bool trial=(step+1)%25 == 0;
    bool accept=(step+1)%50 == 25;
    Vec3 a,b,c;
    if (trial) {
        context.getState(0).getPeriodicBoxVectors(a,b,c);
        // Expansion with zero acceptance uniform must accept. Very high pressure
        // makes the same proposal reject with uniform 0.5, well away from a boundary.
        context.setParameter(MonteCarloBarostat::Pressure(), accept ? 1 : 1e8);
        platform.trial->setTrialUniforms(0.75, accept ? 0 : 0.5);
    }
    integrator.step(1);
    if (trial) {
        auto result=platform.trial->getTrialState();
        ASSERT_EQUAL(16, result.size());
        ASSERT_EQUAL(accept, result[6] != 0);
        Vec3 nextA,nextB,nextC;
        context.getState(0).getPeriodicBoxVectors(nextA,nextB,nextC);
        ASSERT_EQUAL(accept, nextA[0] != a[0]);
        if (!accept) { ASSERT(a == nextA); ASSERT(b == nextB); ASSERT(c == nextC); }
        cout << "owner device-box trial: " << (accept ? "accepted" : "rejected") << endl;
    }
}

static void testCheckpoint(TestPlatform& platform, const string& precision, bool owner=false) {
    if (owner) {
        setenv("OPENMM_METAL_PME_OWNER", "1", 1);
        setenv("OPENMM_METAL_DEVICE_BOX", "1", 1);
        setenv("OPENMM_METAL_DEVICE_BAROSTAT", "1", 1);
        setenv("OPENMM_METAL_COOP_SHAKE", "0", 1);
    }
    System system;
    vector<Vec3> positions;
    createWater(system, positions);
    auto* barostat = new MonteCarloBarostat(1, 300, 25);
    barostat->setRandomNumberSeed(2027);
    system.addForce(barostat);
    map<string, string> properties = {{"Precision", precision}, {"DeterministicForces", "true"},
            {"DeviceIndex", "0"}, {"DisablePmeStream", "true"}};
    string blob;
    vector<State> expected;
    vector<unsigned char> velocities;
    {
        LangevinMiddleIntegrator integrator(300, 1, 0.002);
        integrator.setRandomNumberSeed(2026);
        Context context(system, integrator, platform, properties);
        for (auto& property : properties) {
            string effective=platform.getPropertyValue(context,property.first);
            cout << "checkpoint owner=" << owner << " " << property.first << " requested=" << property.second
                 << " effective=" << effective << endl;
            ASSERT_EQUAL(property.second,effective);
        }
        if (owner) ASSERT(platform.metal(context).getUseDeviceBox());
        context.setPositions(positions);
        context.setVelocitiesToTemperature(300, 2026);
        context.applyConstraints(integrator.getConstraintTolerance());
        // Save on a barostat boundary.  MonteCarloBarostatImpl's phase counter and CustomIntegrator's
        // cached-forces flag are not checkpointed on any platform, so exact replay holds only for a
        // step-aligned save with an integrator that recomputes forces each step, as here.
        integrator.step(2500);
        stringstream checkpoint;
        context.createCheckpoint(checkpoint);
        blob = checkpoint.str();
        velocities = rawVelocities(platform.metal(context));
        for (int i = 0; i < 100; i++) {
            if (owner) advanceOwner(platform,context,integrator,i);
            else integrator.step(1);
            expected.push_back(context.getState(State::Positions|State::Velocities|State::Forces));
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
        if (owner) advanceOwner(platform,context,integrator,i);
        else integrator.step(1);
        State state=context.getState(State::Positions|State::Velocities|State::Forces);
        const vector<Vec3>& actual = state.getPositions();
        double delta = 0;
        for (int j = 0; j < actual.size(); j++)
            for (int k = 0; k < 3; k++)
                delta = max(delta, abs(actual[j][k]-expected[i].getPositions()[j][k]));
        if (memcmp(actual.data(), expected[i].getPositions().data(), actual.size()*sizeof(Vec3)) != 0) {
            cout << "checkpoint mismatch at step " << 2501+i << ", maximum position delta " << delta << endl;
            throw OpenMMException("Checkpoint continuation must be bitwise equal");
        }
        if (owner) {
            ASSERT(memcmp(state.getVelocities().data(),expected[i].getVelocities().data(),actual.size()*sizeof(Vec3)) == 0);
            ASSERT(memcmp(state.getForces().data(),expected[i].getForces().data(),actual.size()*sizeof(Vec3)) == 0);
            Vec3 a,b,c,ea,eb,ec;
            state.getPeriodicBoxVectors(a,b,c); expected[i].getPeriodicBoxVectors(ea,eb,ec);
            ASSERT(a == ea); ASSERT(b == eb); ASSERT(c == ec);
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        TestPlatform platform;
        platform.registerKernelFactory(ApplyMonteCarloBarostatKernel::Name(), new TrialFactory());
        testCheckpoint(platform, argc > 1 ? argv[1] : "mixed");
        if (argc == 1 || string(argv[1]) == "mixed") testCheckpoint(platform, "mixed", true);
        cout << "Done" << endl;
        return 0;
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }
}
