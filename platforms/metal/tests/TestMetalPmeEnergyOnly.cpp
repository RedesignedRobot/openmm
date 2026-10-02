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
#include <vector>

using namespace OpenMM;
using namespace std;

class TestPlatform : public MetalPlatform {
public:
    MetalContext& metal(Context& context) const {
        return *static_cast<PlatformData*>(getContextImpl(context).getPlatformData())->contexts[0];
    }
};

class EnvironmentSetting {
public:
    EnvironmentSetting(const char* name, const char* value) : name(name), existed(getenv(name) != nullptr) {
        if (existed)
            saved = getenv(name);
        setenv(name, value, 1);
    }
    ~EnvironmentSetting() {
        if (existed)
            setenv(name.c_str(), saved.c_str(), 1);
        else
            unsetenv(name.c_str());
    }
private:
    string name, saved;
    bool existed;
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

static void assertSameEnergy(Context& context, bool reproducibleSpread) {
    double energy = context.getState(State::Energy).getPotentialEnergy();
    double withForces = context.getState(State::Energy | State::Forces).getPotentialEnergy();
    ASSERT(isfinite(energy) && isfinite(withForces));
    if (reproducibleSpread) {
        ASSERT_EQUAL(0, memcmp(&energy, &withForces, sizeof(double)));
        return;
    }
    // Float-atomic spreading sums the grid in a different order on every evaluation, so even two
    // force-carrying evaluations differ in the last bits.  A skipped transform would be far outside 1e-6.
    double again = context.getState(State::Energy | State::Forces).getPotentialEnergy();
    double baseline = fabs(again-withForces)/fabs(withForces);
    double delta = fabs(energy-withForces)/fabs(withForces);
    cout << "float-atomic spread: relative energy baseline " << baseline << ", energy-only " << delta << endl;
    ASSERT(delta < 1e-6);
}

static void assertSameForces(const vector<Vec3>& expected, const vector<Vec3>& actual) {
    double error = 0, norm = 0;
    ASSERT_EQUAL(expected.size(), actual.size());
    for (int i = 0; i < expected.size(); i++) {
        Vec3 delta = actual[i]-expected[i];
        error += delta.dot(delta);
        norm += expected[i].dot(expected[i]);
    }
    ASSERT(isfinite(error) && isfinite(norm) && norm > 0);
    ASSERT(sqrt(error/norm) < 1e-5);
}

static void testEvaluations(TestPlatform& platform, const string& precision, bool owner, bool deterministic, bool queue) {
    EnvironmentSetting ownerSetting("OPENMM_METAL_PME_OWNER", owner ? "1" : "0");
    System system;
    vector<Vec3> positions;
    createWater(system, positions);
    // Verlet's shifted kinetic energy requests forces even for getState(Energy).  Middle does not.
    LangevinMiddleIntegrator integrator(300, 1, 0.001), freshIntegrator(300, 1, 0.001);
    map<string, string> properties = {{"Precision", precision}, {"UseCpuPme", "false"},
            {"DeterministicForces", deterministic ? "true" : "false"}, {"DisablePmeStream", queue ? "false" : "true"}};
    Context context(system, integrator, platform, properties);
    auto& metal = platform.metal(context);
    cout << precision << " owner=" << owner << " deterministic=" << deterministic << " queue=" << queue
         << " forcePass=" << metal.getSupportsForcePassKernels()
         << " fixedPoint=" << metal.getUseFixedPointChargeSpreading() << endl;
    ASSERT_EQUAL(!queue, metal.getPlatformData().disablePmeStream);
    if (!owner && !deterministic && (!metal.getSupportsForcePassKernels() || metal.getUseFixedPointChargeSpreading()))
        cout << "Deferred spread unavailable: requires Apple9 float atomics and force pass kernels" << endl;
    if (!metal.getSupportsForcePassKernels())
        cout << "Deferred interpolation unavailable: requires Apple9 force pass kernels" << endl;
    context.setPositions(positions);
    context.computeVirtualSites();
    metal.forceReorder();
    // Populate pmeForces on the queue before energy-only work takes the main queue.
    context.getState(State::Forces);
    ASSERT(metal.getNonbondedUtilities().getUseNeighborList());
    ASSERT(metal.getNonbondedUtilities().getUseRoleOrder());
    assertSameEnergy(context, owner || deterministic);
    context.getState(State::Energy);
    // Translate molecules by different amounts while retaining rigid geometry.
    for (int i = 0; i < positions.size(); i++) {
        int molecule = i/4;
        positions[i] += Vec3(0.021*sin(0.7*molecule), 0.019*cos(0.3*molecule), 0.017*sin(0.5*molecule));
    }
    context.setPositions(positions);
    context.computeVirtualSites();
    metal.forceReorder();
    State moved = context.getState(State::Energy | State::Forces);
    Context fresh(system, freshIntegrator, platform, properties);
    fresh.setPositions(positions);
    fresh.computeVirtualSites();
    platform.metal(fresh).forceReorder();
    State expected = fresh.getState(State::Energy | State::Forces);
    assertSameForces(expected.getForces(), moved.getForces());
    // The next forces-only call must refill pmeForces and pair both queue events correctly.
    context.getState(State::Energy);
    assertSameForces(expected.getForces(), context.getState(State::Forces).getForces());
    assertSameEnergy(context, owner || deterministic);
}

static void testBarostat(TestPlatform& platform, const string& precision, bool device) {
    EnvironmentSetting ownerSetting("OPENMM_METAL_PME_OWNER", "0");
    EnvironmentSetting deviceSetting("OPENMM_METAL_DEVICE_BAROSTAT", device ? "1" : "0");
    System system;
    vector<Vec3> positions;
    createWater(system, positions);
    auto* barostat = new MonteCarloBarostat(1, 300, 25);
    barostat->setRandomNumberSeed(2029);
    system.addForce(barostat);
    LangevinMiddleIntegrator integrator(300, 1, 0.001);
    integrator.setRandomNumberSeed(2029);
    Context context(system, integrator, platform, {{"Precision", precision}, {"DeterministicForces", "true"},
            {"UseCpuPme", "false"}, {"DisablePmeStream", "false"}});
    context.setPositions(positions);
    context.computeVirtualSites();
    context.setVelocitiesToTemperature(300, 2029);
    double initialVolume = context.getState(0).getPeriodicBoxVolume();
    bool moved = false;
    for (int step = 0; step < 200; step += 25) {
        integrator.step(25);
        State state = context.getState(State::Energy | State::Positions);
        ASSERT(isfinite(state.getPotentialEnergy()) && isfinite(state.getKineticEnergy()));
        ASSERT(isfinite(state.getPeriodicBoxVolume()) && state.getPeriodicBoxVolume() > 0);
        for (auto& pos : state.getPositions())
            for (int axis = 0; axis < 3; axis++)
                ASSERT(isfinite(pos[axis]));
        moved |= (state.getPeriodicBoxVolume() != initialVolume);
    }
    ASSERT(moved);
    cout << precision << " barostat device=" << device << " steps=200 frequency=25 volume moved" << endl;
}

int main(int argc, char* argv[]) {
    try {
        TestPlatform platform;
        EnvironmentSetting boxSetting("OPENMM_METAL_DEVICE_BOX", "1");
        EnvironmentSetting deferSetting("OPENMM_METAL_DEFER_BAROSTAT_ENERGIES", "1");
        string requested = argc > 1 ? argv[1] : "all";
        ASSERT(requested == "all" || requested == "single" || requested == "mixed");
        for (string precision : {string("single"), string("mixed")}) {
            if (requested != "all" && requested != precision)
                continue;
            for (bool queue : {false, true}) {
                testEvaluations(platform, precision, true, true, queue);  // Owner spread, deferred interpolation.
                testEvaluations(platform, precision, false, true, queue); // Fixed-point spread, deferred interpolation.
                testEvaluations(platform, precision, false, false, queue);// Deferred float-atomic spread.
            }
            testBarostat(platform, precision, false);
            testBarostat(platform, precision, true);
        }
        cout << "Done" << endl;
    }
    catch (const exception& e) {
        cerr << "exception: " << e.what() << endl;
        return 1;
    }
    return 0;
}
