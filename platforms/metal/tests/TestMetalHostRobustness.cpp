/* Tests overflow recovery, NaN reporting, retained memory, and batched adaptive reordering. */
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "MetalContext.h"
#include "MetalNonbondedUtilities.h"
#include "MetalPlatform.h"
#include "MetalQueue.h"
#include "openmm/Context.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/VirtualSite.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>

using namespace OpenMM;
using namespace std;

namespace OpenMM {
// Exercise the real overflow branch without allocating a second large system to exhaust memory.
class MetalHostRobustnessAccess {
public:
    static void undersize(MetalContext& cc) {
        cc.getCurrentStream().finish();
        auto& nb = cc.getNonbondedUtilities();
        ASSERT(nb.useNeighborList && nb.useRoleOrder);
        nb.waitForCount = true;
        nb.maxTiles = 1;
        nb.maxSinglePairs = 1;
        nb.forceRebuildNeighborList = true;
    }
    static void assertRecovered(MetalContext& cc) {
        auto& nb = cc.getNonbondedUtilities();
        ASSERT(nb.maxTiles > 1 || nb.maxSinglePairs > 1);
        ASSERT(cc.getForcesValid());
    }
    static void rebuild(MetalContext& cc) {
        cc.getNonbondedUtilities().forceRebuildNeighborList = true;
        cc.forceReorder();
    }
    static unsigned int adaptiveReorders(MetalContext& cc) {
        return cc.getNonbondedUtilities().adaptiveReorderCount;
    }
    static unsigned int baseline(MetalContext& cc) {
        auto& nb = cc.getNonbondedUtilities();
        ASSERT(nb.useNeighborList && nb.useRoleOrder && !nb.waitForCount);
        return nb.tilesAfterReorder;
    }
    static unsigned int downloadTileCount(MetalContext& cc) {
        vector<unsigned int> counts;
        cc.getNonbondedUtilities().interactionCount.download(counts);
        return counts[0];
    }
};
}

class HostPlatform : public MetalPlatform {
public:
    MetalContext& metal(Context& context) const {
        auto* data = reinterpret_cast<PlatformData*>(getContextImpl(context).getPlatformData());
        return *data->contexts[0];
    }
};
static HostPlatform platform;

// 17^3 rigid TIP4P-Ew waters, 19652 sites.  Separate LJ-only and charge-only sites enable role order.
static void makeWater(System& system, vector<Vec3>& positions) {
    const int side = 17;
    const double spacing = 0.32, oh = 0.09572, angle = 104.52*3.141592653589793/180;
    const Vec3 h1(oh, 0, 0), h2(oh*cos(angle), oh*sin(angle), 0);
    const double weight = 0.0125/(2*oh*cos(angle/2));
    const double box = side*spacing;
    system.setDefaultPeriodicBoxVectors(Vec3(box, 0, 0), Vec3(0, box, 0), Vec3(0, 0, box));
    auto* nb = new NonbondedForce();
    nb->setNonbondedMethod(NonbondedForce::PME);
    nb->setCutoffDistance(0.9);
    // Keep the screened force discontinuity at the cutoff below the comparison tolerance.
    nb->setEwaldErrorTolerance(1e-6);
    for (int x = 0; x < side; x++)
        for (int y = 0; y < side; y++)
            for (int z = 0; z < side; z++) {
                int o = system.addParticle(15.99943);
                system.addParticle(1.007947);
                system.addParticle(1.007947);
                system.addParticle(0);
                system.setVirtualSite(o+3, new ThreeParticleAverageSite(o, o+1, o+2, 1-2*weight, weight, weight));
                Vec3 p = Vec3(x, y, z)*spacing;
                positions.push_back(p);
                positions.push_back(p+h1);
                positions.push_back(p+h2);
                positions.push_back(p+(h1+h2)*weight);
                nb->addParticle(0, 0.316435, 0.680946);
                nb->addParticle(0.52422, 1, 0);
                nb->addParticle(0.52422, 1, 0);
                nb->addParticle(-1.04844, 1, 0);
                system.addConstraint(o, o+1, oh);
                system.addConstraint(o, o+2, oh);
                system.addConstraint(o+1, o+2, sqrt((h1-h2).dot(h1-h2)));
                nb->addException(o, o+1, 0, 1, 0);
                nb->addException(o, o+2, 0, 1, 0);
                nb->addException(o+1, o+2, 0, 1, 0);
                for (int i = 0; i < 3; i++)
                    nb->addException(o+i, o+3, 0, 1, 0);
            }
    system.addForce(nb);
}

static double compareForces(const State& actual, const State& expected) {
    ASSERT(isfinite(actual.getPotentialEnergy()));
    ASSERT(isfinite(expected.getPotentialEnergy()));
    ASSERT_EQUAL_TOL(expected.getPotentialEnergy(), actual.getPotentialEnergy(), 1e-4);
    const auto& a = actual.getForces();
    const auto& b = expected.getForces();
    double maxError = 0;
    for (int i = 0; i < (int) a.size(); i++) {
        double error = sqrt((a[i]-b[i]).dot(a[i]-b[i]));
        double scale = max(1.0, sqrt(b[i].dot(b[i])));
        ASSERT(isfinite(error));
        maxError = max(maxError, error/scale);
        if (error > 1e-4*scale)
            throw OpenMMException("Force mismatch at atom "+to_string(i)+": error="+to_string(error)+", scale="+to_string(scale));
    }
    return maxError;
}

static void testOverflow() {
    System system;
    vector<Vec3> positions;
    makeWater(system, positions);
    VerletIntegrator integrator(0.0001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocities(vector<Vec3>(positions.size()));
    MetalContext& cc = platform.metal(context);
    cout << "device=" << cc.getDevice()->name()->utf8String() << ", Apple9=" << cc.getDevice()->supportsFamily(MTL::GPUFamilyApple9) << endl;
    unique_ptr<VerletIntegrator> oracleIntegrator;
    unique_ptr<Context> oracle;
    double maxError = 0;
    for (int step = 0; step < 1000; step++) {
        if (step%250 == 0) {
            // Start a fresh context for each forced overflow.  It has never seen an undersized list.
            oracle.reset();
            oracleIntegrator.reset(new VerletIntegrator(0.0001));
            oracle.reset(new Context(system, *oracleIntegrator, platform));
            MetalHostRobustnessAccess::undersize(cc);
        }
        State actual = context.getState(State::Positions | State::Forces | State::Energy);
        MetalHostRobustnessAccess::assertRecovered(cc);
        oracle->setPositions(actual.getPositions());
        try {
            maxError = max(maxError, compareForces(actual, oracle->getState(State::Forces | State::Energy)));
        }
        catch (const exception& e) {
            throw OpenMMException("Step "+to_string(step)+": "+e.what());
        }
        integrator.step(1);
    }
    ASSERT(isfinite(context.getState(State::Energy).getPotentialEnergy()));
    cout << "overflow: 19652 sites, role order, 4 forced overflows, 1000 force comparisons and steps, max scaled force error=" << maxError << endl;
}

static void testNaNRecovery() {
    System system;
    vector<Vec3> positions;
    makeWater(system, positions);
    VerletIntegrator integrator(0.0001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.getState(State::Positions);
    vector<Vec3> invalid = positions;
    invalid[0][0] = numeric_limits<double>::quiet_NaN();
    bool caught = false;
    platform.metal(context).forceReorder();
    try {
        context.setPositions(invalid);
        context.getState(State::Positions);
    }
    catch (const OpenMMException& e) {
        if (string(e.what()).find("NaN") == string::npos)
            throw;
        caught = true;
    }
    ASSERT(caught);
    platform.metal(context).forceReorder();
    context.setPositions(positions);
    context.setVelocities(vector<Vec3>(positions.size()));
    State recovered = context.getState(State::Forces | State::Energy);
    ASSERT(isfinite(recovered.getPotentialEnergy()));
    for (const auto& f : recovered.getForces())
        ASSERT(isfinite(f.dot(f)));
    integrator.step(100);
    ASSERT(isfinite(context.getState(State::Positions | State::Energy).getPotentialEnergy()));
    cout << "NaN recovery: finite forces and 100 steps" << endl;
}

static void testMemory() {
    System system;
    vector<Vec3> positions;
    makeWater(system, positions);
    VerletIntegrator integrator(0.0001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    MetalContext& cc = platform.metal(context);
    auto rebuild = [&]() {
        MetalHostRobustnessAccess::rebuild(cc);
        context.setPositions(positions);
        ASSERT(isfinite(context.getState(State::Forces | State::Energy).getPotentialEnergy()));
        cc.getCurrentStream().finish();
    };
    for (int i = 0; i < 10; i++)
        rebuild();
    size_t before = cc.getDevice()->currentAllocatedSize();
    for (int i = 0; i < 50; i++)
        rebuild();
    size_t after = cc.getDevice()->currentAllocatedSize();
    cout << "memory: 50 reorders and rebuilds, before=" << before << " after=" << after << " bytes" << endl;
    ASSERT_EQUAL(before, after);
}

static void testAdaptiveReorder() {
    System system;
    vector<Vec3> positions;
    makeWater(system, positions);
    LangevinMiddleIntegrator integrator(300, 1, 0.0001);
    integrator.setRandomNumberSeed(2029);
    Context context(system, integrator, platform);
    MetalContext& cc = platform.metal(context);
    cc.getCurrentStream().commitWhenIdle();
    context.setPositions(positions);
    context.setVelocities(vector<Vec3>(positions.size(), Vec3()));
    context.getState(State::Forces | State::Energy);
    unsigned int baseline = MetalHostRobustnessAccess::baseline(cc);
    ASSERT(baseline > 0);
    integrator.step(8);

    // Permute whole waters without changing the physical configuration.  The old
    // atom order now has wide bounding boxes and a much larger interaction list.
    vector<Vec3> scrambled(positions.size());
    int waters = positions.size()/4;
    for (int i = 0; i < waters; ++i)
        for (int site = 0; site < 4; ++site)
            scrambled[4*i+site] = positions[4*((37*i)%waters)+site];
    context.setPositions(scrambled);
    context.getState(State::Forces | State::Energy);
    unsigned int grown = MetalHostRobustnessAccess::downloadTileCount(cc);
    ASSERT(grown > 1.1*baseline);
    ASSERT(cc.getStepsSinceReorder() < 25);
    ASSERT_EQUAL(0, MetalHostRobustnessAccess::adaptiveReorders(cc));

    // FAH batching: no state downloads or finishes in these 200 integration steps.
    // This ends before the fixed 251-step reorder could repair the poor atom order.
    integrator.step(200);
    unsigned int reorders = MetalHostRobustnessAccess::adaptiveReorders(cc);
    cout << "adaptive reorder: baseline=" << baseline << " grown=" << grown
         << " requests=" << reorders << " stepsSinceReorder=" << cc.getStepsSinceReorder() << endl;
    ASSERT_EQUAL(1, reorders);
    ASSERT(cc.getStepsSinceReorder() < 200);
    ASSERT(isfinite(context.getState(State::Energy).getPotentialEnergy()));
}

int main(int argc, char* argv[]) {
    string precision = argc > 1 ? argv[1] : "single";
    platform.setPropertyDefaultValue(MetalPlatform::MetalPrecision(), precision);
    platform.setPropertyDefaultValue("DeterministicForces", "true");
    int failures = 0;
    for (auto test : vector<pair<string, function<void()>>>{{"overflow", testOverflow}, {"NaN recovery", testNaNRecovery}, {"memory", testMemory}, {"adaptive reorder", testAdaptiveReorder}}) {
        try {
            test.second();
            cout << "PASS " << precision << " " << test.first << endl;
        }
        catch (const exception& e) {
            cerr << "FAIL " << precision << " " << test.first << ": " << e.what() << endl;
            failures++;
        }
    }
    return failures == 0 ? 0 : 1;
}
