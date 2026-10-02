/* Tests overflow recovery, NaN reporting, retained memory, and batched adaptive reordering. */
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "MetalContext.h"
#include "MetalFFT3D.h"
#include "../src/MetalAtomReorder.h"
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
#include <set>

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
    static pair<size_t, size_t> compiledObjects(MetalContext& cc) {
        return {cc.loadedModules.size(), cc.kernelArguments.size()};
    }
    static MetalAtomReorder& reorder(MetalContext& cc) {
        auto* reorder = dynamic_cast<MetalAtomReorder*>(cc.deviceReorder);
        ASSERT(reorder != nullptr);
        return *reorder;
    }
    static bool deviceOrderPending(MetalContext& cc) {
        return reorder(cc).hostStale;
    }
    static int tablesVersion(MetalContext& cc) {
        return reorder(cc).tablesVersion;
    }
    static void countHook(void (*hook)(MetalContext&)) {
        MetalNonbondedUtilities::exactCountHook = hook;
    }
    static void constructionHook(bool fft, function<void(const vector<NS::Object*>&)> hook) {
        if (fft)
            MetalFFT3D::constructionHook = hook;
        else
            MetalContext::constructionHook = hook;
    }
    static unique_ptr<MetalContext> construct(MetalContext& cc, const System& system) {
        return unique_ptr<MetalContext>(new MetalContext(system, -1, false,
                cc.getUseMixedPrecision() ? "mixed" : "single", "/tmp", cc.platformData, nullptr));
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

// This observer runs before the host reads the exact count and before integration can consume forces.
static int exactCounts = 0;
static void observeExactCount(MetalContext& cc) {
    ASSERT(cc.getCurrentStream().isDrained());
    ++exactCounts;
}

static void testOverflow() {
    System system;
    vector<Vec3> positions;
    makeWater(system, positions);
    VerletIntegrator integrator(0.0001), oracleIntegrator(0.0001);
    Context context(system, integrator, platform), oracle(system, oracleIntegrator, platform);
    context.setPositions(positions);
    context.setVelocities(vector<Vec3>(positions.size()));
    MetalContext& cc = platform.metal(context);
    cout << "device=" << cc.getDevice()->name()->utf8String() << endl;
    struct Observer {
        Observer() { exactCounts = 0; MetalHostRobustnessAccess::countHook(observeExactCount); }
        ~Observer() { MetalHostRobustnessAccess::countHook(nullptr); }
    } observer;
    double maxError = 0, maxVelocityError = 0;
    for (int step = 0; step < 2048; step++) {
        // Overflow first occurs inside step(), with no intervening force or energy download.
        bool overflow = (step%256 == 0);
        State before = context.getState(State::Positions | State::Velocities);
        if (overflow) {
            oracle.setPositions(before.getPositions());
            oracle.setVelocities(before.getVelocities());
            MetalHostRobustnessAccess::undersize(cc);
        }
        int observations = exactCounts;
        auto countBefore = cc.getNonbondedUtilities().getCountWaitCalls();
        integrator.step(1);
        if (overflow) {
            ASSERT(exactCounts >= observations+2); // rejected overflow, then the valid retry
            ASSERT(cc.getNonbondedUtilities().getCountWaitCalls() > countBefore);
            MetalHostRobustnessAccess::assertRecovered(cc);
            oracleIntegrator.step(1);
            State actual = context.getState(State::Positions | State::Velocities);
            State expected = oracle.getState(State::Positions | State::Velocities);
            for (int i = 0; i < system.getNumParticles(); i++) {
                ASSERT_EQUAL_VEC(expected.getPositions()[i], actual.getPositions()[i], 2e-5);
                Vec3 delta = actual.getVelocities()[i]-expected.getVelocities()[i];
                maxVelocityError = max(maxVelocityError, sqrt(delta.dot(delta)));
                ASSERT_EQUAL_VEC(expected.getVelocities()[i], actual.getVelocities()[i], 2e-4);
            }
        }
        State actual = context.getState(State::Positions | State::Forces | State::Energy);
        oracle.setPositions(actual.getPositions());
        try {
            maxError = max(maxError, compareForces(actual, oracle.getState(State::Forces | State::Energy)));
        }
        catch (const exception& e) {
            throw OpenMMException("Step "+to_string(step)+": "+e.what());
        }
    }
    cout << "overflow: 19652 sites, role order, 8 forced integration overflows, 2048 steps and force comparisons"
         << ", exact completed counts=" << exactCounts << ", max scaled force error=" << maxError
         << ", max overflow-step velocity error=" << maxVelocityError << endl;
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
    MetalContext& cc = platform.metal(context);
    vector<int> oldOrder = cc.getAtomIndex();
    // Reverse whole waters so recovery must produce a different spatial order.
    for (int water = 0; water < (int) positions.size()/8; water++)
        for (int site = 0; site < 4; site++)
            swap(positions[4*water+site], positions[positions.size()-4*(water+1)+site]);
    context.setPositions(positions);
    cc.forceReorder();
    cc.reorderAtoms();
    ASSERT(MetalHostRobustnessAccess::deviceOrderPending(cc));
    cc.getCurrentStream().finish();
    vector<int> deviceOrder;
    cc.getAtomIndexArray().download(deviceOrder);
    ASSERT(!equal(oldOrder.begin(), oldOrder.end(), deviceOrder.begin()));
    context.setVelocities(vector<Vec3>(positions.size()));
    State recovered = context.getState(State::Forces | State::Energy);
    ASSERT(isfinite(recovered.getPotentialEnergy()));
    for (const auto& f : recovered.getForces())
        ASSERT(isfinite(f.dot(f)));
    integrator.step(100);
    ASSERT(isfinite(context.getState(State::Positions | State::Energy).getPotentialEnergy()));
    cout << "NaN recovery: device reorder submitted, device permutation changed, finite forces and 100 steps" << endl;
}

static void testMemory() {
    System system;
    vector<Vec3> positions;
    makeWater(system, positions);
    VerletIntegrator integrator(0.0001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    MetalContext& cc = platform.metal(context);
    auto& nb = dynamic_cast<NonbondedForce&>(system.getForce(0));
    context.getState(State::Forces);
    int iteration = 0;
    auto rebuild = [&]() {
        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        int version = cc.getMoleculeGroupsVersion();
        // Move the distinct oxygen between two waters.  Each update invalidates a molecule group.
        nb.setParticleParameters(0, 0, 0.316435, iteration%2 ? 0.680946 : 0.7);
        nb.setParticleParameters(4, 0, 0.316435, iteration%2 ? 0.7 : 0.680946);
        nb.updateParametersInContext(context);
        ++iteration;
        MetalHostRobustnessAccess::rebuild(cc);
        context.setPositions(positions);
        ASSERT(isfinite(context.getState(State::Forces | State::Energy).getPotentialEnergy()));
        cc.getCurrentStream().finish();
        ASSERT(cc.getMoleculeGroupsVersion() > version);
        ASSERT_EQUAL(cc.getMoleculeGroupsVersion(), MetalHostRobustnessAccess::tablesVersion(cc));
    };
    for (int i = 0; i < 10; i++)
        rebuild();
    auto before = MetalHostRobustnessAccess::compiledObjects(cc);
    size_t bytesBefore = cc.getDevice()->currentAllocatedSize();
    int version = cc.getMoleculeGroupsVersion();
    for (int i = 0; i < 50; i++)
        rebuild();
    auto after = MetalHostRobustnessAccess::compiledObjects(cc);
    cout << "memory: 50 parameter table rebuilds, versions=" << version << "->" << cc.getMoleculeGroupsVersion()
         << ", libraries=" << before.first << "->" << after.first
         << ", pipelines=" << before.second << "->" << after.second
         << ", device bytes=" << bytesBefore << "->" << cc.getDevice()->currentAllocatedSize() << endl;
    ASSERT_EQUAL(before.first, after.first);
    ASSERT_EQUAL(before.second, after.second);
}

static void testConstructorCleanup(bool fft) {
    System system;
    system.addParticle(1);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    MetalContext& cc = platform.metal(context);
    {
        size_t observed = 0;
        for (int iteration = 0; iteration < 10; iteration++) {
            auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
            struct Checkpoint {
                bool fft;
                vector<pair<NS::Object*, NS::UInteger>> retained;
                Checkpoint(bool fft) : fft(fft) {
                    MetalHostRobustnessAccess::constructionHook(fft, [&](const vector<NS::Object*>& objects) {
                        set<NS::Object*> unique(objects.begin(), objects.end());
                        for (auto object : unique) {
                            object->retain();
                            retained.push_back({object, object->retainCount()});
                        }
                        throw OpenMMException("host-test construction failure");
                    });
                }
                ~Checkpoint() {
                    MetalHostRobustnessAccess::constructionHook(fft, {});
                    for (auto entry : retained)
                        entry.first->release();
                }
            } checkpoint(fft);
            bool caught = false;
            try {
                if (fft)
                    MetalFFT3D failed(cc, 28, 25, 30, false);
                else
                    MetalHostRobustnessAccess::construct(cc, system);
            }
            catch (const OpenMMException& e) {
                ASSERT_EQUAL(string("host-test construction failure"), string(e.what()));
                caught = true;
            }
            MetalHostRobustnessAccess::constructionHook(fft, {});
            ASSERT(caught);
            ASSERT(checkpoint.retained.size() >= 3);
            for (auto entry : checkpoint.retained) {
                if (entry.first->retainCount() >= entry.second)
                    throw OpenMMException(string(fft ? "FFT" : "Context")+" did not release an owned device object");
                ++observed;
            }
            // Retry the same constructor and execute GPU work with the recovered object.
            if (fft) {
                const int size = 28*25*30;
                MetalArray input(cc, size, sizeof(mm_float2), "retryInput");
                MetalArray output(cc, size, sizeof(mm_float2), "retryOutput");
                vector<mm_float2> values(size, mm_float2(0, 0));
                values[0].x = 1;
                input.upload(values);
                MetalFFT3D retry(cc, 28, 25, 30, false);
                retry.execFFT(input, output, true);
                output.download(values);
                for (auto value : values) {
                    ASSERT_EQUAL_TOL(1, value.x, 1e-5);
                    ASSERT_EQUAL_TOL(0, value.y, 1e-5);
                }
            }
            else {
                auto retry = MetalHostRobustnessAccess::construct(cc, system);
                retry->getCurrentStream().finish();
            }
        }
        cout << (fft ? "FFT" : "Context") << " cleanup: 10 injected failures and successful retries, "
             << observed << " owned device references released" << endl;
    }
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
    for (auto test : vector<pair<string, function<void()>>>{{"overflow", testOverflow}, {"NaN recovery", testNaNRecovery}, {"memory", testMemory}, {"context cleanup", [] { testConstructorCleanup(false); }}, {"FFT cleanup", [] { testConstructorCleanup(true); }}, {"adaptive reorder", testAdaptiveReorder}}) {
        if (argc > 2 && (string(argv[2]) == "audit" ? test.first == "adaptive reorder" : string(argv[2]) != test.first))
            continue;
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
