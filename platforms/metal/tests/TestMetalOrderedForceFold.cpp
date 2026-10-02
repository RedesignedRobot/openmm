// Regression for consuming role-ordered nonbonded forces inside LangevinMiddle.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "MetalContext.h"
#include "MetalIntegrationUtilities.h"
#include "MetalNonbondedUtilities.h"
#include "MetalPlatform.h"
#include "openmm/Context.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/MonteCarloBarostat.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VirtualSite.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/serialization/XmlSerializer.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace OpenMM;
using namespace std;

class FoldPlatform : public MetalPlatform {
public:
    MetalContext& getMetalContext(Context& context) const {
        return *static_cast<PlatformData*>(getContextImpl(context).getPlatformData())->contexts[0];
    }
    void computeForces(Context& context) const {
        getContextImpl(context).calcForcesAndEnergy(true, false);
    }
};

class FoldSwitch {
public:
    FoldSwitch() : present(getenv("OPENMM_METAL_FOLD_ORDERED_FORCES") != nullptr),
            saved(present ? getenv("OPENMM_METAL_FOLD_ORDERED_FORCES") : "") {}
    ~FoldSwitch() {
        if (present)
            setenv("OPENMM_METAL_FOLD_ORDERED_FORCES", saved.c_str(), 1);
        else
            unsetenv("OPENMM_METAL_FOLD_ORDERED_FORCES");
    }
private:
    bool present;
    string saved;
};

void createWater(System& system, vector<Vec3>& positions) {
    // NonbondedForce enables the neighbor list above 3000 particles.  Role order requires that list.
    const int side = 10;
    const double spacing = 0.6, box = side*spacing;
    system.setDefaultPeriodicBoxVectors(Vec3(box, 0, 0), Vec3(0, box, 0), Vec3(0, 0, box));
    auto* nb = new NonbondedForce();
    nb->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    nb->setCutoffDistance(1.0);
    system.addForce(nb);
    // The M-site must receive context words as well as ordered nonbonded words.
    auto* external = new CustomExternalForce("0.125*x-0.25*y+0.5*z");
    system.addForce(external);
    const double oh = 0.09572, hh = 0.15139;
    const double hy = sqrt(oh*oh-0.25*hh*hh);
    for (int x = 0; x < side; x++)
        for (int y = 0; y < side; y++)
            for (int z = 0; z < side; z++) {
                int o = system.addParticle(15.99943);
                system.addParticle(1.007947);
                system.addParticle(1.007947);
                system.addParticle(0);
                nb->addParticle(0, 0.316435, 0.680946);
                nb->addParticle(0.52422, 1, 0);
                nb->addParticle(0.52422, 1, 0);
                nb->addParticle(-1.04844, 1, 0);
                for (int a = 0; a < 4; a++)
                    for (int b = a+1; b < 4; b++)
                        nb->addException(o+a, o+b, 0, 1, 0);
                system.addConstraint(o, o+1, oh);
                system.addConstraint(o, o+2, oh);
                system.addConstraint(o+1, o+2, hh);
                system.setVirtualSite(o+3, new ThreeParticleAverageSite(o, o+1, o+2,
                        0.786646558, 0.106676721, 0.106676721));
                external->addParticle(o+3, vector<double>());
                Vec3 base(spacing*x+0.1, spacing*y+0.1, spacing*z+0.1);
                positions.push_back(base);
                positions.push_back(base+Vec3(-hh/2, hy, 0));
                positions.push_back(base+Vec3(hh/2, hy, 0));
                positions.push_back(base);
            }
}

void assertSameBits(const vector<Vec3>& a, const vector<Vec3>& b, const string& label = "state") {
    ASSERT_EQUAL(a.size(), b.size());
    for (int i = 0; i < a.size(); i++)
        for (int j = 0; j < 3; j++) {
            double left = a[i][j], right = b[i][j];
            if (memcmp(&left, &right, sizeof(double)) != 0) {
                ostringstream message;
                message << label << " differs at particle " << i << " component " << j
                        << ": " << hexfloat << left << " vs " << right;
                throw OpenMMException(message.str());
            }
        }
}

vector<State> trajectory(FoldPlatform& platform, System& system, const vector<Vec3>& positions, bool fold) {
    if (fold)
        unsetenv("OPENMM_METAL_FOLD_ORDERED_FORCES"); // The default must be on.
    else
        setenv("OPENMM_METAL_FOLD_ORDERED_FORCES", "0", 1);
    LangevinMiddleIntegrator integrator(300, 1, 0.001);
    integrator.setRandomNumberSeed(9127);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocitiesToTemperature(300, 281);
    context.computeVirtualSites();
    integrator.step(1); // Prepare the fused kernel; the first force pass is materialized in both modes.
    MetalContext& metal = platform.getMetalContext(context);
    auto& nb = metal.getNonbondedUtilities();
    cout << "Synthetic fold=" << fold << " roleOrder=" << nb.getUseRoleOrder()
         << " canFold=" << metal.getIntegrationUtilities().canFoldOrderedForces() << endl;
    ASSERT(nb.getUseRoleOrder());
    ASSERT_EQUAL(fold, metal.getIntegrationUtilities().canFoldOrderedForces());
    platform.computeForces(context);
    ASSERT_EQUAL(fold, nb.getRoleForcesPending());
    vector<State> states;
    // A force download must drain the deferred words before distributing the M-site forces.
    states.push_back(context.getState(State::Forces));
    ASSERT(!nb.getRoleForcesPending());
    // Leave a pending result to be superseded by the next evaluation.  Its ordered words must not leak.
    platform.computeForces(context);
    ASSERT_EQUAL(fold, nb.getRoleForcesPending());
    integrator.step(16);
    ASSERT(!nb.getRoleForcesPending());
    states.push_back(context.getState(State::Positions | State::Velocities));
    // Cross the normal reorder interval and keep stepping after the force download.
    integrator.step(256);
    states.push_back(context.getState(State::Positions | State::Velocities | State::Forces));
    integrator.step(8);
    states.push_back(context.getState(State::Positions | State::Velocities));
    return states;
}

template <class T>
unique_ptr<T> readXml(const string& path) {
    ifstream stream(path);
    if (!stream)
        throw OpenMMException("Cannot read " + path);
    return unique_ptr<T>(XmlSerializer::deserialize<T>(stream));
}

vector<State> holdoutTrajectory(FoldPlatform& platform, System& system, const State& initial,
        const string& integratorPath, bool fold) {
    setenv("OPENMM_METAL_FOLD_ORDERED_FORCES", fold ? "1" : "0", 1);
    auto integrator = readXml<Integrator>(integratorPath);
    auto* langevin = dynamic_cast<LangevinMiddleIntegrator*>(integrator.get());
    ASSERT(langevin != nullptr);
    langevin->setRandomNumberSeed(9127);
    Context context(system, *integrator, platform);
    context.setState(initial);
    integrator->step(1);
    MetalContext& metal = platform.getMetalContext(context);
    auto& nb = metal.getNonbondedUtilities();
    platform.computeForces(context);
    if (getenv("OPENMM_METAL_DEVICE_BOX") != nullptr && string(getenv("OPENMM_METAL_DEVICE_BOX")) == "1")
        ASSERT(metal.getUseDeviceBox());
    cout << "Holdout fold=" << fold << " particles=" << system.getNumParticles()
         << " deviceBox=" << metal.getUseDeviceBox()
         << " roleOrder=" << nb.getUseRoleOrder()
         << " canFold=" << metal.getIntegrationUtilities().canFoldOrderedForces()
         << " roleForcesPending=" << nb.getRoleForcesPending() << endl;
    ASSERT(nb.getUseRoleOrder());
    ASSERT_EQUAL(fold, metal.getIntegrationUtilities().canFoldOrderedForces());
    ASSERT_EQUAL(fold, nb.getRoleForcesPending());
    vector<State> states;
    for (int step = 0; step < 50; step++) {
        integrator->step(1);
        ASSERT(!nb.getRoleForcesPending());
        states.push_back(context.getState(State::Positions | State::Velocities));
    }
    return states;
}

void testHoldout(FoldPlatform& platform, const string& systemPath, const string& statePath,
        const string& integratorPath) {
    auto system = readXml<System>(systemPath);
    auto initial = readXml<State>(statePath);
    for (int i = 0; i < system->getNumForces(); i++)
        if (auto* barostat = dynamic_cast<MonteCarloBarostat*>(&system->getForce(i)))
            barostat->setRandomNumberSeed(1729);
    auto off = holdoutTrajectory(platform, *system, *initial, integratorPath, false);
    auto on = holdoutTrajectory(platform, *system, *initial, integratorPath, true);
    for (int i = 0; i < off.size(); i++) {
        string step = "Holdout step " + to_string(i+1);
        assertSameBits(off[i].getPositions(), on[i].getPositions(), step+" positions");
        assertSameBits(off[i].getVelocities(), on[i].getVelocities(), step+" velocities");
        Vec3 a[3], b[3];
        off[i].getPeriodicBoxVectors(a[0], a[1], a[2]);
        on[i].getPeriodicBoxVectors(b[0], b[1], b[2]);
        assertSameBits(vector<Vec3>(a, a+3), vector<Vec3>(b, b+3), step+" box vectors");
    }
    cout << "Holdout positions, velocities, and box vectors match bitwise at all 50 steps" << endl;
}

int main(int argc, char* argv[]) {
    try {
        FoldSwitch restoreEnvironment;
        if (argc != 1 && argc != 2 && argc != 5)
            throw OpenMMException("Usage: TestMetalOrderedForceFold [precision [system.xml state.xml integrator.xml]]");
        FoldPlatform platform;
        platform.setPropertyDefaultValue(MetalPlatform::MetalPrecision(), argc > 1 ? argv[1] : "single");
        // PME's floating point charge atomics vary even between two fold-off runs.
        platform.setPropertyDefaultValue(MetalPlatform::MetalDeterministicForces(), "true");
        if (argc == 5) {
            testHoldout(platform, argv[2], argv[3], argv[4]);
            return 0;
        }
        System system;
        vector<Vec3> positions;
        createWater(system, positions);
        // Role order needs Apple9.  Report a ctest skip, never a false fold-equivalence pass.
        {
            LangevinMiddleIntegrator probe(300, 1, 0.001);
            Context context(system, probe, platform);
            if (!platform.getMetalContext(context).getDevice()->supportsFamily(MTL::GPUFamilyApple9)) {
                cout << "SKIP: ordered force fold requires Apple9" << endl;
                return 77;
            }
        }
        vector<State> off = trajectory(platform, system, positions, false);
        vector<State> on = trajectory(platform, system, positions, true);
        for (int i = 0; i < off.size(); i++) {
            if (off[i].getDataTypes() & State::Positions) {
                assertSameBits(off[i].getPositions(), on[i].getPositions());
                assertSameBits(off[i].getVelocities(), on[i].getVelocities());
            }
            if (off[i].getDataTypes() & State::Forces)
                assertSameBits(off[i].getForces(), on[i].getForces());
        }
    }
    catch (const exception& e) {
        cerr << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Ordered force fold positions, velocities, and forces match bitwise" << endl;
    return 0;
}
