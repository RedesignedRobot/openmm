/* Portions copyright (c) 2026 the Authors. Licensed under the MIT license. */
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "MetalContext.h"
#include "MetalPlatform.h"
#include "openmm/Context.h"
#include "openmm/kernels.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/MonteCarloBarostat.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/VirtualSite.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/serialization/XmlSerializer.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>

using namespace OpenMM;
using namespace std;

class BoxPlatform : public MetalPlatform {
public:
    ContextImpl& impl(Context& context) const { return getContextImpl(context); }
    MetalContext& metal(Context& context) const {
        return *static_cast<PlatformData*>(getContextImpl(context).getPlatformData())->contexts[0];
    }
};

class BoxSetting {
public:
    BoxSetting() : present(getenv("OPENMM_METAL_DEVICE_BOX") != nullptr), value(present ? getenv("OPENMM_METAL_DEVICE_BOX") : "") {}
    ~BoxSetting() {
        if (present) setenv("OPENMM_METAL_DEVICE_BOX", value.c_str(), 1);
        else unsetenv("OPENMM_METAL_DEVICE_BOX");
    }
private:
    bool present;
    string value;
};

static void setDeviceBox(bool enabled) {
    if (enabled) setenv("OPENMM_METAL_DEVICE_BOX", "1", 1);
    else unsetenv("OPENMM_METAL_DEVICE_BOX");
}

static void createWater(System& system, vector<Vec3>& positions, int side) {
    const double spacing = 0.31, length = side*spacing;
    system.setDefaultPeriodicBoxVectors(Vec3(length, 0, 0), Vec3(0, length, 0), Vec3(0, 0, length));
    auto* force = new NonbondedForce();
    force->setNonbondedMethod(NonbondedForce::PME);
    force->setCutoffDistance(0.9);
    force->setEwaldErrorTolerance(1e-4);
    force->setUseDispersionCorrection(true);
    system.addForce(force);
    const double oh = 0.09572, hh = 0.15139, hx = sqrt(oh*oh-0.25*hh*hh);
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
                // Include molecules outside the primary cell to exercise cell offsets on box changes.
                Vec3 base(spacing*x+length, spacing*y-length, spacing*z);
                positions.push_back(base);
                positions.push_back(base+Vec3(hx, hh/2, 0));
                positions.push_back(base+Vec3(hx, -hh/2, 0));
                positions.push_back(base+Vec3(2*0.106676721*hx, 0, 0));
            }
}

static void compareForces(const State& expected, const State& found) {
    ASSERT_EQUAL(expected.getForces().size(), found.getForces().size());
    for (int i = 0; i < expected.getForces().size(); i++) {
        for (int j = 0; j < 3; j++) {
            const double a = expected.getForces()[i][j], b = found.getForces()[i][j];
            ASSERT(memcmp(&a, &b, sizeof(double)) == 0);
        }
    }
}

static void compare(const State& expected, const State& found, double energyTolerance) {
    double energy = expected.getPotentialEnergy();
    ASSERT(isfinite(found.getPotentialEnergy()));
    compareForces(expected, found);
    cout << "energy expected=" << energy << " found=" << found.getPotentialEnergy()
         << " absolute delta=" << abs(energy-found.getPotentialEnergy())
         << " tolerance=" << energyTolerance << " forces=bitwise" << endl;
    ASSERT(abs(energy-found.getPotentialEnergy()) <= energyTolerance);
}

template<class T> static unique_ptr<T> readXml(const string& path) {
    ifstream input(path);
    if (!input) throw OpenMMException("Cannot read "+path);
    return unique_ptr<T>(XmlSerializer::deserialize<T>(input));
}

static void testHostUpdate(BoxPlatform& platform, System& system, const State* initial, const vector<Vec3>& positions,
        const string& integratorPath = "") {
    vector<State> results, initialResults;
    vector<vector<State>> groupStates;
    for (int i = 0; i < system.getNumForces(); i++) {
        auto* nb = dynamic_cast<NonbondedForce*>(&system.getForce(i));
        system.getForce(i).setForceGroup(nb ? 1 : 0);
        if (nb) nb->setReciprocalSpaceForceGroup(2);
    }
    for (bool enabled : {false, true}) {
        setDeviceBox(enabled);
        unique_ptr<Integrator> integrator = integratorPath.empty() ? unique_ptr<Integrator>(new VerletIntegrator(0.001)) : readXml<Integrator>(integratorPath);
        Context context(system, *integrator, platform);
        ASSERT_EQUAL(enabled, platform.metal(context).getUseDeviceBox());
        if (initial) context.setState(*initial);
        else context.setPositions(positions);
        context.computeVirtualSites();
        // This first evaluation wraps and reorders the molecules before the host changes the box.
        State before = context.getState(State::Energy | State::Forces);
        initialResults.push_back(before);
        ASSERT(platform.metal(context).getNonbondedUtilities().getUseRoleOrder());
        Vec3 box[3];
        before.getPeriodicBoxVectors(box[0], box[1], box[2]);
        context.setPeriodicBoxVectors(box[0]*1.015, box[1]*1.015, box[2]*1.015);
        results.push_back(context.getState(State::Energy | State::Forces));
        vector<State> groups;
        for (int group = 0; group < 3; group++)
            groups.push_back(context.getState(State::Energy | State::Forces, false, 1<<group));
        groupStates.push_back(groups);
        stringstream checkpoint;
        context.createCheckpoint(checkpoint);
        context.setPeriodicBoxVectors(box[0], box[1], box[2]);
        context.loadCheckpoint(checkpoint);
        State restored = context.getState(State::Energy | State::Forces);
        compare(results.back(), restored, 0);
        Vec3 found[3];
        restored.getPeriodicBoxVectors(found[0], found[1], found[2]);
        for (int i = 0; i < 3; i++) ASSERT_EQUAL_VEC(box[i]*1.015, found[i], 1e-12);
    }
    compareForces(initialResults[0], initialResults[1]);
    cout << "Initial context forces=bitwise" << endl;
    double magnitude = 0;
    for (int group = 0; group < 3; group++) {
        double off = groupStates[0][group].getPotentialEnergy();
        magnitude += abs(off);
        cout << "group=" << group << " ";
        compare(groupStates[0][group], groupStates[1][group], 1e-7*abs(off));
    }
    compare(results[0], results[1], 1e-8*magnitude);
}

static void testDeviceScale(BoxPlatform& platform, System& system, const State* initial,
        const vector<Vec3>& positions, bool trajectory) {
    // This phase 2a test transfers checkpoints across device-box flags.  Device
    // barostat checkpoints require matching eligibility and have their own test.
    struct BarostatSetting {
        bool present = getenv("OPENMM_METAL_DEVICE_BAROSTAT") != nullptr;
        string value = present ? getenv("OPENMM_METAL_DEVICE_BAROSTAT") : "";
        BarostatSetting() { setenv("OPENMM_METAL_DEVICE_BAROSTAT", "0", 1); }
        ~BarostatSetting() {
            if (present) setenv("OPENMM_METAL_DEVICE_BAROSTAT", value.c_str(), 1);
            else unsetenv("OPENMM_METAL_DEVICE_BAROSTAT");
        }
    } barostatSetting;
    setDeviceBox(true);
    LangevinMiddleIntegrator integrator(300, 1, 0.001);
    integrator.setRandomNumberSeed(2026);
    Context device(system, integrator, platform);
    auto& metal = platform.metal(device);
    ASSERT(metal.getUseDeviceBox());
    if (initial) device.setState(*initial);
    else device.setPositions(positions);
    device.computeVirtualSites();
    device.getState(State::Energy); // Reorder and wrap before scaling.
    stringstream original;
    device.createCheckpoint(original);

    setDeviceBox(false);
    LangevinMiddleIntegrator hostIntegrator(300, 1, 0.001);
    Context host(system, hostIntegrator, platform);
    host.loadCheckpoint(original);
    Kernel barostat = platform.createKernel(ApplyMonteCarloBarostatKernel::Name(), platform.impl(host));
    MonteCarloBarostat thermostat(1, 300);
    auto& hostScale = barostat.getAs<ApplyMonteCarloBarostatKernel>();
    hostScale.initialize(system, thermostat, 1, true);
    hostScale.saveCoordinates(platform.impl(host));
    Vec3 box[3];
    device.getState(0).getPeriodicBoxVectors(box[0], box[1], box[2]);
    const double scale = 1.015;
    MetalArray factor(metal, 1, sizeof(double), "scaleFactor");
    factor.upload(vector<double>{scale});
    metal.scaleDeviceBox(factor);
    host.setPeriodicBoxVectors(box[0]*scale, box[1]*scale, box[2]*scale);
    hostScale.scaleCoordinates(platform.impl(host), scale, scale, scale, 0, 0, 0);
    State scaled = device.getState(State::Positions);
    State hostScaled = host.getState(State::Positions);
    Vec3 scaledBox[3];
    scaled.getPeriodicBoxVectors(scaledBox[0], scaledBox[1], scaledBox[2]);
    for (int i = 0; i < 3; i++) ASSERT_EQUAL_VEC(box[i]*scale, scaledBox[i], 1e-12);
    double maxDelta = 0;
    for (int i = 0; i < system.getNumParticles(); i++)
        for (int j = 0; j < 3; j++)
            maxDelta = max(maxDelta, abs(scaled.getPositions()[i][j]-hostScaled.getPositions()[i][j]));
    cout << "device scale max coordinate delta=" << maxDelta << endl;
    ASSERT(maxDelta < 2e-5);

    // Normalize both coordinate representations to identical copied inputs before force comparisons.
    host.setPeriodicBoxVectors(scaledBox[0], scaledBox[1], scaledBox[2]);
    host.setPositions(scaled.getPositions());
    device.setPositions(scaled.getPositions());
    double magnitude = 0;
    for (int group = 0; group < 3; group++) {
        State expected = host.getState(State::Energy | State::Forces, false, 1<<group);
        magnitude += abs(expected.getPotentialEnergy());
        compare(expected, device.getState(State::Energy | State::Forces, false, 1<<group),
                1e-7*abs(expected.getPotentialEnergy()));
    }
    State expected = host.getState(State::Energy | State::Forces);
    compare(expected, device.getState(State::Energy | State::Forces), 1e-8*magnitude);
    VerletIntegrator referenceIntegrator(0.001);
    Context reference(system, referenceIntegrator, Platform::getPlatformByName("Reference"));
    if (initial) reference.setState(*initial);
    reference.setPeriodicBoxVectors(scaledBox[0], scaledBox[1], scaledBox[2]);
    reference.setPositions(scaled.getPositions());
    State ref = reference.getState(State::Energy | State::Forces);
    double error2 = 0, force2 = 0;
    for (int i = 0; i < system.getNumParticles(); i++) {
        Vec3 delta = expected.getForces()[i]-ref.getForces()[i];
        error2 += delta.dot(delta);
        force2 += ref.getForces()[i].dot(ref.getForces()[i]);
    }
    double relativeRms = sqrt(error2/force2);
    cout << "scaled Reference force relative RMS=" << relativeRms
         << " energy delta/group magnitude=" << abs(expected.getPotentialEnergy()-ref.getPotentialEnergy())/magnitude << endl;
    // Bitwise equality above makes both Metal paths equally accurate against Reference with automatic PME grids.
    ASSERT(relativeRms < 1e-4);
    ASSERT(abs(expected.getPotentialEnergy()-ref.getPotentialEnergy()) < 1e-5*magnitude);

    // Save with a dirty host box, then restore the device-scaled state into a fresh context.
    metal.scaleDeviceBox(factor);
    stringstream checkpoint;
    device.createCheckpoint(checkpoint);
    State saved = device.getState(State::Energy | State::Forces | State::Positions);
    setDeviceBox(true);
    LangevinMiddleIntegrator restoredIntegrator(300, 1, 0.001);
    Context restored(system, restoredIntegrator, platform);
    restored.loadCheckpoint(checkpoint);
    double checkpointMagnitude = 0;
    for (int group = 0; group < 3; group++)
        checkpointMagnitude += abs(device.getState(State::Energy, false, 1<<group).getPotentialEnergy());
    compare(saved, restored.getState(State::Energy | State::Forces), 1e-8*checkpointMagnitude);
    Vec3 savedBox[3], restoredBox[3];
    saved.getPeriodicBoxVectors(savedBox[0], savedBox[1], savedBox[2]);
    State restoredState = restored.getState(State::Positions);
    restoredState.getPeriodicBoxVectors(restoredBox[0], restoredBox[1], restoredBox[2]);
    for (int i = 0; i < 3; i++) ASSERT_EQUAL_VEC(savedBox[i], restoredBox[i], 0);
    for (int i = 0; i < system.getNumParticles(); i++)
        ASSERT_EQUAL_VEC(saved.getPositions()[i], restoredState.getPositions()[i], 0);

    if (trajectory) {
        device.setVelocitiesToTemperature(300, 2026);
        factor.upload(vector<double>{1.0002});
        for (int i = 0; i < 8; i++) {
            integrator.step(25);
            metal.scaleDeviceBox(factor);
        }
        ASSERT(isfinite(device.getState(State::Energy).getPotentialEnergy()));
        cout << "200 steps with eight device scales and no intermediate observations passed" << endl;
    }
    // A bad cutoff is detected on the device and reported only when the host observes it.
    factor.upload(vector<double>{0.1});
    metal.scaleDeviceBox(factor);
    bool caught = false;
    try { device.getState(State::Positions); }
    catch (const OpenMMException& e) { caught = string(e.what()).find("twice the nonbonded cutoff") != string::npos; }
    ASSERT(caught);
}

// Poison the encoded scalar so this fails if an argument is silently copied with setBytes.
// More than 31 arguments also exercises the packed scalar form of the same contract.
static void testBinding(BoxPlatform& platform) {
    System system;
    vector<Vec3> positions;
    createWater(system, positions, 6);
    for (bool enabled : {false, true}) {
        setDeviceBox(enabled);
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
        auto& metal = platform.metal(context);
        for (int padding : {0, 32}) {
            string source = "KERNEL void probe(GLOBAL float* result, float4 periodicBoxSize";
            for (int i = 0; i < padding; i++) source += ", float p"+to_string(i);
            source += ") { result[0] = periodicBoxSize.x";
            for (int i = 0; i < padding; i++) source += "+p"+to_string(i);
            source += "; }";
            auto kernel = metal.compileProgram(source)->createKernel("probe");
            MetalArray result(metal, 1, sizeof(float), "boxProbe");
            kernel->addArg(result);
            kernel->addArg(mm_float4(7, 7, 7, 0));
            for (int i = 0; i < padding; i++) kernel->addArg(0.0f);
            kernel->execute(1, 1);
            vector<float> values;
            result.download(values);
            ASSERT_EQUAL_TOL(enabled ? 6*0.31 : 7.0, values[0], 1e-7);
        }
        if (enabled) {
            vector<Vec3> positive = positions;
            for (auto& p : positive) p[1] += 6*0.31;
            context.setPositions(positive);
            context.setPeriodicBoxVectors(Vec3(1e6, 0, 0), Vec3(0, 1e6, 0), Vec3(0, 0, 1e6));
            auto probe = metal.compileProgram("KERNEL void probeBins(GLOBAL int* result, int zigzagXBins) { result[0] = zigzagXBins; }")->createKernel("probeBins");
            MetalArray result(metal, 1, sizeof(int), "overflowProbe");
            probe->addArg(result);
            probe->addArg(0);
            probe->execute(1, 1);
            vector<int> bins;
            result.download(bins);
            ASSERT_EQUAL(-1, bins[0]);
            metal.forceReorder();
            metal.reorderAtoms();
            State reordered = context.getState(State::Positions);
            for (int i = 0; i < positions.size(); i++)
                ASSERT_EQUAL_VEC(positive[i], reordered.getPositions()[i], 2e-6);
        }
    }
    setDeviceBox(true);
    system.addForce(new CustomExternalForce("0"));
    VerletIntegrator integrator(0.001);
    Context fallback(system, integrator, platform);
    ASSERT(!platform.metal(fallback).getUseDeviceBox());
}

static void testBarostat(BoxPlatform& platform) {
    System system;
    vector<Vec3> positions;
    createWater(system, positions, 11);
    auto* barostat = new MonteCarloBarostat(1, 300, 25);
    barostat->setRandomNumberSeed(2029);
    system.addForce(barostat);
    double mean[2] = {};
    for (int enabled = 0; enabled < 2; enabled++) {
        setDeviceBox(enabled);
        LangevinMiddleIntegrator integrator(300, 1, 0.001);
        integrator.setRandomNumberSeed(9127);
        Context context(system, integrator, platform);
        ASSERT_EQUAL(bool(enabled), platform.metal(context).getUseDeviceBox());
        context.setPositions(positions);
        context.setVelocitiesToTemperature(300, 281);
        for (int i = 0; i < 40; i++) {
            integrator.step(25);
            State state = context.getState(State::Energy);
            ASSERT(isfinite(state.getPotentialEnergy()));
            Vec3 box[3];
            state.getPeriodicBoxVectors(box[0], box[1], box[2]);
            mean[enabled] += box[0][0]*box[1][1]*box[2][2]/40;
        }
        ASSERT(platform.metal(context).getNonbondedUtilities().getUseRoleOrder());
    }
    cout << "1000-step mean volumes: " << mean[0] << " " << mean[1] << endl;
    ASSERT(abs(mean[0]-mean[1]) < 0.02*mean[0]);
}

int main(int argc, char* argv[]) {
    try {
        BoxSetting restore;
        cout << setprecision(16);
        if (argc != 1 && argc != 2 && argc != 5)
            throw OpenMMException("Usage: TestMetalDeviceBox [single|mixed|all [system.xml state.xml integrator.xml]]");
        string precision = argc > 1 ? argv[1] : "all";
        if (precision != "single" && precision != "mixed" && precision != "all")
            throw OpenMMException("Unknown precision: "+precision);
        for (const string& mode : {string("single"), string("mixed")}) {
            if (precision != "all" && precision != mode) continue;
            BoxPlatform platform;
            platform.setPropertyDefaultValue(MetalPlatform::MetalPrecision(), mode);
            platform.setPropertyDefaultValue("DeterministicForces", "true");
            cout << "Device box precision=" << mode << endl;
            testBinding(platform);
            System water;
            vector<Vec3> positions;
            createWater(water, positions, 17);
            cout << "PME water atoms=" << water.getNumParticles() << endl;
            testHostUpdate(platform, water, nullptr, positions);
            testDeviceScale(platform, water, nullptr, positions, true);
            // Nonzero net charge exercises the volume correction as well as neutral water.
            auto& nb = dynamic_cast<NonbondedForce&>(water.getForce(0));
            nb.setParticleParameters(3, -0.94844, 1, 0);
            cout << "Charged PME water" << endl;
            testHostUpdate(platform, water, nullptr, positions);
            nb.addGlobalParameter("chargeOffset", 0.5);
            nb.addParticleParameterOffset("chargeOffset", 3, 0.2, 0, 0);
            cout << "Charged PME water with parameter offset" << endl;
            testHostUpdate(platform, water, nullptr, positions);
            testBarostat(platform);
            if (argc == 5) {
                auto system = readXml<System>(argv[2]);
                auto initial = readXml<State>(argv[3]);
                cout << "FAH XML atoms=" << system->getNumParticles() << endl;
                testHostUpdate(platform, *system, initial.get(), {}, argv[4]);
                testDeviceScale(platform, *system, initial.get(), {}, false);
            }
        }
        cout << "Done" << endl;
        return 0;
    }
    catch (const exception& e) {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
}
