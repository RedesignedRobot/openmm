/* Portions copyright (c) 2026 the Authors. Licensed under the MIT license. */
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "MetalContext.h"
#include "MetalPlatform.h"
#include "openmm/Context.h"
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
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>

using namespace OpenMM;
using namespace std;

class BoxPlatform : public MetalPlatform {
public:
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

static void compare(const State& expected, const State& found) {
    double energy = expected.getPotentialEnergy();
    ASSERT(isfinite(found.getPotentialEnergy()));
    double error = 0, norm = 0;
    for (int i = 0; i < expected.getForces().size(); i++) {
        Vec3 delta = expected.getForces()[i]-found.getForces()[i];
        error += delta.dot(delta);
        norm += expected.getForces()[i].dot(expected.getForces()[i]);
    }
    double relative = sqrt(error/max(1.0, norm));
    cout << "energy expected=" << energy << " found=" << found.getPotentialEnergy() << " relative=" << abs(energy-found.getPotentialEnergy())/max(1.0, abs(energy))
         << " force RMS relative=" << relative << endl;
    ASSERT(abs(energy-found.getPotentialEnergy()) <= 1e-6*max(1.0, abs(energy)));
    ASSERT(relative < 1e-5);
}

static void testHostUpdate(BoxPlatform& platform, System& system, const State* initial, const vector<Vec3>& positions) {
    vector<State> results;
    vector<vector<double>> groupEnergies;
    for (int i = 0; i < system.getNumForces(); i++) {
        auto* nb = dynamic_cast<NonbondedForce*>(&system.getForce(i));
        system.getForce(i).setForceGroup(nb ? 1 : 0);
        if (nb) nb->setReciprocalSpaceForceGroup(2);
    }
    for (bool enabled : {false, true}) {
        setDeviceBox(enabled);
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
        ASSERT_EQUAL(enabled, platform.metal(context).getUseDeviceBox());
        if (initial) context.setState(*initial);
        else context.setPositions(positions);
        context.computeVirtualSites();
        // This first evaluation wraps and reorders the molecules before the host changes the box.
        State before = context.getState(State::Energy | State::Forces);
        ASSERT(platform.metal(context).getNonbondedUtilities().getUseRoleOrder());
        Vec3 box[3];
        before.getPeriodicBoxVectors(box[0], box[1], box[2]);
        context.setPeriodicBoxVectors(box[0]*1.015, box[1]*1.015, box[2]*1.015);
        results.push_back(context.getState(State::Energy | State::Forces));
        vector<double> groups;
        for (int group = 0; group < 3; group++)
            groups.push_back(context.getState(State::Energy, false, 1<<group).getPotentialEnergy());
        groupEnergies.push_back(groups);
        stringstream checkpoint;
        context.createCheckpoint(checkpoint);
        context.setPeriodicBoxVectors(box[0], box[1], box[2]);
        context.loadCheckpoint(checkpoint);
        State restored = context.getState(State::Energy | State::Forces);
        compare(results.back(), restored);
        Vec3 found[3];
        restored.getPeriodicBoxVectors(found[0], found[1], found[2]);
        for (int i = 0; i < 3; i++) ASSERT_EQUAL_VEC(box[i]*1.015, found[i], 1e-12);
    }
    for (int group = 0; group < 3; group++) {
        double off = groupEnergies[0][group], on = groupEnergies[1][group];
        cout << "group=" << group << " off=" << off << " on=" << on << " absolute delta=" << on-off
             << " relative delta=" << abs(on-off)/max(1.0, abs(off)) << endl;
    }
    compare(results[0], results[1]);
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

template<class T> static unique_ptr<T> readXml(const string& path) {
    ifstream input(path);
    if (!input) throw OpenMMException("Cannot read "+path);
    return unique_ptr<T>(XmlSerializer::deserialize<T>(input));
}

int main(int argc, char* argv[]) {
    try {
        BoxSetting restore;
        cout << setprecision(16);
        if (argc != 1 && argc != 2 && argc != 4)
            throw OpenMMException("Usage: TestMetalDeviceBox [single|mixed|all [system.xml state.xml]]");
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
            // Nonzero net charge exercises the volume correction as well as neutral water.
            auto& nb = dynamic_cast<NonbondedForce&>(water.getForce(0));
            nb.setParticleParameters(3, -0.94844, 1, 0);
            testHostUpdate(platform, water, nullptr, positions);
            testBarostat(platform);
            if (argc == 4) {
                auto system = readXml<System>(argv[2]);
                auto initial = readXml<State>(argv[3]);
                cout << "FAH XML atoms=" << system->getNumParticles() << endl;
                testHostUpdate(platform, *system, initial.get(), {});
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
