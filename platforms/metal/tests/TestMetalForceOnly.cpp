#include "MetalContext.h"
#include "MetalPlatform.h"
#include "openmm/Context.h"
#include "openmm/CustomBondForce.h"
#include "openmm/CustomCentroidBondForce.h"
#include "openmm/CustomHbondForce.h"
#include "openmm/CustomManyParticleForce.h"
#include "openmm/CustomNonbondedForce.h"
#include "openmm/GayBerneForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/LCPOForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/RGForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include <iostream>

using namespace OpenMM;
using namespace std;

class TestPlatform : public MetalPlatform {
public:
    ContextImpl& impl(Context& context) { return getContextImpl(context); }
};

// Every force-only dispatch must leave the cleared energy buffer untouched, including
// derivative-only requests.  Alternate variants and force groups to catch stale bindings.
void checkDispatch(TestPlatform& platform, Force* force, int numParticles) {
    System system;
    vector<Vec3> positions;
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        positions.push_back(Vec3(0.2*(i%5), 0.23*((i/5)%5), 0.19*(i/25)));
    }
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    force->setForceGroup(1);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    auto& data = *static_cast<MetalPlatform::PlatformData*>(platform.impl(context).getPlatformData());
    MetalContext& cc = *data.contexts[0];
    auto assertNoEnergyWrites = [&]() {
        if (cc.getUseMixedPrecision()) {
            vector<double> buffer;
            cc.getEnergyBuffer().download(buffer);
            for (double value : buffer)
                ASSERT_EQUAL(0.0, value);
        }
        else {
            vector<float> buffer;
            cc.getEnergyBuffer().download(buffer);
            for (float value : buffer)
                ASSERT_EQUAL(0.0f, value);
        }
    };
    context.getState(State::Forces | State::ParameterDerivatives, false, 2);
    assertNoEnergyWrites();
    for (int groups : {2, 1, 2}) {
        if (context.getParameters().count("lambda"))
            context.setParameter("lambda", context.getParameter("lambda")+0.5);
        State energy = context.getState(State::Forces | State::Energy | State::ParameterDerivatives, false, groups);
        State forces = context.getState(State::Forces | State::ParameterDerivatives, false, groups);
        for (int i = 0; i < numParticles; i++)
            for (int j = 0; j < 3; j++)
                ASSERT_EQUAL(energy.getForces()[i][j], forces.getForces()[i][j]);
        for (auto& deriv : energy.getEnergyParameterDerivatives())
            ASSERT_EQUAL(deriv.second, forces.getEnergyParameterDerivatives().at(deriv.first));
        assertNoEnergyWrites();
        State repeated = context.getState(State::Energy, false, groups);
        ASSERT_EQUAL(energy.getPotentialEnergy(), repeated.getPotentialEnergy());
    }
    cout << force->getName() << " force-only output and variant transitions passed" << endl;
}

int main(int argc, char* argv[]) {
    try {
        TestPlatform platform;
        platform.setPropertyDefaultValue("Precision", argc > 1 ? argv[1] : "single");
        platform.setPropertyDefaultValue("DeterministicForces", "true");
        auto* bonds = new HarmonicBondForce();
        for (int i = 1; i < 96; i++)
            bonds->addBond(i-1, i, 0.18, 10.0);
        checkDispatch(platform, bonds, 96);

        auto* custom = new CustomBondForce("lambda*(r-0.1)^2");
        custom->addGlobalParameter("lambda", 3.0);
        custom->addEnergyParameterDerivative("lambda");
        custom->addBond(0, 1, {});
        checkDispatch(platform, custom, 6);

        auto* centroid = new CustomCentroidBondForce(2, "lambda*distance(g1,g2)^2");
        centroid->addGlobalParameter("lambda", 3.0);
        centroid->addEnergyParameterDerivative("lambda");
        centroid->addGroup({0, 1});
        centroid->addGroup({2, 3});
        centroid->addBond({0, 1}, {});
        checkDispatch(platform, centroid, 6);

        auto* many = new CustomManyParticleForce(2, "distance(p1,p2)^2");
        for (int i = 0; i < 6; i++)
            many->addParticle({});
        many->setNonbondedMethod(CustomManyParticleForce::CutoffPeriodic);
        many->setCutoffDistance(1.0);
        checkDispatch(platform, many, 6);

        auto* hbond = new CustomHbondForce("distance(d1,a1)^2");
        hbond->addDonor(0, -1, -1, {});
        hbond->addAcceptor(1, -1, -1, {});
        checkDispatch(platform, hbond, 6);

        for (bool groups : {false, true}) {
            auto* pairs = new CustomNonbondedForce("lambda*r^2");
            pairs->addGlobalParameter("lambda", 3.0);
            pairs->addEnergyParameterDerivative("lambda");
            for (int i = 0; i < 6; i++)
                pairs->addParticle({});
            if (groups)
                pairs->addInteractionGroup({0, 1, 2}, {3, 4, 5});
            checkDispatch(platform, pairs, 6);
        }
        for (auto method : {NonbondedForce::Ewald, NonbondedForce::PME}) {
            auto* ewald = new NonbondedForce();
            ewald->setNonbondedMethod(method);
            ewald->setCutoffDistance(1.0);
            ewald->addGlobalParameter("lambda", 0.5);
            for (int i = 0; i < 6; i++)
                ewald->addParticle(i%2 == 0 ? 0.1 : -0.1, 0.1, 0.01);
            ewald->addParticleParameterOffset("lambda", 0, 0.02, 0.0, 0.0);
            checkDispatch(platform, ewald, 6);
        }

        auto* gayBerne = new GayBerneForce();
        for (int i = 0; i < 6; i++)
            gayBerne->addParticle(0.1, 0.1, -1, -1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0);
        checkDispatch(platform, gayBerne, 6);
        auto* lcpo = new LCPOForce();
        for (int i = 0; i < 6; i++)
            lcpo->addParticle(0.2, 0.5, -0.2, -0.01, 0.001);
        checkDispatch(platform, lcpo, 6);
        checkDispatch(platform, new RGForce(), 6);
    }
    catch (exception& e) {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
