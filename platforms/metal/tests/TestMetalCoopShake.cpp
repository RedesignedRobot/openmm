// Cooperative SHAKE must preserve a captured constrained step and HMR dynamics.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "MetalContext.h"
#include "MetalIntegrationUtilities.h"
#include "MetalNonbondedUtilities.h"
#include "MetalPlatform.h"
#include "openmm/Context.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace OpenMM;
using namespace std;

class ShakePlatform : public MetalPlatform {
public:
    MetalContext& metal(Context& context) const {
        return *static_cast<PlatformData*>(getContextImpl(context).getPlatformData())->contexts[0];
    }
};

class ShakeSwitch {
public:
    ShakeSwitch() : present(getenv("OPENMM_METAL_COOP_SHAKE") != nullptr),
        saved(present ? getenv("OPENMM_METAL_COOP_SHAKE") : "") {}
    ~ShakeSwitch() {
        if (present) setenv("OPENMM_METAL_COOP_SHAKE", saved.c_str(), 1);
        else unsetenv("OPENMM_METAL_COOP_SHAKE");
    }
private:
    bool present;
    string saved;
};

const double tolerance = 1e-5;

void createSystem(System& system, vector<Vec3>& positions) {
    // 790 HMR carbon/hydrogen groups, connected into ten-carbon fragments by
    // unconstrained bonds, plus 300 rigid waters.  All three SHAKE sizes occur.
    auto* nb = new NonbondedForce();
    nb->setNonbondedMethod(NonbondedForce::PME);
    nb->setCutoffDistance(0.8);
    nb->setUseDispersionCorrection(false);
    system.addForce(nb);
    auto* backbone = new HarmonicBondForce();
    system.addForce(backbone);
    system.setDefaultPeriodicBoxVectors(Vec3(7, 0, 0), Vec3(0, 7, 0), Vec3(0, 0, 7));
    Vec3 directions[] = {Vec3(1, 0, 0), Vec3(-1.0/3, sqrt(8.0/9), 0),
                          Vec3(-1.0/3, -sqrt(2.0/9), sqrt(2.0/3))};
    int previous = -1;
    for (int group = 0; group < 1090; group++) {
        Vec3 origin(0.2+0.6*(group%10), 0.2+0.6*((group/10)%11), 0.2+0.6*(group/110));
        int central = system.getNumParticles();
        bool water = group >= 790;
        int hydrogens = (water ? 2 : 1+group%3);
        system.addParticle((water ? 18.015 : 12.011+hydrogens*1.008)-hydrogens*3.024);
        nb->addParticle(0, 0.25, 0.1);
        positions.push_back(origin);
        if (!water && group%10 != 0)
            backbone->addBond(previous, central, 0.6, 1000);
        previous = central;
        for (int h = 0; h < hydrogens; h++) {
            int atom = system.addParticle(3.024);
            // Charge-only hydrogens and LJ-only centers exercise role ordering.
            nb->addParticle(group%2 == 0 ? 0.01 : -0.01, 0.1, 0);
            positions.push_back(origin+directions[h]*0.11);
            system.addConstraint(central, atom, 0.11);
        }
        if (water)
            system.addConstraint(central+1, central+2, 0.11*sqrt(8.0/3));
        for (int i = central; i <= central+hydrogens; i++)
            for (int j = i+1; j <= central+hydrogens; j++)
                nb->addException(i, j, 0, 1, 0);
    }
    ASSERT(system.getNumParticles() > 3000);
}

vector<double> residuals(const System& system, const State& state) {
    vector<double> result;
    for (int i = 0; i < system.getNumConstraints(); i++) {
        int a, b;
        double length;
        system.getConstraintParameters(i, a, b, length);
        Vec3 r = state.getPositions()[a]-state.getPositions()[b];
        double residual = fabs(sqrt(r.dot(r))/length-1);
        ASSERT(isfinite(residual));
        result.push_back(residual);
    }
    return result;
}

struct Result {
    State first;
    double maxResidual = 0;
    double temperature = 0;
};

Result trajectory(ShakePlatform& platform, const System& system, const State& captured, bool coop, int steps) {
    setenv("OPENMM_METAL_COOP_SHAKE", coop ? "1" : "0", 1);
    LangevinMiddleIntegrator integrator(300, 1, 0.004);
    integrator.setConstraintTolerance(tolerance);
    integrator.setRandomNumberSeed(9127);
    Context context(system, integrator, platform);
    context.setState(captured);
    Result result;
    int dofs = 3*system.getNumParticles()-system.getNumConstraints();
    for (int step = 1; step <= steps; step++) {
        integrator.step(1);
        State state = context.getState(State::Positions | State::Velocities | State::Energy);
        ASSERT(isfinite(state.getPotentialEnergy()));
        ASSERT(isfinite(state.getKineticEnergy()));
        auto errors = residuals(system, state);
        result.maxResidual = max(result.maxResidual, *max_element(errors.begin(), errors.end()));
        if (step == 1) {
            result.first = state;
            // Role order excludes the small-system all-pairs path.  Fold support
            // also proves the fused kernel was prepared, rather than CCMA fallback.
            ASSERT(platform.metal(context).getNonbondedUtilities().getUseRoleOrder());
            ASSERT(platform.metal(context).getIntegrationUtilities().canFoldOrderedForces());
        }
        if (step > 250)
            result.temperature += 2*state.getKineticEnergy()/(dofs*0.00831446261815324*250);
    }
    cout << "coop=" << coop << " steps=" << steps << " maxResidual=" << result.maxResidual
         << " meanTemperature=" << result.temperature << endl;
    return result;
}

void compareCaptured(const System& system, const Result& off, const Result& on) {
    double maxPosition = 0, maxVelocity = 0;
    for (int i = 0; i < system.getNumParticles(); i++) {
        for (int axis = 0; axis < 3; axis++) {
            double dp = fabs(off.first.getPositions()[i][axis]-on.first.getPositions()[i][axis]);
            double dv = fabs(off.first.getVelocities()[i][axis]-on.first.getVelocities()[i][axis]);
            ASSERT(isfinite(dp) && isfinite(dv));
            maxPosition = max(maxPosition, dp);
            maxVelocity = max(maxVelocity, dv);
        }
    }
    cout << "captured maxPositionDifference=" << maxPosition << " maxVelocityDifference=" << maxVelocity << endl;
    ASSERT(maxPosition <= tolerance*0.11);
    ASSERT(maxVelocity <= tolerance);
    auto a = residuals(system, off.first), b = residuals(system, on.first);
    for (int i = 0; i < a.size(); i++) {
        if (b[i] > a[i])
            throw OpenMMException("Captured constraint "+to_string(i)+" worsened with cooperative SHAKE");
        ASSERT(b[i] <= tolerance);
    }
}

int main(int argc, char* argv[]) {
    try {
        ShakeSwitch restore;
        string precision = (argc > 1 ? argv[1] : "mixed");
        bool capturedOnly = (argc > 2 && string(argv[2]) == "--captured-only");
        ShakePlatform platform;
        platform.setPropertyDefaultValue(MetalPlatform::MetalPrecision(), precision);
        platform.setPropertyDefaultValue(MetalPlatform::MetalDeterministicForces(), "true");
        System system;
        vector<Vec3> positions;
        createSystem(system, positions);
        State captured;
        {
            unsetenv("OPENMM_METAL_COOP_SHAKE");
            LangevinMiddleIntegrator integrator(300, 1, 0.004);
            integrator.setConstraintTolerance(tolerance);
            integrator.setRandomNumberSeed(281);
            Context context(system, integrator, platform);
            context.setPositions(positions);
            context.applyConstraints(tolerance);
            context.setVelocitiesToTemperature(300, 281);
            integrator.step(5);
            captured = context.getState(State::Positions | State::Velocities);
        }
        cout << setprecision(17) << "precision=" << precision << " atoms=" << system.getNumParticles()
             << " SHAKE clusters=790 (sizes 2,3,4), SETTLE clusters=300" << endl;
        Result off = trajectory(platform, system, captured, false, capturedOnly ? 1 : 500);
        Result on = trajectory(platform, system, captured, true, capturedOnly ? 1 : 500);
        compareCaptured(system, off, on);
        ASSERT(on.maxResidual <= off.maxResidual);
        ASSERT(on.maxResidual <= tolerance);
        if (!capturedOnly) {
            ASSERT(off.temperature > 0);
            ASSERT(fabs(on.temperature/off.temperature-1) <= 0.03);
        }
        cout << "Done" << endl;
        return 0;
    }
    catch (const exception& e) {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
}
