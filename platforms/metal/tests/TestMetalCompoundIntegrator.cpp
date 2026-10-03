/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2015 Stanford University and the Authors.           *
 * Portions copyright (c) 2026 the Authors.                                   *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "MetalTests.h"
#include "TestCompoundIntegrator.h"
#include "MetalContext.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/NonbondedForce.h"
#include "openmm/VirtualSite.h"

class CompoundTestPlatform : public MetalPlatform {
public:
    vector<Vec3> parentForces(Context& context) const {
        ContextImpl& impl = getContextImpl(context);
        impl.calcForcesAndEnergy(true, false);
        auto& metal = *static_cast<PlatformData*>(impl.getPlatformData())->contexts[0];
        // Read the words Verlet consumes.  getState(Forces) repairs deferred virtual-site forces.
        vector<long long> words;
        metal.getLongForceBuffer().download(words);
        const vector<int>& order = metal.getAtomIndex();
        int padded = metal.getPaddedNumAtoms();
        vector<Vec3> forces(metal.getNumAtoms());
        for (int i = 0; i < metal.getNumAtoms(); i++)
            forces[order[i]] = Vec3(words[i], words[i+padded], words[i+2*padded]) / 4294967296.0;
        return forces;
    }
};

void testSwitchFromLangevinMiddleWithVirtualSites() {
    System system;
    auto* nb = new NonbondedForce();
    system.addForce(nb);
    vector<Vec3> positions;
    const double oh = 0.09572, hh = 0.15139;
    const double hy = sqrt(oh*oh-hh*hh/4);
    for (int water = 0; water < 2; water++) {
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
        Vec3 base(0.4*water, 0, 0);
        positions.push_back(base);
        positions.push_back(base+Vec3(-hh/2, hy, 0));
        positions.push_back(base+Vec3(hh/2, hy, 0));
        positions.push_back(base);
    }
    CompoundTestPlatform testPlatform;
    testPlatform.setPropertyDefaultValue("Precision", platform.getPropertyDefaultValue("Precision"));
    CompoundIntegrator compound;
    auto* langevin = new LangevinMiddleIntegrator(300, 1, 0.002);
    langevin->setRandomNumberSeed(8127);
    compound.addIntegrator(langevin);
    compound.addIntegrator(new VerletIntegrator(0.002));
    VerletIntegrator verlet(0.002);
    Context context(system, compound, testPlatform), oracle(system, verlet, testPlatform);
    context.setPositions(positions);
    context.setVelocities(vector<Vec3>(positions.size()));
    context.computeVirtualSites();
    compound.step(1);
    State start = context.getState(State::Positions | State::Velocities);
    oracle.setPositions(start.getPositions());
    oracle.setVelocities(start.getVelocities());
    compound.setCurrentIntegrator(1);
    compound.step(1); // No force download between the LangevinMiddle and Verlet steps.
    verlet.step(1);
    vector<Vec3> actualPositions = context.getState(State::Positions).getPositions();
    vector<Vec3> expectedPositions = oracle.getState(State::Positions).getPositions();
    vector<Vec3> actualForces = testPlatform.parentForces(context);
    vector<Vec3> expectedForces = testPlatform.parentForces(oracle);
    double positionError = 0, forceError = 0;
    for (int i = 0; i < positions.size(); i++) {
        Vec3 delta = actualPositions[i]-expectedPositions[i];
        positionError = max(positionError, sqrt(delta.dot(delta)));
        if (system.isVirtualSite(i))
            continue;
        delta = actualForces[i]-expectedForces[i];
        forceError = max(forceError, sqrt(delta.dot(delta))/max(1.0, sqrt(expectedForces[i].dot(expectedForces[i]))));
    }
    cout << "LangevinMiddle to Verlet: max position error=" << positionError
         << " max relative parent force error=" << forceError << endl;
    ASSERT(positionError < 1e-5);
    ASSERT(forceError < 1e-4);
}

void runPlatformTests() {
    testSwitchFromLangevinMiddleWithVirtualSites();
}
