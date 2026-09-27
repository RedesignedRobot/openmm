#include "MetalContext.h"
#include "MetalNonbondedUtilities.h"
#include "MetalPlatform.h"
#include "openmm/Context.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

class ListPlatform : public MetalPlatform {
public:
    MetalContext& metal(Context& context) const {
        auto* data = reinterpret_cast<PlatformData*>(getContextImpl(context).getPlatformData());
        return *data->contexts[0];
    }
};

// A pair absent from the active list must enter it before it reaches the force
// cutoff, even when neither atom has exhausted the outer displacement budget.
// Use the list API directly so setPositions/reordering cannot force a rebuild and
// conceal a stale list.  More than 1024 blocks exercises the large-list control.
void testDisplacementBounds(const string& precision) {
    const int numAtoms = 90017;
    ListPlatform platform;
    System system;
    auto* force = new NonbondedForce();
    force->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    force->setCutoffDistance(1.0);
    force->setUseDispersionCorrection(false);
    vector<Vec3> positions;
    for (int i = 0; i < numAtoms; i++) {
        system.addParticle(1.0);
        force->addParticle(i == 0 ? 0.1 : (i == 32 ? -0.1 : 0.0), 1.0, 0.0);
        positions.push_back(Vec3(10+3*(i%46), 10+3*((i/46)%46), 10+3*(i/(46*46))));
    }
    positions[0] = Vec3(1, 1, 1);
    positions[32] = Vec3(2.15, 1, 1);
    system.addForce(force);
    system.setDefaultPeriodicBoxVectors(Vec3(160, 0, 0), Vec3(0, 160, 0), Vec3(0, 0, 160));
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform, {{"Precision", precision}});
    context.setPositions(positions);
    MetalContext& cc = platform.metal(context);
    MetalNonbondedUtilities& nb = cc.getNonbondedUtilities();
    vector<mm_float4> pos(cc.getPaddedNumAtoms());
    cc.getPosq().download(pos);
    for (int step = 0; step <= 4; step++) {
        pos[0].x = 1.0f+0.03f*step;
        pos[32].x = 2.15f-0.03f*step;
        cc.getPosq().upload(pos);
        nb.prepareInteractions(-1);
        vector<unsigned int> counts;
        nb.getInteractionCount().download(counts);
        ASSERT(counts[0] <= nb.getInteractingTiles().getSize());
        ASSERT(counts[1] <= nb.getSinglePairs().getSize());
        vector<int> tiles, atoms;
        vector<mm_int2> pairs;
        nb.getInteractingTiles().download(tiles);
        nb.getInteractingAtoms().download(atoms);
        nb.getSinglePairs().download(pairs);
        bool covered = false;
        for (int i = 0; i < counts[0]; i++)
            for (int j = 0; j < 32; j++)
                covered |= (tiles[i] == 0 && atoms[32*i+j] == 32) || (tiles[i] == 1 && atoms[32*i+j] == 0);
        for (int i = 0; i < counts[1]; i++)
            covered |= (pairs[i].x == 0 && pairs[i].y == 32) || (pairs[i].x == 32 && pairs[i].y == 0);
        ASSERT_EQUAL(step >= 2, covered);
        vector<int> rebuilt;
        nb.getRebuildNeighborList().download(rebuilt);
        ASSERT_EQUAL(step%2 == 0, rebuilt[0] != 0);
    }
}

int main(int argc, char* argv[]) {
    try {
        testDisplacementBounds(argc > 1 ? argv[1] : "single");
    }
    catch (const exception& e) {
        cerr << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
