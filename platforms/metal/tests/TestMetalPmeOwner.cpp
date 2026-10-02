// Owner PME correctness and an explicitly requested, spline-free kill replay.
// The platform hides its metal-cpp selector symbols.  This executable uses
// command-buffer timestamps directly, so it needs its own implementation.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define METALCPP_SYMBOL_VISIBILITY_HIDDEN
#include "MetalContext.h"
#include "MetalQueue.h"
#include "CommonKernelSources.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/Context.h"
#include "openmm/NonbondedForce.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/VirtualSite.h"
#include "openmm/internal/AssertionUtilities.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <vector>

using namespace OpenMM;
using namespace std;
static MetalPlatform platform;

struct OwnerSwitch {
    bool existed;
    string saved;
    OwnerSwitch(bool enabled) : existed(getenv("OPENMM_METAL_PME_OWNER") != nullptr) {
        if (existed) saved = getenv("OPENMM_METAL_PME_OWNER");
        setenv("OPENMM_METAL_PME_OWNER", enabled ? "1" : "0", 1);
    }
    ~OwnerSwitch() {
        if (existed) setenv("OPENMM_METAL_PME_OWNER", saved.c_str(), 1);
        else unsetenv("OPENMM_METAL_PME_OWNER");
    }
};

struct Fixture {
    int nx=15, ny=16, nz=17;
    Vec3 a=Vec3(3, 0, 0), b=Vec3(0.2, 3, 0), c=Vec3(-0.1, 0.3, 3);
    vector<mm_float4> pos;
};

Fixture waterBox() {
    Fixture f;
    // TIP4P-Ew geometry and charges, with molecules crossing periodic seams.
    const Vec3 h1(0.09572, 0, 0), h2(-0.02399872, 0.09266272, 0);
    const double w = 0.106676721;
    for (int x=0; x<4; x++) for (int y=0; y<4; y++) for (int z=0; z<4; z++) {
        Vec3 o = f.a*((x+0.97)/4)+f.b*((y+0.97)/4)+f.c*((z+0.97)/4);
        Vec3 p[] = {o, o+h1, o+h2, o+(h1+h2)*w};
        double q[] = {0, 0.52422, 0.52422, -1.04844};
        for (int j=0; j<4; j++) f.pos.push_back(mm_float4(p[j][0],p[j][1],p[j][2],q[j]));
    }
    return f;
}

void addBox(ComputeKernel kernel, const Fixture& f) {
    const double ax=f.a[0], by=f.b[1], cz=f.c[2];
    kernel->addArg(mm_float4(ax,by,cz,0));
    kernel->addArg(mm_float4(1/ax,1/by,1/cz,0));
    for (auto v : {f.a,f.b,f.c}) kernel->addArg(mm_float4(v[0],v[1],v[2],0));
    kernel->addArg(mm_float4(1/ax,0,0,0));
    kernel->addArg(mm_float4(-f.b[0]/(ax*by),1/by,0,0));
    kernel->addArg(mm_float4((f.b[0]*f.c[1]-by*f.c[0])/(ax*by*cz),-f.c[1]/(by*cz),1/cz,0));
}

struct Spread {
    MetalContext& cc;
    Fixture f;
    int bins, points;
    ComputeArray positions, heads, atoms, splines, grid, fixed, reference, order, charges;
    ComputeKernel bin, gather, atomic, finish;
    Spread(MetalContext& cc, const Fixture& f, bool kill=false) : cc(cc), f(f),
            bins(((f.nx+3)/4)*((f.ny+3)/4)*((f.nz+3)/4)), points(f.nx*f.ny*f.nz) {
        int n=f.pos.size();
        positions.initialize<mm_float4>(cc,n,"ownerTestPositions"); positions.upload(f.pos);
        heads.initialize<int>(cc,bins,"ownerTestHeads");
        atoms.initialize<mm_int4>(cc,n,"ownerTestAtoms");
        splines.initialize<mm_float4>(cc,5*n,"ownerTestSplines");
        grid.initialize<float>(cc,points,"ownerTestGrid");
        fixed.initialize<long long>(cc,points,"ownerTestFixed");
        reference.initialize<float>(cc,points,"ownerTestReference");
        charges.initialize<float>(cc,n,"ownerTestCharges");
        order.initialize<mm_int2>(cc,n,"ownerTestOrder");
        vector<mm_int2> indices;
        for (int i=0; i<n; i++) indices.push_back(mm_int2(i,0));
        order.upload(indices);
        auto d=defines();
        if (kill) d["OWNER_PME_KILL"]="1";
        auto program=cc.compileProgram(cc.replaceStrings(CommonKernelSources::pme, {{"CHARGE","pos.w"}}),d);
        bin=program->createKernel("binPmeOwner");
        bin->addArg(positions); bin->addArg(heads); bin->addArg(atoms); bin->addArg(splines); bin->addArg(charges); addBox(bin,f);
        gather=makeGather(program);
        atomic=program->createKernel("gridSpreadCharge");
        atomic->addArg(positions); atomic->addArg(fixed); addBox(atomic,f); atomic->addArg(order); atomic->addArg(charges);
        finish=program->createKernel("finishSpreadCharge"); finish->addArg(fixed); finish->addArg(reference);
    }
    map<string,string> defines() {
        return {{"PME_OWNER_SPREAD","1"},{"OWNER_CHARGE","pos.w"},{"USE_FIXED_POINT_CHARGE_SPREADING","1"},
            {"NUM_ATOMS",to_string(f.pos.size())},{"PADDED_NUM_ATOMS",to_string(cc.getPaddedNumAtoms())},
            {"PME_ORDER","5"},{"NUM_INDICES","0"},{"GRID_SIZE_X",to_string(f.nx)},
            {"GRID_SIZE_Y",to_string(f.ny)},{"GRID_SIZE_Z",to_string(f.nz)},
            {"EPSILON_FACTOR","11.7870886f"},{"RECIP_EXP_FACTOR","1.0f"},{"M_PI","3.14159265358979323846f"}};
    }
    ComputeKernel makeGather(ComputeProgram program) {
        auto k=program->createKernel("gatherPmeOwner");
        k->addArg(heads); k->addArg(atoms); k->addArg(splines); k->addArg(grid);
        return k;
    }
    void build() { cc.clearBuffer(heads); bin->execute(f.pos.size()); }
    void spread() { build(); gather->execute(64*bins,64); }
    vector<float> check() {
        spread();
        cc.clearBuffer(fixed); atomic->execute(5*f.pos.size()); finish->execute(points);
        vector<float> actual, expected; grid.download(actual); reference.download(expected);
        // Integer contributions and conversion match the atomic fixed-point oracle.
        ASSERT(memcmp(actual.data(),expected.data(),points*sizeof(float)) == 0);
        double sum=accumulate(actual.begin(),actual.end(),0.0);
        double q=0; for (auto p : f.pos) q += p.w*11.7870886f;
        ASSERT(fabs(sum-q) < 2e-5*f.pos.size());
        return actual;
    }
};

void testGrid(const string& precision) {
    Fixture f=waterBox();
    System system; for (auto p : f.pos) system.addParticle(1);
    MetalPlatform::PlatformData data(nullptr,system,"","true",precision,"false",platform.getPropertyDefaultValue(MetalPlatform::MetalTempDirectory()),"true","false",1,nullptr);
    MetalContext& cc=*data.contexts[0]; cc.initialize();
    Spread s(cc,f);
    vector<float> first=s.check();
    for (int i=0; i<3; i++) {
        s.spread(); vector<float> repeated; s.grid.download(repeated);
        ASSERT(memcmp(first.data(),repeated.data(),first.size()*sizeof(float)) == 0);
    }
    // Empty bins must overwrite old grid values when charged atoms move or vanish.
    for (auto& p : f.pos) { p.x += 1.123f; p.z -= 3.01f; if (p.w > 0) p.w=0; }
    s.positions.upload(f.pos); s.f.pos=f.pos; s.check();
    // Two-bin periodic axes require visiting both bins even when the halo starts
    // inside the owner's bin.  Nonmultiples of four also exercise partial tiles.
    f.nx=5; f.ny=6; f.nz=7;
    Spread tiny(cc,f); tiny.check();
}

vector<Vec3> forces(const string& precision, bool owner, bool deterministic, bool queue, bool offsets) {
    OwnerSwitch flag(owner);
    Fixture f=waterBox();
    System system; system.setDefaultPeriodicBoxVectors(f.a,f.b,f.c);
    auto* nb=new NonbondedForce(); system.addForce(nb);
    nb->setNonbondedMethod(NonbondedForce::PME); nb->setCutoffDistance(0.9);
    nb->setPMEParameters(3,16,16,16); nb->setIncludeDirectSpace(false);
    if (offsets) nb->addGlobalParameter("chargeOffset",0.01);
    vector<Vec3> positions;
    for (int i=0; i<(int)f.pos.size(); i++) {
        auto p=f.pos[i]; system.addParticle(i%4 == 3 ? 0 : 1);
        nb->addParticle(p.w,1,0); positions.push_back(Vec3(p.x,p.y,p.z));
        if (offsets && i%4 == 1) nb->addParticleParameterOffset("chargeOffset",i,1,0,0);
        if (offsets && i%4 == 3) nb->addParticleParameterOffset("chargeOffset",i,-1,0,0);
        if (i%4 == 3) system.setVirtualSite(i,new ThreeParticleAverageSite(i-3,i-2,i-1,1-2*0.106676721,0.106676721,0.106676721));
    }
    VerletIntegrator integrator(0.001);
    Context context(system,integrator,platform,{{"Precision",precision},{"DeterministicForces",deterministic ? "true":"false"},
            {"DisablePmeStream",queue ? "false":"true"}});
    context.setPositions(positions); context.computeVirtualSites();
    auto result=context.getState(State::Forces).getForces();
    if (owner && deterministic) {
        for (int repeat=0; repeat<3; repeat++) {
            auto next=context.getState(State::Forces).getForces();
            ASSERT(memcmp(result.data(),next.data(),result.size()*sizeof(Vec3)) == 0);
        }
    }
    // Energy evaluations use the default queue; the next force-only evaluation
    // returns to the PME queue.  Both transitions must leave the same grid.
    auto energy=context.getState(State::Forces|State::Energy);
    ASSERT(isfinite(energy.getPotentialEnergy()));
    for (int i=0; i<(int)result.size(); i++) ASSERT_EQUAL_VEC(result[i],energy.getForces()[i],1e-5);
    auto next=context.getState(State::Forces).getForces();
    for (int i=0; i<(int)result.size(); i++) ASSERT_EQUAL_VEC(result[i],next[i],1e-5);
    return result;
}

void testForces(const string& precision) {
    for (bool offsets : {false,true}) for (bool queue : {false,true}) for (bool deterministic : {false,true}) {
        auto ref=forces(precision,false,deterministic,queue,offsets);
        auto owned=forces(precision,true,deterministic,queue,offsets);
        if (deterministic) {
            auto fresh=forces(precision,true,true,queue,offsets);
            ASSERT(memcmp(owned.data(),fresh.data(),owned.size()*sizeof(Vec3)) == 0);
        }
        double err=0, norm=0;
        for (int i=0; i<(int)ref.size(); i++) { Vec3 d=ref[i]-owned[i]; err+=d.dot(d); norm+=ref[i].dot(ref[i]); }
        ASSERT(sqrt(err/max(norm,1e-30)) < (deterministic ? 1e-6 : 1e-4));
    }
}

// Use command-buffer GPU timestamps so CPU encoding time cannot falsely kill a
// design.  All repetitions share one serial command buffer and one queue.
double gpuTime(MetalContext& cc, const function<void()>& work) {
    auto& q=cc.getCurrentStream();
    for (int i=0; i<5; i++) work();
    q.finish();
    double best=1e30;
    for (int sample=0; sample<5; sample++) {
        MTL::CommandBuffer* buffer;
        {
            lock_guard<recursive_mutex> lock(q.getLock());
            buffer=q.getCommandBuffer()->retain();
            for (int i=0; i<100; i++) work();
            q.commit();
        }
        buffer->waitUntilCompleted();
        double us=(buffer->GPUEndTime()-buffer->GPUStartTime())*1e6/100;
        bool ok=buffer->status() == MTL::CommandBufferStatusCompleted && us > 0;
        buffer->release();
        ASSERT(ok); best=min(best,us);
    }
    return best;
}

int kill(const string& precision, const string& path) {
    Fixture f;
    ifstream in(path);
    int n; in >> n >> f.nx >> f.ny >> f.nz;
    for (Vec3* v : {&f.a,&f.b,&f.c}) for (int i=0; i<3; i++) in >> (*v)[i];
    ASSERT(in && n > 0 && f.nx >= 5 && f.ny >= 5 && f.nz >= 5);
    f.pos.resize(n);
    for (auto& p : f.pos) in >> p.x >> p.y >> p.z >> p.w;
    ASSERT(in);
    System system; for (int i=0; i<n; i++) system.addParticle(1);
    MetalPlatform::PlatformData data(nullptr,system,"","true",precision,"false",platform.getPropertyDefaultValue(MetalPlatform::MetalTempDirectory()),"true","false",1,nullptr);
    MetalContext& cc=*data.contexts[0]; cc.initialize();
    // Resolve the mesh with the same device-dependent rule as production.
    f.nx=cc.findLegalFFTDimension(f.nx); f.ny=cc.findLegalFFTDimension(f.ny); f.nz=cc.findLegalFFTDimension(f.nz);
    Spread s(cc,f,true);
    auto d=s.defines(); d["OWNER_PME_KILL"]="1"; d["OWNER_PME_INTERIOR_ONLY"]="1";
    auto interior=s.makeGather(cc.compileProgram("",d));
    d["OWNER_PME_WRITE_ONLY"]="1";
    auto write=s.makeGather(cc.compileProgram("",d));
    s.build();
    double bin=gpuTime(cc,[&] { s.build(); });
    double writeUs=gpuTime(cc,[&] { write->execute(64*s.bins,64); });
    double inside=gpuTime(cc,[&] { interior->execute(64*s.bins,64); });
    double gather=gpuTime(cc,[&] { s.gather->execute(64*s.bins,64); });
    double combined=gpuTime(cc,[&] { s.spread(); });
    // The count output verifies periodic membership on this captured unit.
    vector<float> grid; s.grid.download(grid);
    double memberships=accumulate(grid.begin(),grid.end(),0.0)*4294967296.0;
    int charged=count_if(f.pos.begin(),f.pos.end(),[](mm_float4 p) { return p.w != 0; });
    ASSERT_EQUAL(125.0*charged,memberships);
    cout << fixed << setprecision(3) << "atoms=" << n << " charged=" << charged << " grid=" << f.nx << 'x' << f.ny << 'x' << f.nz << " bins=" << s.bins << '\n'
         << "us: bin(reset+insert)=" << bin << " gather(interior)=" << inside-writeUs << " halo=" << gather-inside << " write=" << writeUs << '\n'
         << "optimistic sum=" << bin << "+" << inside-writeUs << "+" << gather-inside << "+" << writeUs << "=" << bin+gather
         << " combined=" << combined << " threshold=40.000\n";
    // Use the combined replay for the decision, avoiding a false kill from
    // isolated-dispatch overhead.  Component subtraction is diagnostic only.
    cout << (combined > 40 ? "DEAD" : "SURVIVES LOWER BOUND; full-path timing still required") << endl;
    return combined > 40 ? 2 : 0;
}

int main(int argc, char** argv) {
    try {
        string precision=argc > 1 ? argv[1] : "single";
        if (argc == 4 && string(argv[2]) == "--kill") return kill(precision,argv[3]);
        testGrid(precision); testForces(precision);
        cout << "Done" << endl;
    }
    catch (const exception& e) { cerr << e.what() << endl; return 1; }
    return 0;
}
