/* Portions copyright (c) 2026 the Authors. Licensed under the MIT license. */
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "MetalContext.h"
#include "MetalPlatform.h"
#include "MetalKernels.h"
#include "MetalQueue.h"
#include "openmm/Context.h"
#include "openmm/KernelFactory.h"
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
#include <iomanip>
#include <memory>
#include <limits>
#include <random>
#include <sstream>
using namespace OpenMM;
using namespace std;

class TrialPlatform : public MetalPlatform {
public:
    ContextImpl& impl(Context& c) const { return getContextImpl(c); }
    MetalContext& metal(Context& c) const {
        return *static_cast<PlatformData*>(impl(c).getPlatformData())->contexts[0];
    }
    mutable MetalApplyMonteCarloBarostatKernel* last = nullptr;
};
class TrialFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(string name, const Platform& p, ContextImpl& c) const override {
        auto& platform = static_cast<const TrialPlatform&>(p);
        auto& cu = *static_cast<MetalPlatform::PlatformData*>(c.getPlatformData())->contexts[0];
        platform.last = new MetalApplyMonteCarloBarostatKernel(name, p, cu);
        return platform.last;
    }
};
static void deviceMode(bool enabled) {
    setenv("OPENMM_METAL_DEVICE_BOX", "1", 1);
    setenv("OPENMM_METAL_DEVICE_BAROSTAT", enabled ? "1" : "0", 1);
}
static double volume(const State& s) {
    Vec3 a,b,c; s.getPeriodicBoxVectors(a,b,c); return a[0]*b[1]*c[2];
}
static vector<char> bytes(MetalArray& a) {
    vector<char> b(a.getSize()*a.getElementSize()); a.download(b.data()); return b;
}
static vector<vector<char>> snapshot(MetalContext& c) {
    vector<vector<char>> result = {bytes(c.unwrap(c.getPosq())), bytes(c.unwrap(c.getLongForceBuffer())),
        bytes(c.getDeviceBoxState()), bytes(c.getDeviceCellOffsets()), bytes(c.unwrap(c.getAtomIndexArray()))};
    if (c.getUseMixedPrecision()) result.push_back(bytes(c.unwrap(c.getPosqCorrection())));
    return result;
}
static void sameState(const State& a, const State& b) {
    ASSERT_EQUAL(a.getPositions().size(), b.getPositions().size());
    for (int i=0; i<a.getPositions().size(); i++) {
        ASSERT(memcmp(&a.getPositions()[i], &b.getPositions()[i], sizeof(Vec3)) == 0);
        ASSERT(memcmp(&a.getForces()[i], &b.getForces()[i], sizeof(Vec3)) == 0);
        ASSERT(memcmp(&a.getVelocities()[i], &b.getVelocities()[i], sizeof(Vec3)) == 0);
    }
    Vec3 av[3], bv[3]; a.getPeriodicBoxVectors(av[0],av[1],av[2]); b.getPeriodicBoxVectors(bv[0],bv[1],bv[2]);
    ASSERT(memcmp(av,bv,sizeof(av)) == 0);
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
static MonteCarloBarostat& addBarostat(System& s, int frequency, int seed) {
    auto* b = new MonteCarloBarostat(1, 300, frequency); b->setRandomNumberSeed(seed); s.addForce(b); return *b;
}
static void checkpoint(TrialPlatform& p, System& s, Context& c, MetalApplyMonteCarloBarostatKernel& kernel) {
    stringstream data;
    c.createCheckpoint(data);
    auto expected = c.getState(State::Positions|State::Velocities|State::Forces);
    deviceMode(true);
    LangevinMiddleIntegrator integrator(300,1,0.001);
    Context restored(s, integrator, p);
    auto* restoredKernel = p.last;
    restored.loadCheckpoint(data);
    sameState(expected, restored.getState(State::Positions|State::Velocities|State::Forces));
    ASSERT(kernel.getTrialState() == restoredKernel->getTrialState());
}
static void parity(TrialPlatform& p) {
    System system; vector<Vec3> positions; createWater(system, positions, 17);
    addBarostat(system, 1, 9123);
    deviceMode(false);
    LangevinMiddleIntegrator hi(300,1,0.001); Context host(system,hi,p); auto* hk=p.last;
    host.setPositions(positions); host.computeVirtualSites();
    deviceMode(true);
    LangevinMiddleIntegrator di(300,1,0.001); Context device(system,di,p); auto* dk=p.last;
    device.setPositions(positions); device.computeVirtualSites();
    double scale=volume(host.getState(0))*0.01;
    int attempted=0, accepted=0, accepts=0, rejects=0;
    bool checkedAccept=false, checkedReject=false;
    mt19937 random(918273); const double kT=0.00831446261815324*300, pressure=0.0602214076;
    double maxError=0, maxReductionError=0;
    for (int trial=0; trial<200; trial++) {
        // Import on both sides.  The first setPositions() reorders and wraps; a later setState()
        // unwraps and usually does not reorder.  Comparing only one imported context would compare
        // different float coordinate representations, even in mixed mode (forces read posq).
        State start=host.getState(State::Positions|State::Velocities);
        int wrappedBefore=0;
        if (trial==0)
            for (auto o:p.metal(host).getPosCellOffsets()) wrappedBefore += o.x!=0 || o.y!=0 || o.z!=0;
        host.setState(start);
        device.setState(start);
        host.getState(State::Forces);
        device.getState(State::Forces);
        if (trial==0) {
            int wrappedHost=0,wrappedDevice=0;
            for (auto o:p.metal(host).getPosCellOffsets()) wrappedHost += o.x!=0 || o.y!=0 || o.z!=0;
            for (auto o:p.metal(device).getPosCellOffsets()) wrappedDevice += o.x!=0 || o.y!=0 || o.z!=0;
            cout << "initial cell offsets: host before import=" << wrappedBefore
                 << " host after=" << wrappedHost << " device after=" << wrappedDevice << endl;
        }
        auto saved=snapshot(p.metal(device));
        double u=double(random())/4294967296.0, v=double(random())/4294967296.0;
        hk->setTrialUniforms(u,v); dk->setTrialUniforms(u,v);
        double old=volume(start), delta=scale*2*(u-0.5);
        p.impl(host).updateContextState();
        p.impl(device).updateContextState();
        auto ds=dk->getTrialState(); ASSERT_EQUAL(16, ds.size());
        double w=hk->testFinalEnergy-hk->testInitialEnergy+pressure*delta-(system.getNumParticles()/4)*kT*log((old+delta)/old);
        bool ha=!(w>0 && v>exp(-w/kT)), da=ds[6]!=0;
        double error=abs(w-ds[3])/kT; maxError=max(maxError,error);
        double e0,e1; dk->readTrialEnergies(e0,e1);
        double cpuWork=e1-e0+pressure*ds[2]-(system.getNumParticles()/4)*kT*log((ds[1]+ds[2])/ds[1]);
        double reductionError=abs(cpuWork-ds[3])/kT;
        maxReductionError=max(maxReductionError,reductionError);
        ASSERT(reductionError < 1e-3);
        if (ha!=da) {
            cout << "boundary trial=" << trial << " |w_host|/kT=" << abs(w)/kT
                 << " |w_device|/kT=" << abs(ds[3])/kT << " difference/kT=" << error << endl;
            ASSERT(error < 1e-3);
            double threshold=-kT*log(v);
            ASSERT(threshold>=min(w,ds[3]) && threshold<=max(w,ds[3]));
        }
        if (error >= 1e-3 && p.metal(device).getUseMixedPrecision())
            cout << setprecision(16) << "trial=" << trial << " u=" << u << " v=" << v
                 << " host w=" << w << " device w=" << ds[3] << " error/kT=" << error
                 << " host energies=" << hk->testInitialEnergy << "," << hk->testFinalEnergy
                 << " device energies=" << ds[4] << "," << ds[5]
                 << " host V/dV=" << old << "," << delta << " device V/dV=" << ds[1] << "," << ds[2] << endl;
        if (p.metal(device).getUseMixedPrecision()) ASSERT(error < 1e-3);
        if (!da) { ASSERT(saved == snapshot(p.metal(device))); rejects++; }
        else accepts++;
        if ((da && !checkedAccept) || (!da && !checkedReject)) {
            checkpoint(p,system,device,*dk);
            if (da) checkedAccept=true; else checkedReject=true;
        }
        attempted++; if (ha) accepted++;
        if (attempted>=10) {
            if (accepted<0.25*attempted) {scale/=1.1;attempted=accepted=0;}
            else if (accepted>0.75*attempted) {scale=min(scale*1.1,old*0.3);attempted=accepted=0;}
        }
    }
    cout << "parity trials=200 accepts=" << accepts << " rejects=" << rejects << " max |delta w|/kT=" << maxError << " same-buffer error/kT=" << maxReductionError << endl;
    ASSERT(accepts>0 && rejects>0);
    ASSERT(maxReductionError < 1e-3);
}
static void fallbacks(TrialPlatform& p) {
    for (int mode=0; mode<3; mode++) {
        bool rigid=mode!=0;
        System system; vector<Vec3> positions;
        if (mode==2) {
            system.setDefaultPeriodicBoxVectors(Vec3(9,0,0),Vec3(0,9,0),Vec3(0,0,9));
            auto* force=new NonbondedForce();
            force->setNonbondedMethod(NonbondedForce::PME);
            force->setCutoffDistance(0.9);
            system.addForce(force);
            for (int i=0; i<4097; i++) {
                system.addParticle(1);
                force->addParticle(0,0.2+1e-5*i,0.1);
                positions.push_back(Vec3(0.5*(i%17),0.5*((i/17)%17),0.5*(i/289)));
            }
        } else
            createWater(system,positions,7);
        auto& barostat=addBarostat(system,1,741);
        barostat.setScaleMoleculesAsRigid(rigid);
        deviceMode(true);
        setenv("OPENMM_METAL_DEFER_BAROSTAT_ENERGIES", "1", 1);
        LangevinMiddleIntegrator integrator(300,1,0.001);
        Context c(system,integrator,p); auto* kernel=p.last;
        auto& nb=p.metal(c).getNonbondedUtilities();
        if (mode==1) nb.forceCountWaitForTesting();
        c.setPositions(positions); c.computeVirtualSites();
        if (mode==2) {
            ASSERT_EQUAL(4097,p.metal(c).getMoleculeGroups().size());
            ASSERT(nb.hasBoundedNeighborList());
        }
        stringstream messages;
        struct Capture {
            streambuf* previous;
            Capture(stringstream& s) : previous(cerr.rdbuf(s.rdbuf())) {}
            ~Capture() { cerr.rdbuf(previous); }
        };
        {
            Capture capture(messages);
            for (int trial=0; trial<2; trial++) {
                kernel->testInitialEnergy=numeric_limits<double>::quiet_NaN();
                kernel->setTrialUniforms(0.51,0.5);
                p.impl(c).updateContextState();
                ASSERT(isfinite(kernel->testInitialEnergy)); // Phase 1 supplied both energies.
            }
        }
        string reason=mode==0 ? "requires rigid molecule scaling" : mode==1 ?
                "neighbor list requires host overflow checks" : "reorder tables do not support this molecule layout";
        size_t first=messages.str().find(reason);
        ASSERT(first!=string::npos);
        ASSERT(messages.str().find(reason,first+reason.size())==string::npos);
        auto state=kernel->getTrialState();
        ASSERT(state.empty() || state[9]==0);
        cout << "fallback: " << reason << "; phase 1 used; reason printed once" << endl;
    }
    System system; vector<Vec3> positions; createWater(system,positions,7);
    addBarostat(system,1,741);
    for (bool enabled : {false,true}) {
        deviceMode(enabled);
        LangevinMiddleIntegrator sourceIntegrator(300,1,0.001);
        Context source(system,sourceIntegrator,p);
        source.setPositions(positions); source.computeVirtualSites();
        stringstream checkpoint;
        source.createCheckpoint(checkpoint);
        deviceMode(!enabled);
        LangevinMiddleIntegrator targetIntegrator(300,1,0.001);
        Context target(system,targetIntegrator,p);
        bool clearException=false;
        try { target.loadCheckpoint(checkpoint); }
        catch (const OpenMMException& e) {
            clearException=string(e.what()).find("OPENMM_METAL_DEVICE_BAROSTAT")!=string::npos;
        }
        ASSERT(clearException);
    }
    cout << "checkpoint flag mismatch: clear exception in both directions" << endl;
}
// A restored context rebuilds its neighbor list, so float energy sums (fields 3-5) may round differently even
// with deterministic forces.  The proposal, decision, adaptation and RNG cursor must still match exactly.
static void sameTrialState(const vector<double>& a, const vector<double>& b) {
    ASSERT(a.size()==b.size() && a.size()==16);
    for (int i=0; i<16; i++)
        if (i<3 || i>5)
            ASSERT(a[i]==b[i]);
    const double kT=0.00831446261815324*300;
    ASSERT(fabs(a[3]-b[3]) < 0.1*kT); // Work is in kJ/mol; parity noise is about 0.02 kT.
}
static void checkpointEligibility(TrialPlatform& p) {
    System system; vector<Vec3> positions; createWater(system,positions,7);
    addBarostat(system,1,741);
    deviceMode(true);
    LangevinMiddleIntegrator si(300,1,0.001), ti(300,1,0.001), ri(300,1,0.001);
    Context source(system,si,p); auto* sk=p.last;
    source.setPositions(positions); source.computeVirtualSites();
    for (int trial=0; trial<12; trial++) p.impl(source).updateContextState();
    stringstream checkpoint;
    source.createCheckpoint(checkpoint);
    Context target(system,ti,p);
    target.setPositions(positions);
    // Register storage before forcing fallback, so load must refresh the previous eligibility.
    stringstream beforeFallback;
    target.createCheckpoint(beforeFallback);
    p.metal(target).getNonbondedUtilities().forceCountWaitForTesting();
    bool clearException=false;
    try { target.loadCheckpoint(checkpoint); }
    catch (const OpenMMException& e) {
        clearException=string(e.what()).find("OPENMM_METAL_DEVICE_BAROSTAT")!=string::npos;
    }
    ASSERT(clearException);
    checkpoint.clear(); checkpoint.seekg(0);
    Context restored(system,ri,p); auto* rk=p.last;
    restored.loadCheckpoint(checkpoint);
    ASSERT(sk->getTrialState()==rk->getTrialState());
    for (int trial=0; trial<3; trial++) {
        p.impl(source).updateContextState();
        p.impl(restored).updateContextState();
        sameTrialState(sk->getTrialState(), rk->getTrialState());
        sameState(source.getState(State::Positions|State::Velocities|State::Forces),
                  restored.getState(State::Positions|State::Velocities|State::Forces));
    }
    // A context that switches to phase 1 must also write the phase 1 checkpoint format.
    p.metal(source).getNonbondedUtilities().forceCountWaitForTesting();
    stringstream fallbackCheckpoint;
    source.createCheckpoint(fallbackCheckpoint);
    clearException=false;
    try { restored.loadCheckpoint(fallbackCheckpoint); }
    catch (const OpenMMException& e) {
        clearException=string(e.what()).find("OPENMM_METAL_DEVICE_BAROSTAT")!=string::npos;
    }
    ASSERT(clearException);
    fallbackCheckpoint.clear(); fallbackCheckpoint.seekg(0);
    target.loadCheckpoint(fallbackCheckpoint);
    cout << "checkpoint eligibility: rejects both mode mismatches; eligible continuation bitwise; phase 1 loads" << endl;
}
static void noWait(TrialPlatform& p) {
    System system; vector<Vec3> positions; createWater(system,positions,7);
    auto& barostat=addBarostat(system,25,654321);
    deviceMode(true);
    LangevinMiddleIntegrator integrator(300,1,0.001); integrator.setRandomNumberSeed(7171);
    Context c(system,integrator,p); auto* kernel=p.last;
    c.setPositions(positions); c.setVelocitiesToTemperature(300,1234); c.computeVirtualSites();
    integrator.step(25); c.getState(State::Energy);
    for (int frequency : {0,25}) {
        barostat.setFrequency(frequency);
        auto before=MetalQueue::getFinishCount();
        auto countBefore=p.metal(c).getNonbondedUtilities().getCountWaitCalls();
        auto commits=p.metal(c).getCurrentStream().getCommitCount();
        integrator.step(1000);
        auto finishes=MetalQueue::getFinishCount()-before;
        cout << "steps=1000 frequency=" << frequency << " finishes=" << finishes
             << " commits=" << p.metal(c).getCurrentStream().getCommitCount()-commits << endl;
        ASSERT_EQUAL(0,finishes);
        ASSERT_EQUAL(countBefore,p.metal(c).getNonbondedUtilities().getCountWaitCalls());
        c.getState(State::Energy);
    }
    auto state=kernel->getTrialState(); ASSERT_EQUAL(41, state[9]);
    ASSERT(state[11]>=state[9] && state[11]<=2*state[9]);
}
struct Estimate { double value, se; };
static Estimate blocks(const vector<double>& samples, int width) {
    vector<double> means;
    for (int start=0; start+width<=samples.size(); start+=width) {
        double mean=0; for (int i=0;i<width;i++) mean+=samples[start+i]/width;
        means.push_back(mean);
    }
    ASSERT(means.size()>=10);
    double mean=0, variance=0;
    for (double v:means) mean+=v/means.size();
    for (double v:means) variance+=(v-mean)*(v-mean)/(means.size()-1);
    return {mean,sqrt(variance/means.size())};
}
struct Statistics { Estimate volume, variance, acceptance; };
static Statistics sample(TrialPlatform& p, System& system, const State& initial,
        MonteCarloBarostat& barostat, bool device, int steps) {
    deviceMode(device);
    barostat.setRandomNumberSeed(device ? 389231 : 812891);
    LangevinMiddleIntegrator integrator(300,1,0.002);
    integrator.setRandomNumberSeed(device ? 123989 : 728391);
    Context c(system,integrator,p); auto* kernel=p.last;
    c.setState(initial);
    vector<double> volumes, accepts;
    double last=volume(c.getState(0));
    for (int i=0; i<steps/25; i++) {
        integrator.step(25);
        double v=volume(c.getState(0));
        volumes.push_back(v); accepts.push_back(v!=last ? 1 : 0); last=v;
    }
    int width=steps/25/20;
    Estimate mean=blocks(volumes,width);
    vector<double> deviations;
    for (double v:volumes) deviations.push_back((v-mean.value)*(v-mean.value));
    if (device) {
        auto state=kernel->getTrialState(); ASSERT_EQUAL(steps/25,state[9]);
        double count=0; for(double v:accepts) count+=v;
        ASSERT_EQUAL(count,state[10]);
    }
    return {mean,blocks(deviations,width),blocks(accepts,width)};
}
static void compareEstimate(const string& name, Estimate a, Estimate b) {
    double se=hypot(a.se,b.se), difference=abs(a.value-b.value);
    cout << name << " host=" << a.value << " device=" << b.value
         << " combined SE=" << se << " difference/SE=" << difference/se << endl;
    ASSERT(difference<=3*se);
}
static void statistics(TrialPlatform& p, const string& systemPath, const string& statePath) {
    unique_ptr<System> system;
    unique_ptr<State> supplied;
    vector<Vec3> positions;
    if (systemPath.empty()) {
        system.reset(new System()); createWater(*system,positions,17);
        ASSERT_EQUAL(19652,system->getNumParticles());
    } else {
        ifstream sf(systemPath), st(statePath);
        ASSERT(sf && st);
        system.reset(XmlSerializer::deserialize<System>(sf));
        supplied.reset(XmlSerializer::deserialize<State>(st));
        for (int i=system->getNumForces()-1;i>=0;i--)
            if (dynamic_cast<MonteCarloBarostat*>(&system->getForce(i))) system->removeForce(i);
    }
    auto& barostat=addBarostat(*system,25,789);
    deviceMode(false);
    State initial;
    {
        LangevinMiddleIntegrator equil(300,1,0.002); equil.setRandomNumberSeed(45678);
        Context c(*system,equil,p);
        if (supplied) c.setState(*supplied);
        else { c.setPositions(positions); c.computeVirtualSites(); c.setVelocitiesToTemperature(300,9812); }
        equil.step(supplied ? 1000 : 10000);
        initial=c.getState(State::Positions|State::Velocities|State::Parameters);
    }
    int steps=supplied ? 5000 : 20000;
    cout << "statistics atoms=" << system->getNumParticles() << " steps/arm=" << steps << endl;
    auto host=sample(p,*system,initial,barostat,false,steps);
    auto device=sample(p,*system,initial,barostat,true,steps);
    compareEstimate("mean volume",host.volume,device.volume);
    compareEstimate("volume variance",host.variance,device.variance);
    compareEstimate("acceptance",host.acceptance,device.acceptance);
}

static void fahEligibility(TrialPlatform& p, const char* systemFile, const char* stateFile, const char* integratorFile) {
    ifstream sf(systemFile), st(stateFile), it(integratorFile);
    unique_ptr<System> system(XmlSerializer::deserialize<System>(sf));
    unique_ptr<State> state(XmlSerializer::deserialize<State>(st));
    unique_ptr<Integrator> integrator(XmlSerializer::deserialize<Integrator>(it));
    deviceMode(true);
    Context c(*system,*integrator,p); auto* kernel=p.last;
    c.setState(*state);
    ASSERT(p.metal(c).getNonbondedUtilities().hasBoundedNeighborList());
    integrator->step(25);
    auto trial=kernel->getTrialState();
    ASSERT(trial.size()==16 && trial[9]==1);
    ASSERT(isfinite(trial[3]));
    cout << "FAH eligibility atoms=" << system->getNumParticles()
         << " bounded list; device trials=" << trial[9]
         << " role order=" << p.metal(c).getNonbondedUtilities().getUseRoleOrder() << endl;
}
int main(int argc, char* argv[]) {
    try {
        TrialPlatform platform;
        platform.registerKernelFactory(ApplyMonteCarloBarostatKernel::Name(), new TrialFactory());
        platform.setPropertyDefaultValue(MetalPlatform::MetalDeterministicForces(), "true");
        string precision=argc>1 ? argv[1] : "all";
        int failures=0;
        for (string mode : {string("single"), string("mixed")}) {
            if (precision!="all" && precision!=mode) continue;
            platform.setPropertyDefaultValue(MetalPlatform::MetalPrecision(),mode);
            cout << "precision=" << mode << endl;
            if (argc>2 && string(argv[2])=="statistics") {
                statistics(platform,"","");
                if (argc==5) statistics(platform,argv[3],argv[4]);
            } else {
                try { parity(platform); } catch (const exception& e) { cerr << "parity: " << e.what() << endl; failures++; }
                try { noWait(platform); } catch (const exception& e) { cerr << "no-wait: " << e.what() << endl; failures++; }
                try { fallbacks(platform); } catch (const exception& e) { cerr << "fallbacks: " << e.what() << endl; failures++; }
                try { checkpointEligibility(platform); } catch (const exception& e) { cerr << "checkpoint eligibility: " << e.what() << endl; failures++; }
                if (argc==5) {
                    try { fahEligibility(platform,argv[2],argv[3],argv[4]); }
                    catch (const exception& e) { cerr << "FAH eligibility: " << e.what() << endl; failures++; }
                }
            }
        }
        if (failures) return 1;
        cout << "Done" << endl;
    } catch (const exception& e) {cerr << "exception: " << e.what() << endl; return 1;}
    return 0;
}
