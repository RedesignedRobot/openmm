/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "MetalCalcCustomGBForceKernel.h"
#include "MetalKernels.h"
#include "openmm/CustomGBForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"
#include "openmm/common/CommonCalcCustomGBForceKernel.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace OpenMM;
using namespace std;

namespace {

// The expressions openmm.app's customgbforces.GBSAOBC2Force and _createEnergyTerms write, with kappa 0.
const string OBC2_I = "select(step(r+sr2-or1), 0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r), 0);"
                      "U=r+sr2;L=max(or1, D);D=abs(r-sr2)";
const string OBC2_B = "1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);psi=I*or; radius=or+offset; offset=0.009";
const string SELF_ENERGY = "-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*charge^2/B";
const string ACE_ENERGY = "28.3919551*(radius+0.14)^2*(radius/B)^6; radius=or+offset";
const string PAIR_ENERGY = "-138.935485*(1/soluteDielectric-1/solventDielectric)*charge1*charge2";
const string PAIR_F = ";f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))";
const double OBC2_OFFSET = 0.009;   // GBSAOBCForce's fixed dielectric offset
// GBSAOBCForce adds OBC's term for an atom engulfed by a neighbor's scaled sphere, 2*(1/or1-1/L), which the openmm.app
// template leaves out.  It is nonzero only for pairs closer than sr2-or1, so the two agree whenever the largest sr
// minus the smallest or is under every bond length.  Protein and nucleic acid parameters give 0.053 nm (sulfur's sr
// against hydrogen's or); the shortest X-H bond is 0.096 nm.
const double MAX_ENGULF_DISTANCE = 0.09;
const double ACE_PREFACTOR = 28.3919551;   // 4*pi times GBSAOBCForce's default surface area energy, 2.25936 kJ/mol/nm^2

// Python's "%.16g", which _createEnergyTerms uses for the named constants.
string formatLikePython(double value) {
    char buffer[40];
    snprintf(buffer, sizeof(buffer), "%.16g", value);
    return buffer;
}

bool readConstant(const string& expression, const string& name, double& value) {
    size_t at = expression.find("; "+name+"=");
    if (at == string::npos)
        return false;
    const char* start = expression.c_str()+at+name.size()+3;
    char* end;
    value = strtod(start, &end);
    return end != start;
}

bool hasComputedValue(const CustomGBForce& force, int index, const string& expectedName, const string& expectedExpression,
        CustomGBForce::ComputationType expectedType) {
    string name, expression;
    CustomGBForce::ComputationType type;
    force.getComputedValueParameters(index, name, expression, type);
    return name == expectedName && expression == expectedExpression && type == expectedType;
}

bool hasEnergyTerm(const CustomGBForce& force, int index, const string& expectedExpression, CustomGBForce::ComputationType expectedType) {
    string expression;
    CustomGBForce::ComputationType type;
    force.getEnergyTermParameters(index, expression, type);
    return expression == expectedExpression && type == expectedType;
}

// With a cutoff the pair term shifts by 1/cutoff, which Python writes with str(), so read the number back instead of
// formatting it.
bool isShiftedPairTerm(const string& expression, const string& params, double cutoff) {
    string head = PAIR_ENERGY+"*(1/f-", tail = ")"+PAIR_F+params;
    if (expression.size() <= head.size()+tail.size() || expression.compare(0, head.size(), head) != 0 ||
            expression.compare(expression.size()-tail.size(), tail.size(), tail) != 0)
        return false;
    string shift = expression.substr(head.size(), expression.size()-head.size()-tail.size());
    char* end;
    double inverseCutoff = strtod(shift.c_str(), &end);
    return end != shift.c_str() && *end == '\0' && fabs(inverseCutoff*cutoff-1) < 1e-12;
}

} // namespace

MetalCalcCustomGBForceKernel::MetalCalcCustomGBForceKernel(string name, const Platform& platform, MetalContext& cu, const System& system) :
        CalcCustomGBForceKernel(name, platform), cu(cu) {
}

unique_ptr<GBSAOBCForce> MetalCalcCustomGBForceKernel::equivalentOBC(const CustomGBForce& force) {
    if (force.getNumPerParticleParameters() != 3 || force.getPerParticleParameterName(0) != "charge" ||
            force.getPerParticleParameterName(1) != "or" || force.getPerParticleParameterName(2) != "sr")
        return nullptr;
    if (force.getNumGlobalParameters() != 0 || force.getNumTabulatedFunctions() != 0 ||
            force.getNumEnergyParameterDerivatives() != 0 || force.getNumComputedValues() != 2)
        return nullptr;
    if (!hasComputedValue(force, 0, "I", OBC2_I, CustomGBForce::ParticlePairNoExclusions) ||
            !hasComputedValue(force, 1, "B", OBC2_B, CustomGBForce::SingleParticle))
        return nullptr;
    int numTerms = force.getNumEnergyTerms();
    if (numTerms != 2 && numTerms != 3)
        return nullptr;
    string selfTerm;
    CustomGBForce::ComputationType type;
    force.getEnergyTermParameters(0, selfTerm, type);
    double solventDielectric, soluteDielectric;
    if (!readConstant(selfTerm, "solventDielectric", solventDielectric) || !readConstant(selfTerm, "soluteDielectric", soluteDielectric))
        return nullptr;
    bool useCutoff = (force.getNonbondedMethod() != CustomGBForce::NoCutoff);
    string params = "; solventDielectric="+formatLikePython(solventDielectric)+"; soluteDielectric="+formatLikePython(soluteDielectric)+
            "; kappa=0; offset="+formatLikePython(OBC2_OFFSET);
    if (useCutoff)
        params += "; cutoff="+formatLikePython(force.getCutoffDistance());
    if (!hasEnergyTerm(force, 0, SELF_ENERGY+params, CustomGBForce::SingleParticle))
        return nullptr;
    bool useACE = (numTerms == 3);
    if (useACE && !hasEnergyTerm(force, 1, ACE_ENERGY+params, CustomGBForce::SingleParticle))
        return nullptr;
    string pairTerm;
    force.getEnergyTermParameters(numTerms-1, pairTerm, type);
    if (type != CustomGBForce::ParticlePairNoExclusions)
        return nullptr;
    if (useCutoff ? !isShiftedPairTerm(pairTerm, params, force.getCutoffDistance()) : pairTerm != PAIR_ENERGY+"/f"+PAIR_F+params)
        return nullptr;

    unique_ptr<GBSAOBCForce> obc(new GBSAOBCForce());
    if (force.getNonbondedMethod() == CustomGBForce::NoCutoff)
        obc->setNonbondedMethod(GBSAOBCForce::NoCutoff);
    else if (force.getNonbondedMethod() == CustomGBForce::CutoffNonPeriodic)
        obc->setNonbondedMethod(GBSAOBCForce::CutoffNonPeriodic);
    else
        obc->setNonbondedMethod(GBSAOBCForce::CutoffPeriodic);
    obc->setCutoffDistance(force.getCutoffDistance());
    obc->setSolventDielectric(solventDielectric);
    obc->setSoluteDielectric(soluteDielectric);
    obc->setSurfaceAreaEnergy(useACE ? ACE_PREFACTOR/(4*M_PI) : 0.0);
    obc->setForceGroup(force.getForceGroup());
    for (int i = 0; i < force.getNumParticles(); i++)
        obc->addParticle(0, 1, 1);
    if (!copyParticles(force, *obc))
        return nullptr;
    return obc;
}

bool MetalCalcCustomGBForceKernel::copyParticles(const CustomGBForce& force, GBSAOBCForce& obc) {
    // customgbforces.py stores or = radius-offset and sr = scalingFactor*or.
    vector<double> params;
    double minOffsetRadius = numeric_limits<double>::max(), maxScaledRadius = 0;
    for (int i = 0; i < force.getNumParticles(); i++) {
        force.getParticleParameters(i, params);
        if (params[1] <= 0)
            return false;
        obc.setParticleParameters(i, params[0], params[1]+OBC2_OFFSET, params[2]/params[1]);
        minOffsetRadius = min(minOffsetRadius, params[1]);
        maxScaledRadius = max(maxScaledRadius, params[2]);
    }
    return maxScaledRadius-minOffsetRadius < MAX_ENGULF_DISTANCE;
}

void MetalCalcCustomGBForceKernel::initialize(const System& system, const CustomGBForce& force) {
    int numCustomGB = 0;
    for (int i = 0; i < system.getNumForces(); i++)
        if (dynamic_cast<const CustomGBForce*>(&system.getForce(i)) != nullptr)
            numCustomGB++;
    // The GBSAOBC kernel names its parameters after the force's index in the System, which a force built here doesn't
    // have, so only one CustomGBForce per System takes this path.
    if (numCustomGB == 1)
        obc = equivalentOBC(force);
    if (obc == nullptr) {
        inner = Kernel(new CommonCalcCustomGBForceKernel(getName(), getPlatform(), cu, system));
        inner.getAs<CommonCalcCustomGBForceKernel>().initialize(system, force);
        return;
    }
    inner = Kernel(new MetalCalcGBSAOBCForceKernel(CalcGBSAOBCForceKernel::Name(), getPlatform(), cu));
    inner.getAs<MetalCalcGBSAOBCForceKernel>().initialize(system, *obc);
}

double MetalCalcCustomGBForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (obc == nullptr)
        return inner.getAs<CommonCalcCustomGBForceKernel>().execute(context, includeForces, includeEnergy);
    return inner.getAs<MetalCalcGBSAOBCForceKernel>().execute(context, includeForces, includeEnergy);
}

void MetalCalcCustomGBForceKernel::copyParametersToContext(ContextImpl& context, const CustomGBForce& force) {
    if (obc == nullptr) {
        inner.getAs<CommonCalcCustomGBForceKernel>().copyParametersToContext(context, force);
        return;
    }
    if (force.getNumParticles() != obc->getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (!copyParticles(force, *obc))
        throw OpenMMException("updateParametersInContext: The new radii leave the OBC2 form this Context was built for "
                "(largest sr minus smallest or must stay under 0.09 nm)");
    inner.getAs<MetalCalcGBSAOBCForceKernel>().copyParametersToContext(context, *obc);
}
