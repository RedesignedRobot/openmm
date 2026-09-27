/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.      *
 * Portions copyright (c) 2020-2022 Advanced Micro Devices, Inc.              *
 * Authors: Peter Eastman, Nicholas Curtis                                    *
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

#include "MetalKernels.h"
#include "MetalForceInfo.h"
#include "openmm/Context.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/common/ContextSelector.h"
#include "CommonKernelSources.h"
#include "MetalBondedUtilities.h"
#include "MetalExpressionUtilities.h"
#include "MetalIntegrationUtilities.h"
#include "MetalNonbondedUtilities.h"
#include "MetalKernelSources.h"
#include "MetalQueue.h"
#include "ReferenceTabulatedFunction.h"
#include "SimTKOpenMMRealType.h"
#include "SimTKOpenMMUtilities.h"
#include "lepton/CustomFunction.h"
#include "lepton/Operation.h"
#include "lepton/Parser.h"
#include "lepton/ParsedExpression.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>
#include <set>
#include <assert.h>

using namespace OpenMM;
using namespace std;

static void getMetalPmeParameters(MetalContext& cu, bool& usePmeQueue, bool& useFixedPointChargeSpreading) {
    usePmeQueue = (!cu.getPlatformData().disablePmeStream && !cu.getPlatformData().useCpuPme);
    useFixedPointChargeSpreading = cu.getUseDoublePrecision() || !cu.getSupportsHardwareFloatGlobalAtomicAdd() || cu.getPlatformData().deterministicForces;
}

// Sets the queue's dispatch type for a scope and restores serial dispatch on the way out, even when an
// exception leaves the scope.
struct ConcurrentDispatchScope {
    ConcurrentDispatchScope(MetalQueue& queue, bool concurrent) : queue(queue) {
        queue.setConcurrentDispatch(concurrent);
    }
    ~ConcurrentDispatchScope() {
        queue.setConcurrentDispatch(false);
    }
    MetalQueue& queue;
};

void MetalCalcForcesAndEnergyKernel::initialize(const System& system) {
}

void MetalCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    cu.setForcesValid(true);
    ContextSelector selector(cu);
    cu.clearAutoclearBuffers();
    cu.updateGlobalParamValues();
    for (auto computation : cu.getPreComputations())
        computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    MetalNonbondedUtilities& nb = cu.getNonbondedUtilities();
    cu.setComputeForceCount(cu.getComputeForceCount()+1);
    nb.prepareInteractions(groups);
    map<string, double>& derivs = cu.getEnergyParamDerivWorkspace();
    for (auto& param : context.getParameters())
        derivs[param.first] = 0;
}

double MetalCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups, bool& valid) {
    ContextSelector selector(cu);

    // The bonded and nonbonded kernels share no written buffer except the force buffer, and both only add
    // to it with integer atomics, so they can run side by side and still give the same bits.  Each also does
    // a plain += into the energy buffer (bonded on every step, nonbonded only with energy), and each writes
    // energy parameter derivatives when there are any, so those steps keep them in order.  The pass
    // boundaries on either side order them against the clear before and the readers after.

    {
        bool overlap = (includeForces && !includeEnergy && cu.getEnergyParamDerivNames().empty());
        ConcurrentDispatchScope scope(cu.getCurrentStream(), overlap);
        cu.getBondedUtilities().computeInteractions(groups);
        cu.getNonbondedUtilities().computeInteractions(groups, includeForces, includeEnergy);
    }
    double sum = 0.0;
    for (auto computation : cu.getPostComputations())
        sum += computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    cu.getIntegrationUtilities().distributeForcesFromVirtualSites();
    if (includeEnergy)
        sum += cu.reduceEnergy();
    if (!cu.getForcesValid())
        valid = false;
    return sum;
}

void MetalCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
    bool usePmeQueue, useFixedPointChargeSpreading;
    getMetalPmeParameters(cu, usePmeQueue, useFixedPointChargeSpreading);
    // Sorting atoms by PME grid index costs more than it saves: reorderAtoms already keeps the stored order spatially coherent.
    commonInitialize(system, force, usePmeQueue, false, useFixedPointChargeSpreading, cu.getPlatformData().useCpuPme, false);
}

void MetalCalcConstantPotentialForceKernel::initialize(const System& system, const ConstantPotentialForce& force) {
    bool usePmeQueue, useFixedPointChargeSpreading;
    getMetalPmeParameters(cu, usePmeQueue, useFixedPointChargeSpreading);
    commonInitialize(system, force, false, useFixedPointChargeSpreading);
}

/**
 * Find an operation in an expression that Metal mixed precision cannot evaluate per DOF, where every value is a
 * double-single pair.  Integer powers are multiplied out, so only other exponents need pow().  Returns an empty
 * string if there is none.
 */
static string findUnsupportedMixedOperation(const Lepton::ExpressionTreeNode& node) {
    const Lepton::Operation& op = node.getOperation();
    switch (op.getId()) {
        case Lepton::Operation::TAN:
        case Lepton::Operation::ASIN:
        case Lepton::Operation::ACOS:
        case Lepton::Operation::ATAN:
        case Lepton::Operation::ATAN2:
        case Lepton::Operation::SINH:
        case Lepton::Operation::COSH:
        case Lepton::Operation::TANH:
        case Lepton::Operation::ERF:
        case Lepton::Operation::ERFC:
            return op.getName();
        case Lepton::Operation::POWER:
            return "^ with a variable exponent";
        case Lepton::Operation::POWER_CONSTANT: {
            double exponent = dynamic_cast<const Lepton::Operation::PowerConstant&>(op).getValue();
            if (exponent != (int) exponent) {
                stringstream name;
                name << "^" << exponent;
                return name.str();
            }
            break;
        }
        default:
            break;
    }
    for (auto& child : node.getChildren()) {
        string found = findUnsupportedMixedOperation(child);
        if (!found.empty())
            return found;
    }
    return "";
}

static void checkMixedPrecisionExpression(const string& expression, const string& where, const map<string, Lepton::CustomFunction*>& functions) {
    string operation = findUnsupportedMixedOperation(Lepton::Parser::parse(expression, functions).optimize().getRootNode());
    if (!operation.empty())
        throw OpenMMException("Metal mixed precision cannot evaluate "+operation+" in the CustomIntegrator "+where+" '"+expression+
            "'.  In mixed precision, per-DOF expressions cannot use tan, asin, acos, atan, atan2, sinh, cosh, tanh, erf, erfc or a non-integer power."
            "  Use Precision=single, or compute the value in a global expression.");
}

void MetalIntegrateCustomStepKernel::initialize(const System& system, const CustomIntegrator& integrator) {
    if (cu.getUseMixedPrecision()) {
        // Parse per-DOF expressions the way CustomIntegratorUtilities::analyzeComputations does.

        vector<unique_ptr<Lepton::CustomFunction> > tables;
        map<string, Lepton::CustomFunction*> functions;
        for (int i = 0; i < integrator.getNumTabulatedFunctions(); i++) {
            tables.emplace_back(createReferenceTabulatedFunction(integrator.getTabulatedFunction(i)));
            functions[integrator.getTabulatedFunctionName(i)] = tables.back().get();
        }
        Lepton::PlaceholderFunction fn1(1), fn2(2), fn3(3);
        for (string name : {"_x", "_y", "_z"})
            functions[name] = &fn1;
        for (string name : {"deriv", "dot", "cross"})
            functions[name] = &fn2;
        functions["vector"] = &fn3;
        for (int step = 0; step < integrator.getNumComputations(); step++) {
            CustomIntegrator::ComputationType type;
            string variable, expression;
            integrator.getComputationStep(step, type, variable, expression);
            if (type == CustomIntegrator::ComputePerDof || type == CustomIntegrator::ComputeSum)
                checkMixedPrecisionExpression(expression, "step "+cu.intToString(step)+" expression", functions);
        }
        checkMixedPrecisionExpression(integrator.getKineticEnergyExpression(), "kinetic energy expression", functions);
    }
    CommonIntegrateCustomStepKernel::initialize(system, integrator);
}

void MetalIntegrateLangevinMiddleStepKernel::integrate(double tol) {
    if (!cu.getIntegrationUtilities().integrateLangevinMiddle(params, tol))
        CommonIntegrateLangevinMiddleStepKernel::integrate(tol);
}
