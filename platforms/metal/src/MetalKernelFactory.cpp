/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2026 Stanford University and the Authors.      *
 * Portions copyright (c) 2020 Advanced Micro Devices, Inc.                   *
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

#include "MetalKernelFactory.h"
#include "MetalKernels.h"
#include "MetalPlatform.h"
#include "openmm/common/CommonKernels.h"
#include "openmm/common/CommonCalcCustomGBForceKernel.h"
#include "openmm/common/CommonCalcCustomHbondForceKernel.h"
#include "openmm/common/CommonCalcCustomManyParticleForceKernel.h"
#include "openmm/common/CommonCalcCustomNonbondedForceKernel.h"
#include "openmm/common/CommonIntegrateCustomStepKernel.h"
#include "openmm/common/CommonIntegrateNoseHooverStepKernel.h"
#include "openmm/common/CommonIntegrateQTBStepKernel.h"
#include "openmm/common/CommonMinimizeKernel.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;

KernelImpl* MetalKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    MetalPlatform::PlatformData& data = *static_cast<MetalPlatform::PlatformData*>(context.getPlatformData());
    MetalContext& cu = *data.contexts[0];
    if (name == CalcForcesAndEnergyKernel::Name())
        return new MetalCalcForcesAndEnergyKernel(name, platform, cu);
    if (name == UpdateStateDataKernel::Name())
        return new MetalUpdateStateDataKernel(name, platform, cu);
    if (name == ApplyConstraintsKernel::Name())
        return new CommonApplyConstraintsKernel(name, platform, cu);
    if (name == VirtualSitesKernel::Name())
        return new CommonVirtualSitesKernel(name, platform, cu);
    if (name == MinimizeKernel::Name())
        return new CommonMinimizeKernel(name, platform, cu);
    if (name == CalcHarmonicBondForceKernel::Name())
        return new CommonCalcHarmonicBondForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomBondForceKernel::Name())
        return new CommonCalcCustomBondForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcHarmonicAngleForceKernel::Name())
        return new CommonCalcHarmonicAngleForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomAngleForceKernel::Name())
        return new CommonCalcCustomAngleForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcPeriodicTorsionForceKernel::Name())
        return new CommonCalcPeriodicTorsionForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcRBTorsionForceKernel::Name())
        return new CommonCalcRBTorsionForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCMAPTorsionForceKernel::Name())
        return new CommonCalcCMAPTorsionForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomTorsionForceKernel::Name())
        return new CommonCalcCustomTorsionForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcNonbondedForceKernel::Name())
        return new MetalCalcNonbondedForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcConstantPotentialForceKernel::Name())
        return new MetalCalcConstantPotentialForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomNonbondedForceKernel::Name())
        return new CommonCalcCustomNonbondedForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcGBSAOBCForceKernel::Name())
        return new MetalCalcGBSAOBCForceKernel(name, platform, cu);
    if (name == CalcCustomGBForceKernel::Name())
        return new CommonCalcCustomGBForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomExternalForceKernel::Name())
        return new CommonCalcCustomExternalForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomHbondForceKernel::Name())
        return new CommonCalcCustomHbondForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomCentroidBondForceKernel::Name())
        return new CommonCalcCustomCentroidBondForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomCompoundBondForceKernel::Name())
        return new CommonCalcCustomCompoundBondForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomCVForceKernel::Name())
        return new MetalCalcCustomCVForceKernel(name, platform, cu);
    if (name == CalcCustomCPPForceKernel::Name())
        return new CommonCalcCustomCPPForceKernel(name, platform, context, cu);
    if (name == CalcOrientationRestraintForceKernel::Name())
        return new CommonCalcOrientationRestraintForceKernel(name, platform, cu);
    if (name == CalcPythonForceKernel::Name())
        return new CommonCalcPythonForceKernel(name, platform, context, cu);
    if (name == CalcRGForceKernel::Name())
        return new CommonCalcRGForceKernel(name, platform, cu);
    if (name == CalcRMSDForceKernel::Name())
        return new CommonCalcRMSDForceKernel(name, platform, cu);
    if (name == CalcCustomManyParticleForceKernel::Name())
        return new CommonCalcCustomManyParticleForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcGayBerneForceKernel::Name())
        return new CommonCalcGayBerneForceKernel(name, platform, cu);
    if (name == CalcLCPOForceKernel::Name())
        return new CommonCalcLCPOForceKernel(name, platform, cu);
    if (name == IntegrateVerletStepKernel::Name())
        return new CommonIntegrateVerletStepKernel(name, platform, cu);
    if (name == IntegrateLangevinMiddleStepKernel::Name())
        return new MetalIntegrateLangevinMiddleStepKernel(name, platform, cu);
    if (name == IntegrateBrownianStepKernel::Name())
        return new CommonIntegrateBrownianStepKernel(name, platform, cu);
    if (name == IntegrateVariableVerletStepKernel::Name())
        return new CommonIntegrateVariableVerletStepKernel(name, platform, cu);
    if (name == IntegrateVariableLangevinStepKernel::Name())
        return new CommonIntegrateVariableLangevinStepKernel(name, platform, cu);
    if (name == IntegrateCustomStepKernel::Name())
        return new MetalIntegrateCustomStepKernel(name, platform, cu);
    if (name == IntegrateDPDStepKernel::Name())
        return new CommonIntegrateDPDStepKernel(name, platform, cu);
    if (name == IntegrateQTBStepKernel::Name())
        return new CommonIntegrateQTBStepKernel(name, platform, cu);
    if (name == ApplyAndersenThermostatKernel::Name())
        return new CommonApplyAndersenThermostatKernel(name, platform, cu);
    if (name == IntegrateNoseHooverStepKernel::Name())
        return new CommonIntegrateNoseHooverStepKernel(name, platform, cu);
    if (name == ApplyMonteCarloBarostatKernel::Name())
        return new MetalApplyMonteCarloBarostatKernel(name, platform, cu);
    if (name == RemoveCMMotionKernel::Name())
        return new MetalRemoveCMMotionKernel(name, platform, cu);
    if (name == CalcATMForceKernel::Name() )
        return new MetalCalcATMForceKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
