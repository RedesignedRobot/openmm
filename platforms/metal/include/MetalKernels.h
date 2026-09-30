#ifndef OPENMM_METALKERNELS_H_
#define OPENMM_METALKERNELS_H_

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

#include "MetalPlatform.h"
#include "MetalArray.h"
#include "MetalContext.h"
#include "openmm/kernels.h"
#include "openmm/System.h"
#include "openmm/common/CommonKernels.h"
#include "openmm/common/CommonCalcNonbondedForce.h"
#include "openmm/common/CommonCalcConstantPotentialForce.h"
#include "openmm/common/CommonIntegrateCustomStepKernel.h"
#include "openmm/common/ComputeSort.h"
#include "openmm/common/FFT3D.h"

namespace OpenMM {

/**
 * This kernel is invoked at the beginning and end of force and energy computations.  It gives the
 * Platform a chance to clear buffers and do other initialization at the beginning, and to do any
 * necessary work at the end to determine the final results.
 */
class MetalCalcForcesAndEnergyKernel : public CalcForcesAndEnergyKernel {
public:
    MetalCalcForcesAndEnergyKernel(std::string name, const Platform& platform, MetalContext& cu) : CalcForcesAndEnergyKernel(name, platform), cu(cu) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     */
    void initialize(const System& system);
    /**
     * This is called at the beginning of each force/energy computation, before calcForcesAndEnergy() has been called on
     * any ForceImpl.
     *
     * @param context       the context in which to execute this kernel
     * @param includeForce  true if forces should be computed
     * @param includeEnergy true if potential energy should be computed
     * @param groups        a set of bit flags for which force groups to include
     */
    void beginComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups);
    /**
     * This is called at the end of each force/energy computation, after calcForcesAndEnergy() has been called on
     * every ForceImpl.
     *
     * @param context       the context in which to execute this kernel
     * @param includeForce  true if forces should be computed
     * @param includeEnergy true if potential energy should be computed
     * @param groups        a set of bit flags for which force groups to include
     * @param valid         the method may set this to false to indicate the results are invalid and the force/energy
     *                      calculation should be repeated
     * @return the potential energy of the system.  This value is added to all values returned by ForceImpls'
     * calcForcesAndEnergy() methods.  That is, each force kernel may <i>either</i> return its contribution to the
     * energy directly, <i>or</i> add it to an internal buffer so that it will be included here.
     */
    double finishComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups, bool& valid);
private:
    /**
     * Checked mode: compare the forces the pass just computed to the copy taken before it, and throw if they differ.
     */
    void compareCachedForces();
    MetalContext& cu;
    ComputeArray cachedForces;
    bool checkingPass = false;
    int checks = 0;
    double worstCheckRatio = 0;
};

/**
 * This kernel is invoked by NonbondedForce to calculate the forces acting on the system.
 */
class MetalCalcNonbondedForceKernel : public CommonCalcNonbondedForceKernel {
public:
    MetalCalcNonbondedForceKernel(std::string name, const Platform& platform, MetalContext& cu, const System& system) :
            CommonCalcNonbondedForceKernel(name, platform, cu, system), cu(cu) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the NonbondedForce this kernel will be used for
     */
    void initialize(const System& system, const NonbondedForce& force);
    /**
     * Dispatches nothing while the force cache skips the pass; otherwise the common kernel's execute().
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
private:
    MetalContext& cu;
};

/**
 * This kernel is invoked by ConstantPotentialForce to calculate the forces acting on the system.
 */
class MetalCalcConstantPotentialForceKernel : public CommonCalcConstantPotentialForceKernel {
public:
    MetalCalcConstantPotentialForceKernel(std::string name, const Platform& platform, MetalContext& cu, const System& system) :
            CommonCalcConstantPotentialForceKernel(name, platform, cu, system), cu(cu) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the ConstantPotentialForce this kernel will be used for
     */
    void initialize(const System& system, const ConstantPotentialForce& force);
private:
    MetalContext& cu;
};

/**
 * This kernel is invoked by GBSAOBCForce to calculate the forces acting on the system.  With a cutoff, on a device
 * whose launch has few block pairs per SIMD group, it uses the block-pair kernels in kernels/gbsaObcOwned.metal,
 * which accumulate without atomics, in place of the common half-list kernels.
 */
class MetalCalcGBSAOBCForceKernel : public CommonCalcGBSAOBCForceKernel {
public:
    MetalCalcGBSAOBCForceKernel(std::string name, const Platform& platform, MetalContext& cu) :
            CommonCalcGBSAOBCForceKernel(name, platform, cu), cu(cu) {
    }
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    bool createOwnedKernels();
    MetalContext& cu;
    bool hasCheckedOwned = false, useOwned = false;
    ComputeKernel ownedBornSumKernel, ownedForce1Kernel;
};

/**
 * Changing the periodic box moves every atom a reorder has wrapped back to its unwrapped position first, so that its
 * position stays the same.  The common kernel does that on the host with a download and an upload of all positions,
 * and MonteCarloBarostat changes the box on every trial: 0.6 ms of a 2.4 ms trial on a 29,451-atom unit.  This one
 * uploads the cell offsets and shifts the wrapped atoms on the device.
 */
class MetalUpdateStateDataKernel : public CommonUpdateStateDataKernel {
public:
    MetalUpdateStateDataKernel(std::string name, const Platform& platform, MetalContext& cu) :
            CommonUpdateStateDataKernel(name, platform, cu), cu(cu) {
    }
    void initialize(const System& system);
    void setPeriodicBoxVectors(ContextImpl& context, const Vec3& a, const Vec3& b, const Vec3& c);
    /**
     * The fused LangevinMiddle kernel may have left the virtual sites' forces undistributed; distribute them first.
     */
    void getForces(ContextImpl& context, std::vector<Vec3>& forces);
private:
    MetalContext& cu;
    ComputeArray cellOffsets, boxVectors;
    ComputeKernel unwrapKernel;
};

/**
 * The common kernel saves and restores the long force buffer with the positions.  This one also carries the force
 * cache's stamp across a rejected trial: if the buffer held current forces when saveCoordinates() ran, it holds
 * them again after restoreCoordinates(), so the integrator's pass can reuse them (see MetalContext::getForceCache()).
 */
class MetalApplyMonteCarloBarostatKernel : public CommonApplyMonteCarloBarostatKernel {
public:
    MetalApplyMonteCarloBarostatKernel(std::string name, const Platform& platform, MetalContext& cu) :
            CommonApplyMonteCarloBarostatKernel(name, platform, cu), cu(cu) {
    }
    void saveCoordinates(ContextImpl& context);
    void restoreCoordinates(ContextImpl& context);
private:
    MetalContext& cu;
    bool savedForcesCurrent = false;
    int savedGroups = 0;
};

/**
 * This kernel is invoked by LangevinMiddleIntegrator to take one time step.
 */
class MetalIntegrateLangevinMiddleStepKernel : public CommonIntegrateLangevinMiddleStepKernel {
public:
    MetalIntegrateLangevinMiddleStepKernel(std::string name, const Platform& platform, MetalContext& cu) :
            CommonIntegrateLangevinMiddleStepKernel(name, platform, cu), cu(cu) {
    }
    void initialize(const System& system, const LangevinMiddleIntegrator& integrator);
private:
    void integrate(double tol);
    MetalContext& cu;
};

/**
 * This kernel is invoked by CustomIntegrator to take one time step.  In mixed precision it refuses, when the
 * Context is created, per-DOF expressions that use an operation with no double-single implementation.
 */
class MetalIntegrateCustomStepKernel : public CommonIntegrateCustomStepKernel {
public:
    MetalIntegrateCustomStepKernel(std::string name, const Platform& platform, MetalContext& cu) :
            CommonIntegrateCustomStepKernel(name, platform, cu), cu(cu) {
    }
    void initialize(const System& system, const CustomIntegrator& integrator);
private:
    MetalContext& cu;
};

/**
 * This kernel is invoked by CMMotionRemover to remove center of mass motion.  In single and mixed precision with the
 * fused LangevinMiddle kernel it only schedules the work: calcCenterOfMassVelocity runs alongside the force kernels, and
 * the fused kernel subtracts the center of mass velocity as it loads the velocities, which takes both removal kernels
 * off the step's chain of dependent dispatches.  That is safe only when nothing reads or writes the velocities between
 * this force and the integration, so every later force in the System must be one known not to touch them.
 */
class MetalRemoveCMMotionKernel : public RemoveCMMotionKernel {
public:
    MetalRemoveCMMotionKernel(std::string name, const Platform& platform, MetalContext& cu) : RemoveCMMotionKernel(name, platform), cu(cu) {
    }
    /**
     * Initialize the kernel, setting up the particle masses.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CMMotionRemover this kernel will be used for
     */
    void initialize(const System& system, const CMMotionRemover& force);
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     */
    void execute(ContextImpl& context);
private:
    static const int MaxCMBlockSize = 1024;
    MetalContext& cu;
    bool laterForcesIgnoreVelocities;
    int cmBlockSize;
    ComputeArray cmMomentum, cmVelocity;
    ComputeKernel kernel1, kernel2, kernel3;
};

/**
 * This kernel is invoked by CustomCVForce to calculate the forces acting on the system and the energy of the system.
 */
class MetalCalcCustomCVForceKernel : public CommonCalcCustomCVForceKernel {
public:
    MetalCalcCustomCVForceKernel(std::string name, const Platform& platform, ComputeContext& cc) : CommonCalcCustomCVForceKernel(name, platform, cc) {
    }
    ComputeContext& getInnerComputeContext(ContextImpl& innerContext) {
        return *reinterpret_cast<MetalPlatform::PlatformData*>(innerContext.getPlatformData())->contexts[0];
    }
};

/**
 * This kernel is invoked by ATMForce to calculate the forces acting on the system and the energy of the system.
 */
class MetalCalcATMForceKernel : public CommonCalcATMForceKernel {
public:
    MetalCalcATMForceKernel(std::string name, const Platform& platform, ComputeContext& cc) : CommonCalcATMForceKernel(name, platform, cc) {
    }
    ComputeContext& getInnerComputeContext(ContextImpl& innerContext) {
        return *reinterpret_cast<MetalPlatform::PlatformData*>(innerContext.getPlatformData())->contexts[0];
    }
};

} // namespace OpenMM

#endif /*OPENMM_METALKERNELS_H_*/
