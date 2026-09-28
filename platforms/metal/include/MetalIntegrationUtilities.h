#ifndef OPENMM_METALINTEGRATIONUTILITIES_H_
#define OPENMM_METALINTEGRATIONUTILITIES_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2019 Stanford University and the Authors.      *
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

#include "MetalArray.h"
#include "openmm/System.h"
#include "openmm/common/IntegrationUtilities.h"
#include "openmm/common/windowsExportCommon.h"
#include "openmm/common/ComputeEvent.h"

namespace OpenMM {

class MetalContext;

/**
 * This class implements features that are used by many different integrators, including
 * common workspace arrays, random number generation, and enforcing constraints.
 */

class OPENMM_EXPORT_COMMON MetalIntegrationUtilities : public IntegrationUtilities {
public:
    MetalIntegrationUtilities(MetalContext& context, const System& system);
    ~MetalIntegrationUtilities();
    /**
     * Get the array which contains position deltas.
     */
    MetalArray& getPosDelta();
    /**
     * Get the array which contains random values.  Each element is a float4, whose components
     * are independent, normally distributed random numbers with mean 0 and variance 1.
     */
    MetalArray& getRandom();
    /**
     * Get the array which contains the current step size.
     */
    MetalArray& getStepSize();
    /**
     * Distribute forces from virtual sites to the atoms they are based on.
     */
    void distributeForcesFromVirtualSites();
    /**
     * Update the positions and velocities for one LangevinMiddleIntegrator step and enforce the constraints, all in one
     * kernel.  This does nothing and returns false if CCMA enforces any of the constraints, or in double precision.
     *
     * @param params   the velocity scale and noise scale of the integrator
     * @param tol      the constraint tolerance
     */
    bool integrateLangevinMiddle(ComputeArray& params, double tol);
    /**
     * Get whether integrateLangevinMiddle() can remove center of mass motion for a CMMotionRemover.  That holds in
     * single and mixed precision when the fused kernel runs.
     */
    bool canRemoveCenterOfMassMomentum() const;
    /**
     * Remove center of mass motion as part of the next step.  calcKernel runs in the force computation's concurrent
     * pass (see calcPendingCenterOfMassMomentum()) and writes the center of mass velocity to cmVelocity[0].  The next
     * integrateLangevinMiddle() subtracts it from every atom as it loads it, in place of the
     * removeCenterOfMassMomentum kernel.
     *
     * @param calcKernel   the calcCenterOfMassVelocity kernel, with its arguments set
     * @param blockSize    the size of calcKernel's single thread group
     * @param cmVelocity   the array calcKernel writes to
     */
    void removeCenterOfMassMomentum(ComputeKernel& calcKernel, int blockSize, ComputeArray& cmVelocity);
    /**
     * Run the calcCenterOfMassVelocity kernel that removeCenterOfMassMomentum() left pending, if there is one.  It
     * reads the velocities and writes only its own output, so it can share a concurrent pass with the force kernels.
     */
    void calcPendingCenterOfMassMomentum();
private:
    void applyConstraintsImpl(bool constrainVelocities, double tol);
    int* ccmaConvergedMemory;
    metalDeviceptr_t ccmaConvergedDeviceMemory;
    ComputeEvent ccmaEvent;
    ComputeKernel langevinMiddleKernel;
    ComputeArray freeAtoms;
    int numLangevinMiddleThreads;
    bool flatLangevinMiddle; // Probe (r2-mixed, not for merge).
    ComputeArray probeIters; // Probe (r2-mixed, not for merge).
    std::string probeItersTag;
    int probeCalls;
    ComputeKernel pendingCMCalc;
    int pendingCMBlockSize;
    ComputeArray* pendingCMVelocity;
};

} // namespace OpenMM

#endif /*OPENMM_METALINTEGRATIONUTILITIES_H_*/
