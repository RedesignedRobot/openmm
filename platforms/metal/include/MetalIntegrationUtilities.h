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
private:
    void applyConstraintsImpl(bool constrainVelocities, double tol);
    int* ccmaConvergedMemory;
    metalDeviceptr_t ccmaConvergedDeviceMemory;
    ComputeEvent ccmaEvent;
    ComputeKernel langevinMiddleKernel;
    ComputeArray freeAtoms;
    int numLangevinMiddleThreads;
};

} // namespace OpenMM

#endif /*OPENMM_METALINTEGRATIONUTILITIES_H_*/
