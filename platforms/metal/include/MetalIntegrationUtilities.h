#ifndef OPENMM_METALINTEGRATIONUTILITIES_H_
#define OPENMM_METALINTEGRATIONUTILITIES_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2026 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
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

namespace OpenMM {

class MetalContext;

/**
 * This class implements features that are used by many different integrators, including
 * common workspace arrays, random number generation, and enforcing constraints.
 */

class MetalIntegrationUtilities : public IntegrationUtilities {
public:
    MetalIntegrationUtilities(MetalContext& context, const System& system);
    ~MetalIntegrationUtilities();
    /**
     * Distribute forces from virtual sites to the atoms they are based on.
     */
    void distributeForcesFromVirtualSites();
    /**
     * Update the positions and velocities for one LangevinMiddleIntegrator step and enforce the constraints, all in one
     * kernel.  This does nothing and returns false unless the context uses mixed precision and no constraint goes
     * through CCMA.
     *
     * @param params   the velocity scale and noise scale of the integrator
     * @param tol      the constraint tolerance
     */
    bool integrateLangevinMiddle(ComputeArray& params, double tol);
    /**
     * Get whether integrateLangevinMiddle() can remove center of mass motion for a CMMotionRemover.  That holds when
     * the fused kernel runs: in mixed precision with no constraint going through CCMA.
     */
    bool canRemoveCenterOfMassMomentum() const;
    /**
     * Remove center of mass motion as part of the next step.  calcCenterOfMassMomentum has already written its partial
     * momenta, and the next integrateLangevinMiddle() sums them the way removeCenterOfMassMomentum does and subtracts
     * the center of mass velocity from every atom as it loads it, in place of that kernel.
     *
     * @param cmMomentum   the array calcCenterOfMassMomentum wrote one partial momentum per thread group to
     * @param numGroups    the number of partial momenta, the thread groups calcCenterOfMassMomentum ran with
     */
    void removeCenterOfMassMomentum(ComputeArray& cmMomentum, int numGroups);
private:
    void applyConstraintsImpl(bool constrainVelocities, double tol);
    MetalArray ccmaConvergedBuffer;
    ComputeKernel langevinMiddleKernel;
    ComputeArray freeAtoms;
    int numLangevinMiddleThreads;
    double totalMass;
    ComputeArray* pendingCMMomentum;
    int pendingCMGroups;
};

} // namespace OpenMM

#endif /*OPENMM_METALINTEGRATIONUTILITIES_H_*/
