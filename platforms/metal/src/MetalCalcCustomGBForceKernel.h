#ifndef OPENMM_METALCALCCUSTOMGBFORCEKERNEL_H_
#define OPENMM_METALCALCCUSTOMGBFORCEKERNEL_H_

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

#include "MetalContext.h"
#include "openmm/GBSAOBCForce.h"
#include "openmm/Kernel.h"
#include "openmm/kernels.h"
#include <memory>

namespace OpenMM {

/**
 * This kernel is invoked by CustomGBForce.  openmm.app writes OBC2 implicit solvent (implicit/obc2.xml,
 * customgbforces.GBSAOBC2Force) as a CustomGBForce, which the common kernel runs with generated code: its chain rule
 * rides in the nonbonded kernel and its pair energy in a generic N^2 kernel, about twice the step time of the
 * hand-written GBSAOBCForce kernels on the Folding@home OBC2 unit.  When the force is exactly that template (the
 * expressions openmm.app writes, no salt, dielectric offset 0.009, ACE or no surface term, no globals, tables or
 * parameter derivatives, one CustomGBForce in the System) and its radii keep OBC's engulfing term at zero at every
 * bonded distance, this kernel builds the equivalent GBSAOBCForce and runs the Metal GBSAOBC kernel.  Any other
 * CustomGBForce runs the common kernel unchanged.
 */
class MetalCalcCustomGBForceKernel : public CalcCustomGBForceKernel {
public:
    MetalCalcCustomGBForceKernel(std::string name, const Platform& platform, MetalContext& cu, const System& system);
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomGBForce this kernel will be used for
     */
    void initialize(const System& system, const CustomGBForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CustomGBForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CustomGBForce& force);
private:
    static std::unique_ptr<GBSAOBCForce> equivalentOBC(const CustomGBForce& force);
    static bool copyParticles(const CustomGBForce& force, GBSAOBCForce& obc);
    MetalContext& cu;
    // The GBSAOBC kernel's ForceInfo keeps a reference to this force, so it lives as long as the kernel.
    std::unique_ptr<GBSAOBCForce> obc;
    Kernel inner;
};

} // namespace OpenMM

#endif /*OPENMM_METALCALCCUSTOMGBFORCEKERNEL_H_*/
