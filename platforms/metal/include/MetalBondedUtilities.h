#ifndef OPENMM_METALBONDEDUTILITIES_H_
#define OPENMM_METALBONDEDUTILITIES_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2011-2024 Stanford University and the Authors.      *
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

#include "openmm/common/BondedUtilities.h"
#include "openmm/common/windowsExportCommon.h"

namespace OpenMM {

/**
 * This class gives each threadgroup a block of contiguous atoms to own.  The threadgroup evaluates every
 * term of every force that touches one of its atoms, sums the forces on its own atoms in threadgroup
 * memory as 64-bit fixed point, and adds each atom's total to the force buffer once.  A term that touches
 * several blocks is evaluated once in each, and only its home block (the block of its first atom) counts
 * its energy.  So each atom takes one atomic add per component, instead of one for every term that
 * touches it.
 */

class OPENMM_EXPORT_COMMON MetalBondedUtilities : public BondedUtilities {
public:
    MetalBondedUtilities(ComputeContext& context) : BondedUtilities(context) {
        useEnergyVariants = true;
    }
    /**
     * Initialize this object in preparation for a simulation.
     */
    void initialize(const System& system);
protected:
    std::string createForceSource(int forceIndex, int numBonds, int numAtoms, int group, const std::string& computeForce);
private:
    std::string createBlockSource(int forceIndex, int numAtoms, int group, const std::string& computeForce);
    ComputeArray plan;
    std::string planArgument;
    std::vector<int> blockStartOffset, termOffset;
    std::vector<std::string> blockSource;
    int blockSize, numBlocks, lastBlockForce;
};

} // namespace OpenMM

#endif /*OPENMM_METALBONDEDUTILITIES_H_*/
