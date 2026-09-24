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
 * This class evaluates each force's terms in chunks of one threadgroup, and sums the chunk's forces on
 * each atom in threadgroup memory before adding them to the force buffer.  That takes one atomic add per
 * atom and component for each chunk, instead of one for every term that touches the atom.
 */

class OPENMM_EXPORT_COMMON MetalBondedUtilities : public BondedUtilities {
public:
    MetalBondedUtilities(ComputeContext& context) : BondedUtilities(context) {
    }
    /**
     * Initialize this object in preparation for a simulation.
     */
    void initialize(const System& system);
protected:
    std::string createForceSource(int forceIndex, int numBonds, int numAtoms, int group, const std::string& computeForce);
private:
    ComputeArray plan;
    std::string planArgument;
    std::vector<int> entryStartOffset, entryOffset, slotOffset;
    int firstChunkedForce, maxChunkedAtoms;
};

} // namespace OpenMM

#endif /*OPENMM_METALBONDEDUTILITIES_H_*/
