#ifndef OPENMM_METALPROGRAM_H_
#define OPENMM_METALPROGRAM_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2019 Stanford University and the Authors.           *
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

#include "openmm/common/ComputeProgram.h"
#include "MetalContext.h"

namespace OpenMM {

/**
 * This is the METAL implementation of the ComputeProgramImpl interface.
 */

class MetalProgram : public ComputeProgramImpl {
public:
    /**
     * Create a new MetalProgram.
     *
     * @param context      the context this kernel belongs to
     * @param module       the compiled module
     */
    MetalProgram(MetalContext& context, metalModule_t module);
    /**
     * Create a ComputeKernel for one of the kernels in this program.
     *
     * @param name    the name of the kernel to get
     */
    ComputeKernel createKernel(const std::string& name);
private:
    MetalContext& context;
    metalModule_t module;
};

} // namespace OpenMM

#endif /*OPENMM_METALPROGRAM_H_*/
