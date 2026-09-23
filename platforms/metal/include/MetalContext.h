#ifndef OPENMM_METALCONTEXT_H_
#define OPENMM_METALCONTEXT_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2026 Stanford University and the Authors.      *
 * Portions copyright (c) 2026 the Authors.                                   *
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
#include <map>
#include <string>
#include "openmm/common/windowsExportCommon.h"
#include "MetalArray.h"
#include "MetalIntegrationUtilities.h"
#include "MetalNonbondedUtilities.h"
#include "MetalPlatform.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/BondedUtilities.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ExpressionUtilities.h"
#include "openmm/Kernel.h"

namespace MTL {
    class ComputePipelineState;
    class Device;
    class Library;
}

namespace OpenMM {

class MetalQueue;

/**
 * This class contains the information associated with a Context by the Metal Platform.  Each
 * MetalContext is specific to a particular device, and manages data structures and kernels
 * for that device.
 *
 * Kernels are compiled at runtime from the common kernel sources.  Before compiling, each
 * KERNEL signature is rewritten so that parameters passed by value are declared as constant
 * references, which is how Metal passes arguments that are not buffers.
 */

class OPENMM_EXPORT_COMMON MetalContext : public ComputeContext {
public:
    static const int ThreadBlockSize;
    static const int TileSize;
    MetalContext(const System& system, const std::string& precision, const std::string& tempDir, MetalPlatform::PlatformData& platformData, MetalContext* originalContext);
    ~MetalContext();
    /**
     * This is called to initialize internal data structures after all Forces in the system
     * have been initialized.
     */
    void initialize();
    /**
     * Get the Metal device associated with this object.
     */
    MTL::Device* getDevice() {
        return device;
    }
    /**
     * Get the number of GPU cores on the device.
     */
    int getNumCores() const {
        return numCores;
    }
    /**
     * Get the PlatformData object this context is part of.
     */
    MetalPlatform::PlatformData& getPlatformData() {
        return platformData;
    }
    /**
     * Get the number of contexts being used for the current simulation.
     * This is relevant when a simulation is parallelized across multiple devices.  In that case,
     * one MetalContext is created for each device.
     */
    int getNumContexts() const {
        return platformData.contexts.size();
    }
    /**
     * Get the index of this context in the list stored in the PlatformData.
     */
    int getContextIndex() const {
        return contextIndex;
    }
    /**
     * Get a list of all contexts being used for the current simulation.
     * This is relevant when a simulation is parallelized across multiple devices.  In that case,
     * one ComputeContext is created for each device.
     */
    std::vector<ComputeContext*> getAllContexts();
    /**
     * Get the ContextImpl this context is associated with.
     */
    ContextImpl* getContextImpl() {
        return platformData.context;
    }
    /**
     * Get a workspace used for accumulating energy when a simulation is parallelized across
     * multiple devices.
     */
    double& getEnergyWorkspace();
    /**
     * Construct a new ComputeQueue.
     */
    ComputeQueue createQueue();
    /**
     * Get the MetalQueue that is currently being used for kernel execution.
     */
    MetalQueue& getCurrentMetalQueue();
    /**
     * Construct an uninitialized array of the appropriate class for this platform.  The returned
     * value should be created on the heap with the "new" operator.
     */
    MetalArray* createArray();
    /**
     * Construct a ComputeEvent object of the appropriate class for this platform.
     */
    ComputeEvent createEvent();
    /**
     * Construct a ComputeSort object of the appropriate class for this platform.
     *
     * @param trait      a SortTrait defining the type of data to sort.  It should have been allocated
     *                   on the heap with the "new" operator.  This object takes over ownership of it,
     *                   and deletes it when the ComputeSort is deleted.
     * @param length     the length of the arrays this object will be used to sort
     * @param uniform    whether the input data is expected to follow a uniform or nonuniform
     *                   distribution.  This argument is used only as a hint.  It allows parts
     *                   of the algorithm to be tuned for faster performance on the expected
     *                   distribution.
     */
    ComputeSort createSort(ComputeSortImpl::SortTrait* trait, unsigned int length, bool uniform=true);
    /**
     * Create an FFT3D object for performing FFTs.
     *
     * @param xsize           the first dimension of the data sets on which FFTs will be performed
     * @param ysize           the second dimension of the data sets on which FFTs will be performed
     * @param zsize           the third dimension of the data sets on which FFTs will be performed
     * @param realToComplex   if true, a real-to-complex transform will be done.  Otherwise, it is complex-to-complex.
     */
    FFT3D createFFT(int xsize, int ysize, int zsize, bool realToComplex=false);
    /**
     * Get the smallest legal size for a dimension of the grid.
     */
    int findLegalFFTDimension(int minimum);
    /**
     * Compile source code to create a ComputeProgram.
     *
     * @param source             the source code of the program
     * @param defines            a set of preprocessor definitions (name, value) to define when compiling the program
     */
    ComputeProgram compileProgram(const std::string source, const std::map<std::string, std::string>& defines=std::map<std::string, std::string>());
    /**
     * Convert a number to a string in a format suitable for including in a kernel.  Metal has no double
     * precision literals, so when a double precision constant is requested in mixed precision mode and a
     * float cannot hold the value exactly, this writes it as a df64: the sum of two floats.
     */
    std::string doubleToString(double value, bool mixedIsDouble=false) const;
    /**
     * Get the definitions of SQRT, RSQRT, RECIP, EXP, and LOG to compile programs with.  SQRT is always
     * correctly rounded.  Each of the others uses the fast version if its flag is set, and the precise
     * version otherwise.
     */
    static std::map<std::string, std::string> getMathFunctionDefines(bool fastRsqrt, bool fastRecip, bool fastExp, bool fastLog);
    /**
     * Convert an array to a MetalArray.  If the argument is already a MetalArray, this simply casts it.
     * If the argument is a ComputeArray that wraps a MetalArray, this returns the wrapped array.  For any
     * other argument, this throws an exception.
     */
    MetalArray& unwrap(ArrayInterface& array) const;
    /**
     * Get the array which contains a contribution to each force represented as a real4.
     */
    ArrayInterface& getFloatForceBuffer() {
        throw OpenMMException("Metal platform does not use floating point force buffers");
    }
    /**
     * On devices that do not support 64 bit atomics, this returns an array containing buffers of type real4 in which
     * forces can be accumulated.
     */
    ArrayInterface& getForceBuffers() {
        throw OpenMMException("Metal platform does not use floating point force buffers");
    }
    /**
     * Get the array which contains the buffer in which energy is computed.
     */
    MetalArray& getEnergyBuffer() {
        return energyBuffer;
    }
    /**
     * Get a pointer to a block of host memory that can be used for transfers between host and device.
     * This is guaranteed to be at least as large as any of the arrays returned by methods of this class.
     */
    void* getPinnedBuffer() {
        return pinnedBuffer.data();
    }
    /**
     * Get a shared ThreadPool that code can use to parallelize operations.
     */
    ThreadPool& getThreadPool() {
        return getPlatformData().threads;
    }
    /**
     * Compile Metal source code into a library.
     *
     * @param source             the source code of the library
     * @param defines            a set of preprocessor definitions (name, value) to define when compiling the program
     * @return the compiled library.  The caller takes ownership of it.
     */
    MTL::Library* createLibrary(const std::string& source, const std::map<std::string, std::string>& defines=std::map<std::string, std::string>());
    /**
     * Create a compute pipeline for a kernel in a library.
     *
     * @param library    the library containing the kernel
     * @param name       the name of the kernel function
     * @return the pipeline.  The caller takes ownership of it.
     */
    MTL::ComputePipelineState* createPipeline(MTL::Library* library, const std::string& name);
    /**
     * Compute the largest thread block size that can be used for a kernel that requires a particular amount of
     * threadgroup memory per thread.
     *
     * @param memory        the number of bytes of threadgroup memory per thread
     */
    int computeThreadBlockSize(double memory) const;
    /**
     * Sum the buffer containing energy.
     */
    double reduceEnergy();
    /**
     * Get the maximum number of threads in a thread block supported by this device.
     */
    int getMaxThreadBlockSize() const {
        return 256;
    }
    /**
     * Get whether the device being used is a CPU.  In some cases, different algorithms
     * may be more efficient on CPUs and GPUs.
     */
    bool getIsCPU() const {
        return false;
    }
    /**
     * Get the SIMD width of the device being used.
     */
    int getSIMDWidth() const {
        return 32;
    }
    /**
     * Get whether the device being used supports 64 bit atomic operations on global memory.
     * Apple GPUs have no 64 bit atomic addition.  The prelude emulates it for the integers that
     * fixed point forces use, but not for doubles, which the minimizer needs in mixed precision.
     */
    bool getSupports64BitGlobalAtomics() const {
        return false;
    }
    /**
     * Get whether the device being used supports double precision math.  Apple GPUs do not, but in
     * mixed precision mode kernels have df64, which stands in for double.
     */
    bool getSupportsDoublePrecision() const {
        return useMixedPrecision;
    }
    /**
     * Get the vectors defining the periodic box.
     */
    void getPeriodicBoxVectors(Vec3& a, Vec3& b, Vec3& c) const {
        a = periodicBoxVectors[0];
        b = periodicBoxVectors[1];
        c = periodicBoxVectors[2];
    }
    /**
     * Set the vectors defining the periodic box.
     */
    void setPeriodicBoxVectors(const Vec3& a, const Vec3& b, const Vec3& c) {
        periodicBoxVectors[0] = a;
        periodicBoxVectors[1] = b;
        periodicBoxVectors[2] = c;
    }
    /**
     * Get the MetalIntegrationUtilities for this context.
     */
    MetalIntegrationUtilities& getIntegrationUtilities() {
        return *integration;
    }
    /**
     * Get the ExpressionUtilities for this context.
     */
    ExpressionUtilities& getExpressionUtilities() {
        return *expression;
    }
    /**
     * Get the BondedUtilities for this context.
     */
    BondedUtilities& getBondedUtilities() {
        return *bonded;
    }
    /**
     * Get the MetalNonbondedUtilities for this context.
     */
    MetalNonbondedUtilities& getNonbondedUtilities() {
        return *nonbonded;
    }
    /**
     * Create a new NonbondedUtilities for use with this context.  This should be called
     * only in unusual situations, when a Force needs its own NonbondedUtilities object
     * separate from the standard one.  The caller is responsible for deleting the object
     * when it is no longer needed.
     */
    MetalNonbondedUtilities* createNonbondedUtilities() {
        return new MetalNonbondedUtilities(*this);
    }
    /**
     * This should be called by the Integrator from its own initialize() method.
     * It ensures all contexts are fully initialized.
     */
    void initializeContexts();
    /**
     * Wait until all work that has been queued (kernel executions, asynchronous data transfers, etc.)
     * has been submitted to the device.  This does not mean it has necessarily been completed.
     * Calling this periodically may improve the responsiveness of the computer's GUI, but at the
     * expense of reduced simulation performance.
     */
    void flushQueue();
private:
    MetalPlatform::PlatformData& platformData;
    int contextIndex;
    int numCores;
    std::string tempDir;
    Vec3 periodicBoxVectors[3];
    std::map<std::string, std::string> compilationDefines;
    MTL::Device* device;
    std::vector<double> pinnedBuffer;
    MetalArray energyBuffer;
    MetalIntegrationUtilities* integration;
    ExpressionUtilities* expression;
    BondedUtilities* bonded;
    MetalNonbondedUtilities* nonbonded;
};

} // namespace OpenMM

#endif /*OPENMM_METALCONTEXT_H_*/
