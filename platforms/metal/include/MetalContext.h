#ifndef OPENMM_METALCONTEXT_H_
#define OPENMM_METALCONTEXT_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2026 Stanford University and the Authors.      *
 * Portions copyright (c) 2020-2023 Advanced Micro Devices, Inc.              *
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


#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#define __CL_ENABLE_EXCEPTIONS
#ifdef _MSC_VER
    // Prevent Windows from defining macros that interfere with other code.
    #define NOMINMAX
#endif
#include "MetalQueue.h"
#include "openmm/common/windowsExportCommon.h"
#include "MetalArray.h"
#include "MetalBondedUtilities.h"
#include "MetalExpressionUtilities.h"
#include "MetalIntegrationUtilities.h"
#include "MetalNonbondedUtilities.h"
#include "MetalPlatform.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/Kernel.h"

typedef unsigned int tileflags;

namespace OpenMM {

typedef mm_int2 int2;
typedef mm_float4 float4;
typedef mm_double4 double4;
typedef unsigned int uint1;
inline int2 make_int2(int x, int y) { return int2(x, y); }
inline float4 make_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }
inline double4 make_double4(double x, double y, double z, double w) { return double4(x, y, z, w); }

/**
 * This class contains the information associated with a Context by the METAL Platform.  Each MetalContext is
 * specific to a particular device, and manages data structures and kernels for that device.  When running a simulation
 * in parallel on multiple devices, there is a separate MetalContext for each one.  The list of all contexts is
 * stored in the MetalPlatform::PlatformData.
 * <p>
 * In addition, a worker thread is created for each MetalContext.  This is used for parallel computations, so that
 * blocking calls to one device will not block other devices.  When only a single device is being used, the worker
 * thread is not used and calculations are performed on the main application thread.
 */

class OPENMM_EXPORT_COMMON MetalContext : public ComputeContext {
public:
    class WorkTask;
    class WorkThread;
    class ReorderListener;
    class ForcePreComputation;
    class ForcePostComputation;
    static const int ThreadBlockSize;
    static const int TileSize;
    static const int SpreadBlockSize;
    MetalContext(const System& system, int deviceIndex, bool useBlockingSync, const std::string& precision,
            const std::string& tempDir, MetalPlatform::PlatformData& platformData, MetalContext* originalContext);
    ~MetalContext();
    /**
     * This is called to initialize internal data structures after all Forces in the system
     * have been initialized.
     */
    void initialize();
    /**
     * Get whether the context associated with this object is valid.
     */
    bool getContextIsValid() const {
        return contextIsValid;
    }
    /**
     * Set the device associated with this object to be the current device.  If the context is not
     * valid, this returns without doing anything.
     */
    void setAsCurrent();
    /**
     * Push the device associated with this object to be the current device.  If the context is not
     * valid, this returns without doing anything.
     */
    void pushAsCurrent();
    /**
     * Pop the device associated with this object off the stack of contexts.  If the context is not
     * valid, this returns without doing anything.
     */
    void popAsCurrent();
    /**
     * Get the metalDevice_t associated with this object.
     */
    metalDevice_t getDevice() {
        return device;
    }
    /**
     * Get the index of the metalDevice_t associated with this object.
     */
    int getDeviceIndex() const {
        return deviceIndex;
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
     * Get the ContextImpl is ComputeContext is associated with.
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
     * Create a new ComputeQueue for use with this context.
     */
    ComputeQueue createQueue();
    /**
     * Get the stream currently being used for execution.
     */
    MetalQueue& getCurrentStream();
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
     * Create an object for performing 3D FFTs.  The caller is responsible for deleting
     * the object when it is no longer needed.
     *
     * @param xsize   the first dimension of the data sets on which FFTs will be performed
     * @param ysize   the second dimension of the data sets on which FFTs will be performed
     * @param zsize   the third dimension of the data sets on which FFTs will be performed
     * @param realToComplex  if true, a real-to-complex transform will be done.  Otherwise, it is complex-to-complex.
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
    std::string doubleToString(double value, bool mixedIsDouble=false) const;
    /**
     * Convert an array to an MetalArray.  If the argument is already an MetalArray, this simply casts it.
     * If the argument is a ComputeArray that wraps a MetalArray, this returns the wrapped array.  For any
     * other argument, this throws an exception.
     */
    MetalArray& unwrap(ArrayInterface& array) const;
    /**
     * Get the array which contains the force on each atom (represented as three long longs in 64 bit fixed point).
     */
    ArrayInterface& getForce() {
        return longForceBuffer;
    }
    /**
     * The METAL platform does not use floating point force buffers, so this throws an exception.
     */
    ArrayInterface& getFloatForceBuffer() {
        throw OpenMMException("METAL platform does not use floating point force buffers");
    }
    /**
     * Not all METAL devices support 64 bit atomics, so this throws an exception.
     * @return
     */
    ArrayInterface& getForceBuffers() {
        throw OpenMMException("METAL platform does not use floating point force buffers");
    }
    /**
     * Get the array which contains the buffer in which energy is computed.
     */
    MetalArray& getEnergyBuffer() {
        return energyBuffer;
    }
    /**
     * Get a pointer to a block of pinned memory that can be used for efficient transfers between host and device.
     * This is guaranteed to be at least as large as any of the arrays returned by methods of this class.
     */
    void* getPinnedBuffer() {
        return pinnedBuffer;
    }
    /**
     * Get a shared ThreadPool that code can use to parallelize operations.
     *
     * Because this object is freely available to all code, care is needed to avoid conflicts.  Only use it
     * from the main thread, and make sure all operations are complete before you invoke any other code that
     * might make use of it
     */
    ThreadPool& getThreadPool() {
        return getPlatformData().threads;
    }
    /**
     * Get a file name in tempDir unique for the current process and context.
     */
    std::string getTempFileName() const;
    /**
     * Get src hash.
     */
    std::string getHash(const std::string& src) const;
    /**
     * Create a METAL module from source code.
     *
     * @param source             the source code of the module
     */
    metalModule_t createModule(const std::string source);
    /**
     * Create a METAL module from source code.
     *
     * @param source             the source code of the module
     * @param defines            a set of preprocessor definitions (name, value) to define when compiling the program
     */
    metalModule_t createModule(const std::string source, const std::map<std::string, std::string>& defines);
    /**
     * Get a kernel from a METAL module.
     *
     * @param module    the module to get the kernel from
     * @param name      the name of the kernel to get
     */
    metalFunction_t getKernel(metalModule_t& module, const std::string& name);
    /**
     * Allocate a shared buffer, or throw with its name and requested byte size.
     */
    metalDeviceptr_t createBuffer(size_t bytes, const std::string& name);
    /**
     * How a kernel's buffer argument reads memory: as ordinary values, as float pairs (a mixed4s buffer in mixed
     * precision), or as raw 32 bit words that work with either, as the clear kernels do.
     */
    enum BufferLayout : char {PlainLayout, PairLayout, AnyLayout};
    /**
     * Execute a kernel.
     *
     * @param kernel       the kernel to execute
     * @param arguments    an array of pointers to the kernel arguments
     * @param threads      the maximum number of threads that should be used
     * @param blockSize    the size of each thread block to use
     * @param sharedSize   the amount of dynamic shared memory to allocated for the kernel, in bytes
     */
    void executeKernel(metalFunction_t kernel, void** arguments, int threads, int blockSize = -1, unsigned int sharedSize = 0);
    /**
     * Execute a kernel with full grid.
     *
     * @param kernel       the kernel to execute
     * @param arguments    an array of pointers to the kernel arguments
     * @param threads      the total number of threads that should be used
     * @param blockSize    the size of each thread block to use
     * @param sharedSize   the amount of dynamic shared memory to allocated for the kernel, in bytes
     */
    void executeKernelFlat(metalFunction_t kernel, void** arguments, int threads, int blockSize = -1, unsigned int sharedSize = 0,
            MTL::Buffer* indirect = NULL, size_t indirectOffset = 0);
    /**
     * Get the layout each buffer argument of a kernel reads (see MetalArray::setPairStorage).
     */
    const std::vector<BufferLayout>& getArgumentLayouts(metalFunction_t kernel) {
        return kernelArguments[kernel].layouts;
    }
    /**
     * Compute the largest thread block size that can be used for a kernel that requires a particular amount of
     * shared memory per thread.
     *
     * @param memory        the number of bytes of shared memory per thread
     */
    int computeThreadBlockSize(double memory) const;
    /**
     * Sum the buffer containing energy.
     */
    double reduceEnergy();
    /** Queue an energy evaluation into a private reduction buffer, retaining the two host sums. */
    void enqueueEnergy(MetalArray& result, int groups, double& forceEnergy, double& platformEnergy);
    /** Read a completed reduction in the same order as reduceEnergy(). */
    double readEnergy(MetalArray& result);
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
        return simdWidth;
    }
    /**
     * Get the number of multiprocessors (compute units) of the device being used.
     */
    int getMultiprocessors() const {
        return multiprocessors;
    }
    /**
     * Get whether the device being used supports 64 bit atomic operations on global memory.
     */
    bool getSupports64BitGlobalAtomics() const {
        return false;
    }
    /**
     * Get whether the device being used supports 32 bit floating point atomic operations
     * on global memory (fast hardware instructions, not a compare-and-swap loop implementation).
     */
    bool getSupportsHardwareFloatGlobalAtomicAdd() const {
        return supportsHardwareFloatGlobalAtomicAdd;
    }
    /**
     * Get whether PME spreads charge with 64 bit fixed point atomics.  Otherwise it spreads with
     * float atomics through per-SIMD-group slabs (pme.cc), launched with SpreadBlockSize threads.
     */
    bool getUseFixedPointChargeSpreading() {
        return useDoublePrecision || !supportsHardwareFloatGlobalAtomicAdd || platformData.deterministicForces;
    }
    /**
     * Get whether gridInterpolateForce gives each atom PME_ORDER lanes (pme.cc, PME_INTERP_LANES).  MetalKernel
     * then launches 32 threads per 32/PME_ORDER atoms.  Only Apple9 does.
     */
    bool getUsePmeInterpolationLanes() const {
        return !useDoublePrecision && supportsHardwareFloatGlobalAtomicAdd;
    }
    /**
     * Get whether the device being used supports double precision math.
     */
    bool getSupportsDoublePrecision() const {
        return useMixedPrecision;
    }
    /**
     * Get the vectors defining the periodic box.
     */
    void getPeriodicBoxVectors(Vec3& a, Vec3& b, Vec3& c) const {
        a = Vec3(periodicBoxVecX.x, periodicBoxVecX.y, periodicBoxVecX.z);
        b = Vec3(periodicBoxVecY.x, periodicBoxVecY.y, periodicBoxVecY.z);
        c = Vec3(periodicBoxVecZ.x, periodicBoxVecZ.y, periodicBoxVecZ.z);
    }
    /**
     * Set the vectors defining the periodic box.
     */
    void setPeriodicBoxVectors(const Vec3& a, const Vec3& b, const Vec3& c) {
        periodicBoxVecX = make_double4(a[0], a[1], a[2], 0.0);
        periodicBoxVecY = make_double4(b[0], b[1], b[2], 0.0);
        periodicBoxVecZ = make_double4(c[0], c[1], c[2], 0.0);
        periodicBoxVecXFloat = make_float4((float) a[0], (float) a[1], (float) a[2], 0.0f);
        periodicBoxVecYFloat = make_float4((float) b[0], (float) b[1], (float) b[2], 0.0f);
        periodicBoxVecZFloat = make_float4((float) c[0], (float) c[1], (float) c[2], 0.0f);
        periodicBoxSize = make_double4(a[0], b[1], c[2], 0.0);
        invPeriodicBoxSize = make_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0);
        periodicBoxSizeFloat = make_float4((float) a[0], (float) b[1], (float) c[2], 0.0f);
        invPeriodicBoxSizeFloat = make_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f);
    }
    /**
     * Get the size of the periodic box.
     */
    double4 getPeriodicBoxSize() const {
        return periodicBoxSize;
    }
    /**
     * Get the inverse of the size of the periodic box.
     */
    double4 getInvPeriodicBoxSize() const {
        return invPeriodicBoxSize;
    }
    /**
     * Get a pointer to the size of the periodic box, represented as either a float4 or double4 depending on
     * this context's precision.  This value is suitable for passing to kernels as an argument.
     */
    void* getPeriodicBoxSizePointer() {
        return (useDoublePrecision ? reinterpret_cast<void*>(&periodicBoxSize) : reinterpret_cast<void*>(&periodicBoxSizeFloat));
    }
    /**
     * Get a pointer to the inverse of the size of the periodic box, represented as either a float4 or double4 depending on
     * this context's precision.  This value is suitable for passing to kernels as an argument.
     */
    void* getInvPeriodicBoxSizePointer() {
        return (useDoublePrecision ? reinterpret_cast<void*>(&invPeriodicBoxSize) : reinterpret_cast<void*>(&invPeriodicBoxSizeFloat));
    }
    /**
     * Get a pointer to the first periodic box vector, represented as either a float4 or double4 depending on
     * this context's precision.  This value is suitable for passing to kernels as an argument.
     */
    void* getPeriodicBoxVecXPointer() {
        return (useDoublePrecision ? reinterpret_cast<void*>(&periodicBoxVecX) : reinterpret_cast<void*>(&periodicBoxVecXFloat));
    }
    /**
     * Get a pointer to the second periodic box vector, represented as either a float4 or double4 depending on
     * this context's precision.  This value is suitable for passing to kernels as an argument.
     */
    void* getPeriodicBoxVecYPointer() {
        return (useDoublePrecision ? reinterpret_cast<void*>(&periodicBoxVecY) : reinterpret_cast<void*>(&periodicBoxVecYFloat));
    }
    /**
     * Get a pointer to the third periodic box vector, represented as either a float4 or double4 depending on
     * this context's precision.  This value is suitable for passing to kernels as an argument.
     */
    void* getPeriodicBoxVecZPointer() {
        return (useDoublePrecision ? reinterpret_cast<void*>(&periodicBoxVecZ) : reinterpret_cast<void*>(&periodicBoxVecZFloat));
    }
    /**
     * Get the MetalIntegrationUtilities for this context.
     */
    MetalIntegrationUtilities& getIntegrationUtilities() {
        return *integration;
    }
    /**
     * Get the MetalExpressionUtilities for this context.
     */
    MetalExpressionUtilities& getExpressionUtilities() {
        return *expression;
    }
    /**
     * Get the MetalBondedUtilities for this context.
     */
    MetalBondedUtilities& getBondedUtilities() {
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
    /**
     * Get whether addForcePassKernel() defers kernels.  Only Apple9 does: the M3 Ultra has idle cores during the
     * tail of the nonbonded kernel, and Apple8 was never measured.
     */
    bool getSupportsForcePassKernels() const;
    /**
     * Run a kernel at the start of the next force pass (MetalCalcForcesAndEnergyKernel::finishComputation).  It
     * must not write any buffer the bonded or nonbonded kernels use, except to add to the force buffers with
     * integer atomics.
     */
    void addForcePassKernel(ComputeKernel kernel, int workUnits) {
        forcePassKernels.push_back(std::make_pair(kernel, workUnits));
    }
    /**
     * Run work right after the next force pass, before the post computations.
     */
    void addAfterForcePass(std::function<void()> work) {
        afterForcePass.push_back(work);
    }
    /**
     * Execute the kernels given to addForcePassKernel() since the last call, in order.
     */
    void executeForcePassKernels();
    /**
     * Get the buffers that clearAutoclearBuffers() zeroes.
     */
    const std::vector<ArrayInterface*>& getAutoclearBuffers() const {
        return autoclearBuffers;
    }
    /**
     * Record that the kernel just encoded zeroes every autoclear buffer when it runs.  The next
     * clearAutoclearBuffersIfNeeded() then skips its dispatch, unless any work was encoded or any array written
     * in between, or a buffer was added.
     */
    void markAutoclearBuffersCleared();
    /**
     * Get whether the mark markAutoclearBuffersCleared() set still holds: no work was encoded, no array written and no
     * buffer added since.
     */
    bool getAutoclearBuffersMarkedCleared() const;
    /**
     * Call clearAutoclearBuffers() unless markAutoclearBuffersCleared() says the buffers are already zero.
     */
    void clearAutoclearBuffersIfNeeded();
    /**
     * Run the work given to addAfterForcePass() since the last call, in order.
     */
    void executeAfterForcePass();
    int getNumEnergyPartialSums() const {
        return energySum.getSize();
    }
    void setDeferredEnergy(ArrayInterface* dest, int index) {
        deferredEnergy = dest;
        deferredEnergyIndex = index;
    }
private:
    MetalArray* barostatEnergy = nullptr;   // set only inside enqueueEnergy()
    friend class MetalSort;
    friend class MetalHostRobustnessAccess;
#ifdef OPENMM_METAL_TEST_HOOKS
    static std::function<void(const std::vector<NS::Object*>&)> constructionHook;
#endif
    void releaseResources();
    void releaseModule(metalModule_t module);
    void releaseKernel(metalFunction_t kernel);
    /**
     * How to bind a kernel's arguments.  sizes holds the byte size of each value argument and 0 for
     * each buffer.  When the kernel has too many arguments for separate bindings, offsets holds where
     * each value goes in one struct bound at index 0, and packedSize is that struct's size.  layouts
     * holds the layout each buffer argument reads.
     */
    struct KernelArguments {
        std::vector<int> sizes, offsets;
        std::vector<BufferLayout> layouts;
        int packedSize = 0;
    };
    /**
     * Compile Metal source into a library.  On failure this returns NULL and sets log to the compiler output.
     */
    metalModule_t createLibrary(const std::string& source, std::string& log, bool saveTemps);
    /**
     * Find where each argument of a kernel with packed values goes, from the reflection of its argument
     * struct.  Returns false if the struct has more buffer markers than the kernel has buffer bindings.
     */
    bool getPackedArguments(MTL::BufferBinding* binding, int numBuffers, KernelArguments& args);
    /**
     * Get the layout a buffer argument reads, from its element type.
     */
    static BufferLayout getBufferLayout(MTL::BufferBinding* binding);
    MetalPlatform::PlatformData& platformData;
    int deviceIndex;
    int contextIndex;
    int simdWidth;
    int multiprocessors;
    int sharedMemPerBlock;
    bool supportsHardwareFloatGlobalAtomicAdd;
    bool useBlockingSync, contextIsValid;
    std::vector<std::pair<ComputeKernel, int> > forcePassKernels;
    long long autoclearWorkCount;
    int autoclearBufferCount;
    std::vector<std::function<void()> > afterForcePass;
    std::string tempDir;
    float4 periodicBoxVecXFloat, periodicBoxVecYFloat, periodicBoxVecZFloat, periodicBoxSizeFloat, invPeriodicBoxSizeFloat;
    double4 periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ, periodicBoxSize, invPeriodicBoxSize;
    std::map<std::string, std::string> compilationDefines;
    std::vector<metalModule_t> loadedModules;
    std::set<metalModule_t> packedModules;
    std::map<metalFunction_t, KernelArguments> kernelArguments;
    metalDevice_t device;
    void* pinnedBuffer;
    MetalArray energyBuffer;
    ArrayInterface* deferredEnergy = NULL;
    int deferredEnergyIndex = 0;
    MetalIntegrationUtilities* integration;
    MetalExpressionUtilities* expression;
    MetalBondedUtilities* bonded;
    MetalNonbondedUtilities* nonbonded;
};

/**
 * This class exists only for backward compatibility.  Use ComputeContext::WorkTask instead.
 */
class OPENMM_EXPORT_COMMON MetalContext::WorkTask : public ComputeContext::WorkTask {
};

/**
 * This class exists only for backward compatibility.  Use ComputeContext::ReorderListener instead.
 */
class OPENMM_EXPORT_COMMON MetalContext::ReorderListener : public ComputeContext::ReorderListener {
};

/**
 * This class exists only for backward compatibility.  Use ComputeContext::ForcePreComputation instead.
 */
class OPENMM_EXPORT_COMMON MetalContext::ForcePreComputation : public ComputeContext::ForcePreComputation {
};

/**
 * This class exists only for backward compatibility.  Use ComputeContext::ForcePostComputation instead.
 */
class OPENMM_EXPORT_COMMON MetalContext::ForcePostComputation : public ComputeContext::ForcePostComputation {
};

} // namespace OpenMM

#endif /*OPENMM_METALCONTEXT_H_*/
