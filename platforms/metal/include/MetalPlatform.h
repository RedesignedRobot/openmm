#ifndef OPENMM_METALPLATFORM_H_
#define OPENMM_METALPLATFORM_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2026 Stanford University and the Authors.      *
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

#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/internal/ThreadPool.h"
#include "openmm/common/windowsExportCommon.h"

namespace OpenMM {

class MetalContext;

/**
 * This Platform subclass uses METAL implementations of the OpenMM kernels.
 */

class OPENMM_EXPORT_COMMON MetalPlatform : public Platform {
public:
    class PlatformData;
    MetalPlatform();
    const std::string& getName() const {
        static const std::string name = "Metal";
        return name;
    }
    double getSpeed() const;
    bool supportsDoublePrecision() const;
    const std::string& getPropertyValue(const Context& context, const std::string& property) const;
    void setPropertyValue(Context& context, const std::string& property, const std::string& value) const;
    std::vector<std::map<std::string, std::string> > getDevices(const std::map<std::string, std::string>& filters={}) const;
    void contextCreated(ContextImpl& context, const std::map<std::string, std::string>& properties) const;
    void linkedContextCreated(ContextImpl& context, ContextImpl& originalContext) const;
    void contextDestroyed(ContextImpl& context) const;
    /**
     * This is the name of the parameter for selecting which METAL device or devices to use.
     */
    static const std::string& MetalDeviceIndex() {
        static const std::string key = "DeviceIndex";
        return key;
    }
    /**
     * This is the name of the parameter that reports the METAL device or devices being used.
     */
    static const std::string& MetalDeviceName() {
        static const std::string key = "DeviceName";
        return key;
    }
    /**
     * This is the name of the parameter for selecting whether METAL should sync or spin loop while waiting for results.
     */
    static const std::string& MetalUseBlockingSync() {
        static const std::string key = "UseBlockingSync";
        return key;
    }
    /**
     * This is the name of the parameter for selecting what numerical precision to use.
     */
    static const std::string& MetalPrecision() {
        static const std::string key = "Precision";
        return key;
    }
    /**
     * This is the name of the parameter for selecting whether to use the CPU based PME calculation.
     */
    static const std::string& MetalUseCpuPme() {
        static const std::string key = "UseCpuPme";
        return key;
    }
    /**
     * This is the name of the parameter for specifying the path to the directory for creating temporary files.
     */
    static const std::string& MetalTempDirectory() {
        static const std::string key = "TempDirectory";
        return key;
    }
    /**
     * This is the name of the parameter for selecting whether to disable use of a separate stream for PME.
     */
    static const std::string& MetalDisablePmeStream() {
        static const std::string key = "DisablePmeStream";
        return key;
    }
    /**
     * This is the name of the parameter for requesting that force computations be fully deterministic.
     */
    static const std::string& MetalDeterministicForces() {
        static const std::string key = "DeterministicForces";
        return key;
    }
};

class OPENMM_EXPORT_COMMON MetalPlatform::PlatformData {
public:
    PlatformData(ContextImpl* context, const System& system, const std::string& deviceIndexProperty, const std::string& blockingProperty, const std::string& precisionProperty,
            const std::string& cpuPmeProperty, const std::string& tempProperty,
            const std::string& pmeStreamProperty, const std::string& deterministicForcesProperty, int numThreads, ContextImpl* originalContext);
    ~PlatformData();
    void initializeContexts(const System& system);
    void syncContexts();
    ContextImpl* context;
    std::vector<MetalContext*> contexts;
    std::vector<double> contextEnergy;
    bool hasInitializedContexts, removeCM, useCpuPme, disablePmeStream, deterministicForces;
    int cmMotionFrequency, computeForceCount;
    long long stepCount;
    double time;
    std::map<std::string, std::string> propertyValues;
    ThreadPool threads;
};

} // namespace OpenMM

#endif /*OPENMM_METALPLATFORM_H_*/
