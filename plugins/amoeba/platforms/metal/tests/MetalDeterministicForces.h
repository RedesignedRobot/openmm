#ifndef OPENMM_METAL_DETERMINISTIC_FORCES_H_
#define OPENMM_METAL_DETERMINISTIC_FORCES_H_

#include "MetalPlatform.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/AssertionUtilities.h"
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// evaluateForces must reset positions before every evaluation to invalidate cached forces.
template <class EvaluateForces>
void assertDeterministicForces(OpenMM::System& system, OpenMM::MetalPlatform& platform, EvaluateForces evaluateForces) {
    using OpenMM::throwException;
    std::map<std::string, std::string> properties;
    properties[OpenMM::MetalPlatform::MetalDeterministicForces()] = "true";
    const std::string precision = platform.getPropertyDefaultValue(OpenMM::MetalPlatform::MetalPrecision());
    std::vector<OpenMM::Vec3> expected;
    for (int instance = 0; instance < 2; instance++) {
        OpenMM::VerletIntegrator integrator(0.001);
        OpenMM::Context context(system, integrator, platform, properties);
        ASSERT_EQUAL(precision, platform.getPropertyValue(context, OpenMM::MetalPlatform::MetalPrecision()));
        ASSERT_EQUAL(std::string("true"), platform.getPropertyValue(context, OpenMM::MetalPlatform::MetalDeterministicForces()));
        for (int evaluation = 0; evaluation < 4; evaluation++) {
            const std::vector<OpenMM::Vec3> forces = evaluateForces(context);
            ASSERT_EQUAL(system.getNumParticles(), forces.size());
            if (instance == 0 && evaluation == 0) {
                expected = forces;
                double norm = 0.0;
                for (const auto& force : forces)
                    norm += force.dot(force);
                ASSERT(std::isfinite(norm) && norm > 0.0);
            }
            else {
                for (int i = 0; i < system.getNumParticles(); i++) {
                    for (int axis = 0; axis < 3; axis++) {
                        const double reference = expected[i][axis];
                        const double value = forces[i][axis];
                        ASSERT(std::memcmp(&reference, &value, sizeof(double)) == 0);
                    }
                }
            }
        }
    }
}

#endif
