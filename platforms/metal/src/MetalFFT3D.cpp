/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2025 Stanford University and the Authors.      *
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
// This is the only translation unit that includes vkFFT.h.  With the Metal backend it defines
// NS_PRIVATE_IMPLEMENTATION, CA_PRIVATE_IMPLEMENTATION, and MTL_PRIVATE_IMPLEMENTATION before
// including metal-cpp, which emits the metal-cpp symbols for the whole library.
#define VKFFT_BACKEND 5
#include <iostream> // vkFFT.h prints Metal compiler errors to std::cout without including it
#include "vkFFT.h"
#include "MetalFFT3D.h"
#include "MetalContext.h"
#include "MetalQueue.h"

using namespace OpenMM;
using namespace std;

struct MetalFFT3D::VkFFTState {
    VkFFTApplication app;
    MTL::Buffer* inputBuffer;
    MTL::Buffer* buffer;
};

MetalFFT3D::MetalFFT3D(MetalContext& context, int xsize, int ysize, int zsize, bool realToComplex) : context(context) {
    state = new VkFFTState();
    MTL::Device* device = context.getDevice();
    MTL::CommandQueue* queue = context.getCurrentMetalQueue().getQueue();
    VkFFTConfiguration config = {};
    config.FFTdim = 3;
    config.size[0] = zsize;
    config.size[1] = ysize;
    config.size[2] = xsize;
    config.performR2C = realToComplex;
    config.device = device;
    config.queue = queue;
    config.inverseReturnToInputBuffer = true;
    config.isInputFormatted = 1;
    config.inputBufferStride[0] = zsize;
    config.inputBufferStride[1] = ysize*zsize;
    config.inputBufferStride[2] = xsize*ysize*zsize;
    // VkFFT compiles its kernels with fast math, so use lookup tables for the twiddle factors
    // rather than trusting the fast trig functions.
    config.useLUT = 1;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    VkFFTResult result = initializeVkFFT(&state->app, config);
    pool->release();
    if (result != VKFFT_SUCCESS) {
        delete state;
        throw OpenMMException("Error initializing VkFFT: "+context.intToString(result));
    }
}

MetalFFT3D::~MetalFFT3D() {
    deleteVkFFT(&state->app);
    delete state;
}

void MetalFFT3D::execFFT(ArrayInterface& in, ArrayInterface& out, bool forward) {
    if (forward) {
        state->inputBuffer = context.unwrap(in).getBuffer();
        state->buffer = context.unwrap(out).getBuffer();
    }
    else {
        state->inputBuffer = context.unwrap(out).getBuffer();
        state->buffer = context.unwrap(in).getBuffer();
    }
    // VkFFT's Metal backend encodes only through commandEncoder, so the FFT is batched into the
    // open command buffer like any other kernel.
    VkFFTLaunchParams params = {};
    params.commandEncoder = context.getCurrentMetalQueue().getEncoder();
    params.inputBuffer = &state->inputBuffer;
    params.buffer = &state->buffer;
    VkFFTResult result = VkFFTAppend(&state->app, forward ? -1 : 1, &params);
    if (result != VKFFT_SUCCESS)
        throw OpenMMException("Error executing VkFFT: "+context.intToString(result));
}

int MetalFFT3D::findLegalDimension(int minimum) {
    if (minimum < 1)
        return 1;
    const int maxFactor = 13;
    while (true) {
        // Attempt to factor the current value.

        int unfactored = minimum;
        for (int factor = 2; factor <= maxFactor; factor++) {
            while (unfactored > 1 && unfactored%factor == 0)
                unfactored /= factor;
        }
        if (unfactored == 1)
            return minimum;
        minimum++;
    }
}
