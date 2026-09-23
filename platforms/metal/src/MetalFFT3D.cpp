/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2015 Stanford University and the Authors.      *
 * Portions copyright (c) 2021 Advanced Micro Devices, Inc.                   *
 * Authors:                                                                   *
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

struct MetalFFT3D::VkFFTApp : VkFFTApplication {
};

MetalFFT3D::MetalFFT3D(MetalContext& context, int xsize, int ysize, int zsize, bool realToComplex) : context(context) {
    size_t valueSize = context.getUseDoublePrecision() ? sizeof(double) : sizeof(float);
    inputBufferSize = zsize * ysize * xsize * valueSize;
    if (realToComplex) {
        outputBufferSize = (zsize/2 + 1) * ysize * xsize * valueSize * 2;
    }
    else {
        outputBufferSize = zsize * ysize * xsize * valueSize;
    }

    VkFFTConfiguration configuration = {};
    configuration.performR2C = realToComplex;
    configuration.device = context.getDevice();
    configuration.queue = context.getCurrentStream().getStream();
    configuration.doublePrecision = context.getUseDoublePrecision();

    configuration.FFTdim = 3;
    configuration.size[0] = zsize;
    configuration.size[1] = ysize;
    configuration.size[2] = xsize;

    configuration.inverseReturnToInputBuffer = true;
    configuration.isInputFormatted = true;
    configuration.inputBufferSize = &inputBufferSize;
    configuration.inputBuffer = &inputBuffer;
    configuration.inputBufferStride[0] = zsize;
    configuration.inputBufferStride[1] = configuration.inputBufferStride[0] * ysize;
    configuration.inputBufferStride[2] = configuration.inputBufferStride[1] * xsize;

    configuration.bufferSize = &outputBufferSize;
    configuration.buffer = &outputBuffer;
    configuration.bufferStride[0] = realToComplex ? (zsize/2 + 1) : zsize;
    configuration.bufferStride[1] = configuration.bufferStride[0] * ysize;
    configuration.bufferStride[2] = configuration.bufferStride[1] * xsize;

    // VkFFT compiles its kernels with fast math, so use lookup tables for the twiddle factors
    // rather than trusting the fast trig functions.
    configuration.useLUT = 1;

    app = new VkFFTApp();
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    VkFFTResult fftResult = initializeVkFFT(app, configuration);
    pool->release();
    if (fftResult != VKFFT_SUCCESS) {
        throw OpenMMException("Error executing initializeVkFFT: "+context.intToString(fftResult));
    }
}

MetalFFT3D::~MetalFFT3D() {
    deleteVkFFT(app);
    delete app;
}

void MetalFFT3D::execFFT(ArrayInterface& in, ArrayInterface& out, bool forward) {
    if (forward) {
        inputBuffer = context.unwrap(in).getDevicePointer();
        outputBuffer = context.unwrap(out).getDevicePointer();
    }
    else {
        inputBuffer = context.unwrap(out).getDevicePointer();
        outputBuffer = context.unwrap(in).getDevicePointer();
    }
    // VkFFT's Metal backend encodes only through commandEncoder, so the FFT is batched into the
    // open command buffer like any other kernel.
    MetalQueue& queue = context.getCurrentStream();
    std::lock_guard<std::recursive_mutex> guard(queue.getLock());
    VkFFTLaunchParams params = {};
    params.commandEncoder = queue.getEncoder();
    VkFFTResult fftResult = VkFFTAppend(app, forward ? -1 : 1, &params);
    if (fftResult != VKFFT_SUCCESS) {
        throw OpenMMException("Error executing VkFFTAppend: "+context.intToString(fftResult));
    }
}

int MetalFFT3D::findLegalDimension(int minimum) {
    if (minimum < 1)
        return 1;
    while (true) {
        // Attempt to factor the current value.

        int unfactored = minimum;
        // VkFFT supports prime factors up to 13
        for (int factor = 2; factor <= 13; factor++) {
            while (unfactored > 1 && unfactored%factor == 0)
                unfactored /= factor;
        }
        if (unfactored == 1)
            return minimum;
        minimum++;
    }
}
