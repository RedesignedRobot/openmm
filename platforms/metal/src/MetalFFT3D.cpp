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

#define VKFFT_BACKEND 5
#include <iostream>
#include "vkFFT.h"
#include "MetalFFT3D.h"
#include "MetalContext.h"
#include "MetalKernelSources.h"
#include <algorithm>
#include <cmath>

using namespace OpenMM;
using namespace std;

struct MetalFFT3D::VkFFTApp : VkFFTApplication {
};

// The radices of an in-place FFT of size n, largest first, or an empty list when n has a prime factor above 7.
static vector<int> fftRadices(int n) {
    vector<int> radices;
    for (int radix : {8, 4, 2, 7, 5, 3})
        while (n%radix == 0) {
            radices.push_back(radix);
            n /= radix;
        }
    if (n != 1)
        return vector<int>();
    sort(radices.begin(), radices.end(), greater<int>());
    return radices;
}

bool MetalFFT3D::canUsePmeConvolution(MetalContext& context, int xsize, int ysize, int zsize) {
    // The PME convolution kernels hold a y-z plane, or an x-z plane of the half complex grid, and its roots of unity in
    // threadgroup memory.  On the M3 Ultra they take 22 us per step at 56^3, against 41 us for VkFFT and
    // reciprocalConvolution.  Apple8 keeps VkFFT.
    if (context.getUseDoublePrecision() || !context.getDevice()->supportsFamily(MTL::GPUFamilyApple9))
        return false;
    if (fftRadices(xsize).empty() || fftRadices(ysize).empty() || fftRadices(zsize).empty())
        return false;
    int planeBytes = (ysize*zsize+ysize+zsize)*sizeof(mm_float2);
    int lineBytes = (xsize*(zsize/2+1)+xsize)*sizeof(mm_float2);
    int maxBytes = context.getDevice()->maxThreadgroupMemoryLength();
    return (planeBytes <= maxBytes && lineBytes <= maxBytes && max(xsize, ysize) <= context.getNumThreadBlocks());
}

MetalFFT3D::MetalFFT3D(MetalContext& context, int xsize, int ysize, int zsize, bool realToComplex) : context(context),
        xsize(xsize), ysize(ysize), zsize(zsize) {
    usePmeConvolution = (realToComplex && canUsePmeConvolution(context, xsize, ysize, zsize));
    radicesX = fftRadices(xsize);
    radicesY = fftRadices(ysize);
    radicesZ = fftRadices(zsize);
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
    MetalQueue& queue = context.getCurrentStream();
    std::lock_guard<std::recursive_mutex> guard(queue.getLock());
    VkFFTLaunchParams params = {};
    params.commandEncoder = queue.getEncoder();
    VkFFTResult fftResult = VkFFTAppend(app, forward ? -1 : 1, &params);
    if (fftResult != VKFFT_SUCCESS) {
        throw OpenMMException("Error executing VkFFTAppend: "+context.intToString(fftResult));
    }
}

static string joinRadices(const vector<int>& radices) {
    string result;
    for (int radix : radices)
        result += (result.empty() ? "" : ",")+to_string(radix);
    return result;
}

// The stages of a decimation in frequency transform as M(radix, length) entries, in order or reversed.
static string stageList(int size, const vector<int>& radices, bool reverse) {
    vector<string> stages;
    for (int radix : radices) {
        stages.push_back("M("+to_string(radix)+", "+to_string(size)+")");
        size /= radix;
    }
    if (reverse)
        std::reverse(stages.begin(), stages.end());
    string result;
    for (string& stage : stages)
        result += stage+" ";
    return result;
}

static void uploadRoots(MetalContext& context, MetalArray& roots, int n, const string& name) {
    vector<mm_float2> values(n);
    for (int t = 0; t < n; t++)
        values[t] = mm_float2((float) cos(2*M_PI*t/n), (float) -sin(2*M_PI*t/n));
    roots.initialize<mm_float2>(context, n, name);
    roots.upload(values);
}

void MetalFFT3D::createPmeConvolution() {
    map<string, string> defines;
    defines["GRID_SIZE_X"] = context.intToString(xsize);
    defines["GRID_SIZE_Y"] = context.intToString(ysize);
    defines["GRID_SIZE_Z"] = context.intToString(zsize);
    const char* axes[] = {"X", "Y", "Z"};
    int sizes[] = {xsize, ysize, zsize};
    vector<int>* radices[] = {&radicesX, &radicesY, &radicesZ};
    for (int i = 0; i < 3; i++) {
        string axis = axes[i];
        defines["RADICES_"+axis] = joinRadices(*radices[i]);
        defines["NUM_STAGES_"+axis] = context.intToString(radices[i]->size());
        defines["STAGES_"+axis+"(M)"] = stageList(sizes[i], *radices[i], false);
        defines["STAGES_"+axis+"_REV(M)"] = stageList(sizes[i], *radices[i], true);
    }
    for (int radix : {3, 5, 7}) {
        string c, s;
        for (int t = 0; t < radix; t++) {
            c += (t == 0 ? "" : ",")+context.doubleToString(cos(2*M_PI*t/radix));
            s += (t == 0 ? "" : ",")+context.doubleToString(sin(2*M_PI*t/radix));
        }
        defines["ROOTS_C"+to_string(radix)] = c;
        defines["ROOTS_S"+to_string(radix)] = s;
    }
    ComputeProgram program = context.compileProgram(MetalKernelSources::pmeConvolution, defines);
    planesForwardKernel = program->createKernel("planesForward");
    convolveLinesKernel = program->createKernel("convolveLines");
    planesInverseKernel = program->createKernel("planesInverse");
    uploadRoots(context, rootsX, xsize, "pmeRootsX");
    uploadRoots(context, rootsY, ysize, "pmeRootsY");
    uploadRoots(context, rootsZ, zsize, "pmeRootsZ");
    for (int i = 0; i < 4; i++) {
        planesForwardKernel->addArg();
        planesInverseKernel->addArg();
    }
    for (int i = 0; i < 9; i++)
        convolveLinesKernel->addArg();
}

// Each kernel runs one threadgroup per plane or line, as wide as the pipeline allows up to 1024 threads.
static void executePerGroup(ComputeKernel& kernel, int groups) {
    int threads = min(1024, kernel->getMaxBlockSize()/32*32);
    kernel->execute(groups*threads, threads);
}

bool MetalFFT3D::execPmeConvolution(ArrayInterface& grid, ArrayInterface& scratch, ArrayInterface& moduliX, ArrayInterface& moduliY,
        ArrayInterface& moduliZ, const mm_double4* recipBoxVectors, double recipExpFactor) {
    for (int stage = 0; stage < 3; stage++)
        if (!execPmeConvolutionStage(stage, grid, scratch, moduliX, moduliY, moduliZ, recipBoxVectors, recipExpFactor))
            return false;
    return true;
}

bool MetalFFT3D::execPmeConvolutionStage(int stage, ArrayInterface& grid, ArrayInterface& scratch, ArrayInterface& moduliX,
        ArrayInterface& moduliY, ArrayInterface& moduliZ, const mm_double4* recipBoxVectors, double recipExpFactor) {
    if (!usePmeConvolution)
        return false;
    if (!planesForwardKernel)
        createPmeConvolution();
    if (stage == 0) {
        planesForwardKernel->setArg(0, grid);
        planesForwardKernel->setArg(1, scratch);
        planesForwardKernel->setArg(2, rootsY);
        planesForwardKernel->setArg(3, rootsZ);
        executePerGroup(planesForwardKernel, xsize);
    }
    else if (stage == 1) {
        convolveLinesKernel->setArg(0, scratch);
        convolveLinesKernel->setArg(1, rootsX);
        convolveLinesKernel->setArg(2, moduliX);
        convolveLinesKernel->setArg(3, moduliY);
        convolveLinesKernel->setArg(4, moduliZ);
        for (int i = 0; i < 3; i++)
            convolveLinesKernel->setArg(5+i, mm_float4((float) recipBoxVectors[i].x, (float) recipBoxVectors[i].y, (float) recipBoxVectors[i].z, 0));
        convolveLinesKernel->setArg(8, (float) recipExpFactor);
        executePerGroup(convolveLinesKernel, ysize);
    }
    else {
        planesInverseKernel->setArg(0, scratch);
        planesInverseKernel->setArg(1, grid);
        planesInverseKernel->setArg(2, rootsY);
        planesInverseKernel->setArg(3, rootsZ);
        executePerGroup(planesInverseKernel, xsize);
    }
    return true;
}

int MetalFFT3D::findLegalDimension(int minimum, int maxFactor) {
    if (minimum < 1)
        return 1;
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
