/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
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

#include "MetalProgram.h"
#include "MetalKernel.h"

using namespace OpenMM;
using namespace std;

/**
 * Get the layout a buffer argument reads, from its element type.
 */
static MetalKernel::BufferLayout getBufferLayout(MTL::BufferBinding* binding) {
    MTL::DataType type = binding->bufferDataType();
    if (type == MTL::DataTypeInt || type == MTL::DataTypeUInt)
        return MetalKernel::AnyLayout;
    if (type != MTL::DataTypeStruct)
        return MetalKernel::PlainLayout;
    // A df64s_4 element is the only struct whose member x is a struct with a member shi.
    MTL::StructMember* x = binding->bufferStructType()->memberByName(NS::String::string("x", NS::UTF8StringEncoding));
    if (x == nullptr || x->dataType() != MTL::DataTypeStruct)
        return MetalKernel::PlainLayout;
    return (x->structType()->memberByName(NS::String::string("shi", NS::UTF8StringEncoding)) != nullptr ? MetalKernel::PairLayout : MetalKernel::PlainLayout);
}

MetalProgram::MetalProgram(MetalContext& context, MTL::Library* library, bool packArgs) : context(context),
        library(library), packArgs(packArgs) {
}

ComputeKernel MetalProgram::createKernel(const string& name) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Function* function = library->newFunction(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
    if (function == nullptr) {
        pool->release();
        throw OpenMMException("Error creating kernel "+name+": no such function");
    }
    NS::Error* error = nullptr;
    MTL::ComputePipelineState* pipeline;
    vector<MetalKernel::BufferLayout> layouts;
    if (context.getUseMixedPrecision() && !packArgs) {
        // Record which buffer arguments read float pairs, so execute() can check the arrays bound to them.  Packed
        // arguments carry no type information, so those kernels go unchecked.
        MTL::ComputePipelineReflection* reflection = nullptr;
        pipeline = context.getDevice().newComputePipelineState(function, MTL::PipelineOptionBindingInfo | MTL::PipelineOptionBufferTypeInfo, &reflection, &error);
        for (int i = 0; pipeline != nullptr && reflection != nullptr && i < reflection->bindings()->count(); i++) {
            MTL::Binding* binding = reflection->bindings()->object<MTL::Binding>(i);
            if (binding->type() != MTL::BindingTypeBuffer)
                continue;
            if (binding->index() >= layouts.size())
                layouts.resize(binding->index()+1, MetalKernel::PlainLayout);
            layouts[binding->index()] = getBufferLayout((MTL::BufferBinding*) binding);
        }
    }
    else
        pipeline = context.getDevice().newComputePipelineState(function, &error);
    if (pipeline == nullptr) {
        string message = "Error creating pipeline state for kernel "+name;
        if (error != nullptr)
            message += string(": ")+error->localizedDescription()->utf8String();
        pool->release();
        function->release();
        throw OpenMMException(message);
    }
    pool->release();
    return shared_ptr<ComputeKernelImpl>(new MetalKernel(context, pipeline, name, packArgs, layouts));
}