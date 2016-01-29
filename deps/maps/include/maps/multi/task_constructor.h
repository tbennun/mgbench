// MAPS - Memory Access Pattern Specification Framework
// http://maps-gpu.github.io/
// Copyright (c) 2015, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef __MAPS_MULTI_TASK_CONSTRUCTOR_H
#define __MAPS_MULTI_TASK_CONSTRUCTOR_H

#include <vector>
#include "common.h"
#include "input_containers.h"
#include "output_containers.h"
#include "graph_containers.h"
#include "aggregators.h"

namespace maps
{
    namespace multi
    {

        inline void ConstructArgs(Task& task)
        {
            // Do nothing
        }

        // Defined for the rest of the types (PODs)
        template<typename Arg>
        inline void ConstructArgs(Task& task, const Arg& arg)
        {
            MAPS_STATIC_ASSERT(std::is_pod<Arg>::value == true, "Argument must be POD");          

            // Uncommenting the next line shows what type is erroneous
            //arg.non_existing_method();

            std::shared_ptr<Arg> ptr = std::make_shared<Arg>();
            *ptr = arg; // copy arg by value

            task.constants.push_back(ptr); // std::shared_ptr<Arg> is implicitly converted to std::shared_ptr<void>
            
            task.argument_ordering.push_back(AT_CONSTANT);
        }

        /////////////////////////////////////////////////////////
        // INPUT CONTAINERS        

        template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int WINDOW_APRON,
                 BorderBehavior BORDER, int ITEMS_PER_THREAD, int ROWS_PER_THREAD>
        inline void ConstructArgs(Task& task, const Window2D<T, BLOCK_WIDTH, BLOCK_HEIGHT, WINDOW_APRON, BORDER, ITEMS_PER_THREAD, ROWS_PER_THREAD>& arg)
        {
            task.inputs.push_back(TaskInput(arg.datum, arg.CreateSegmenter(), arg.CreateContainerFactory()));
            task.argument_ordering.push_back(AT_INPUT);
        }

        template<typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, int WINDOW_APRON, int IPX, int IPY, int IPZ,
                 BorderBehavior BORDER>
        inline void ConstructArgs(Task& task, const Window<T, DIMS, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, WINDOW_APRON, IPX, IPY, IPZ, BORDER>& arg)
        {
            task.inputs.push_back(TaskInput(arg.datum, arg.CreateSegmenter(), arg.CreateContainerFactory()));
            task.argument_ordering.push_back(AT_INPUT);
        }

        template<typename T, int WINDOW_APRON, BorderBehavior BORDER>
        inline void ConstructArgs(Task& task, const Window2DUnmodified<T, WINDOW_APRON, BORDER>& arg)
        {
            task.inputs.push_back(TaskInput(arg.datum, arg.CreateSegmenter(), arg.CreateContainerFactory()));
            task.argument_ordering.push_back(AT_INPUT);
        }

        template<typename T, int WINDOW_APRON, BorderBehavior BORDER>
        inline void ConstructArgs(Task& task, const Window4DUnmodified<T, WINDOW_APRON, BORDER>& arg)
        {
            task.inputs.push_back(TaskInput(arg.datum, arg.CreateSegmenter(), arg.CreateContainerFactory()));
            task.argument_ordering.push_back(AT_INPUT);
        }

        template<typename T, int WINDOW_APRON_X, int WINDOW_APRON_Y, int STRIDE_X, int STRIDE_Y>
        inline void ConstructArgs(Task& task, const Window3DLayeredUnmodified<T, WINDOW_APRON_X, WINDOW_APRON_Y, STRIDE_X, STRIDE_Y>& arg)
        {
            task.inputs.push_back(TaskInput(arg.datum, arg.CreateSegmenter(), arg.CreateContainerFactory()));
            task.argument_ordering.push_back(AT_INPUT);
        }

        template<typename T, int DIMS, int PRINCIPAL_DIM, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, int IPX, int IPY, int IPZ,
                 BorderBehavior BORDER>
            inline void ConstructArgs(Task& task, const Block<T, DIMS, PRINCIPAL_DIM, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, IPX, IPY, IPZ, BORDER>& arg)
        {
            task.inputs.push_back(TaskInput(arg.datum, arg.CreateSegmenter(), arg.CreateContainerFactory()));
            task.argument_ordering.push_back(AT_INPUT);
        }

        template<typename T>
        inline void ConstructArgs(Task& task, const Block1DUnmodified<T>& arg)
        {
            task.inputs.push_back(TaskInput(arg.datum, arg.CreateSegmenter(), arg.CreateContainerFactory()));
            task.argument_ordering.push_back(AT_INPUT);
        }

        template<bool TRANSPOSED, typename T>
        inline void ConstructArgs(Task& task, const Block2DUnmodified<TRANSPOSED, T>& arg)
        {
            task.inputs.push_back(TaskInput(arg.datum, arg.CreateSegmenter(), arg.CreateContainerFactory()));
            task.argument_ordering.push_back(AT_INPUT);
        }

        template<bool TRANSPOSED, typename T>
        inline void ConstructArgs(Task& task, const Block2D4DUnmodified<TRANSPOSED, T>& arg)
        {
            task.inputs.push_back(TaskInput(arg.datum, arg.CreateSegmenter(), arg.CreateContainerFactory()));
            task.argument_ordering.push_back(AT_INPUT);
        }

        template<typename TWeight, typename TValue, int BLOCK_SIZE>
        inline void ConstructArgs(Task& task, const Adjacency<TWeight, TValue, BLOCK_SIZE>& arg)
        {
            arg.AppendArgsToTask(task);
        }

        template<typename T, int DIMS>
        inline void ConstructArgs(Task& task, const IrregularInput<T, DIMS>& arg)
        {
            task.inputs.push_back(TaskInput(arg.datum, arg.CreateSegmenter(), arg.CreateContainerFactory()));
            task.argument_ordering.push_back(AT_INPUT);
        }

        /////////////////////////////////////////////////////////
        // OUTPUT CONTAINERS

        template<typename T, int DIMS, int ITEMS_PER_THREAD, int ROWS_PER_THREAD, ILPScheme ILP_SCHEME>
        inline void ConstructArgs(Task& task, const StructuredInjectiveOutput<T, DIMS, ITEMS_PER_THREAD, ROWS_PER_THREAD, ILP_SCHEME>& arg)
        {
            // Notify the scheduler to segment the kernels by this output automatically
            if (task.segmentation_output_index < 0)
                task.segmentation_output_index = (int)task.outputs.size();

            task.outputs.push_back(TaskOutput(arg.datum, arg.CreateSegmenter(), 
                                              arg.CreateContainerFactory(), OCT_STRUCTURED_INJECTIVE, nullptr));
            task.argument_ordering.push_back(AT_OUTPUT);
        }

        template<typename T, int LENGTH, int BLOCK_WIDTH, int ITEMS_PER_THREAD, typename Agg>
        inline void ConstructArgs(Task& task, const ReductiveStaticOutput<T, LENGTH, BLOCK_WIDTH, ITEMS_PER_THREAD, Agg>& arg)
        {
            task.outputs.push_back(TaskOutput(arg.datum, arg.CreateSegmenter(), 
                                                   arg.CreateContainerFactory(), OCT_REDUCTIVE_CONSTANT, arg.CreateAggregator()));
            task.argument_ordering.push_back(AT_OUTPUT);
        }

        /////////////////////////////////////////////////////////

        template<typename First, typename... Args>
        inline void ConstructArgs(Task& task, const First& first, const Args&... args)
        {
            ConstructArgs(task, first);
            ConstructArgs(task, args...);
        }

        template<typename K>
        inline void ConstructCall(Task& task, K kernel)
        {
            task.kernel = (void *)kernel;
        }

        /**
         * Generates the grid segmentation of the task based on its output containers
         */
        static void ConstructSegmentation(Task& task, unsigned int numGPUs, bool bInvokeAll)
        {
            std::vector<GridSegment> segmentation;

            if (bInvokeAll)
            {
                // If invocation is done on all devices, segmentation is simple
                GridSegment gs;                
                gs.offset = 0;
                gs.block_dims = task.block_size;
                gs.total_grid_dims = task.grid_size;

                // Segment by output container: Override grid dims and blocks according to container
                if (task.segmentation_output_index >= 0)
                {
                    IDatum *container = task.outputs[task.segmentation_output_index].datum;
                    ISegmenter *segmenter = task.outputs[task.segmentation_output_index].segmenter.get();

                    unsigned int dim_three_and_beyond = 1;
                    for (unsigned int i = 2; i < container->GetDimensions(); ++i)
                        dim_three_and_beyond *= (unsigned int)container->GetDimension(i);

                    // Compute ideal grid dimensions for output container
                    gs.total_grid_dims = dim3(::maps::RoundUp((unsigned int)container->GetDimension(0) / segmenter->ItemsPerThread(0), task.block_size.x),
                                              ::maps::RoundUp((unsigned int)container->GetDimension(1) / segmenter->ItemsPerThread(1), task.block_size.y),
                                              ::maps::RoundUp(dim_three_and_beyond                     / segmenter->ItemsPerThread(2), task.block_size.z));
                    // Avoids zeros in dimensions where container is undefined
                    gs.total_grid_dims.x = std::max(1U, gs.total_grid_dims.x);
                    gs.total_grid_dims.y = std::max(1U, gs.total_grid_dims.y);
                    gs.total_grid_dims.z = std::max(1U, gs.total_grid_dims.z);
                }

                gs.blocks = gs.total_grid_dims.x * gs.total_grid_dims.y * gs.total_grid_dims.z;
#ifndef NDEBUG
                if (gs.blocks == 0)
                {
                    printf("SANITY CHECK FAILED: Zero blocks requested!\n");
                    return;
                }
#endif


                for (unsigned int i = 0; i < numGPUs; ++i)
                    segmentation.push_back(gs);

                task.segmentation = segmentation;
                return;
            }

            // Decide how the individual task is to be segmented
            if (task.segmentation_output_index < 0)
            {
                // Segment by blocks: Create a virtual environment of [ceil((x*y*z)/devs), 1, 1] grid size, only
                // to be modified back to emulate the real grid inside the kernel.
                unsigned int launch_grid_size = (task.grid_size.x * task.grid_size.y * task.grid_size.z);
                unsigned int segs = launch_grid_size / numGPUs;
                unsigned int remainder = segs + (launch_grid_size % numGPUs);

                GridSegment gs;                
                gs.block_dims = task.block_size;
                gs.total_grid_dims = task.grid_size;

                // Perform an even segmentation
                for (unsigned int i = 0; i < numGPUs; ++i)
                {
                    gs.offset = i * segs;
                    gs.blocks = (i == numGPUs - 1) ? remainder : segs;
#ifndef NDEBUG
                    if (gs.blocks == 0)
                    {
                        printf("SANITY CHECK FAILED: Zero blocks requested!\n");
                        return;
                    }
#endif
                    segmentation.push_back(gs);
                }
            }
            else
            {
                // Segment by output container: Divide output container size by last dimension (so that chunks are contiguous)
                IDatum *container = task.outputs[task.segmentation_output_index].datum;
                ISegmenter *segmenter = task.outputs[task.segmentation_output_index].segmenter.get();

                unsigned int dim_three_and_beyond = 1;
                for (unsigned int i = 2; i < container->GetDimensions(); ++i)
                    dim_three_and_beyond *= (unsigned int)container->GetDimension(i);

                // Compute ideal grid dimensions for output container
                dim3 output_grid_dims = dim3(::maps::RoundUp((unsigned int)container->GetDimension(0) / segmenter->ItemsPerThread(0), task.block_size.x),
                                             ::maps::RoundUp((unsigned int)container->GetDimension(1) / segmenter->ItemsPerThread(1), task.block_size.y),
                                             ::maps::RoundUp(dim_three_and_beyond                     / segmenter->ItemsPerThread(2), task.block_size.z));
                // Avoids zeros in dimensions where container is undefined
                output_grid_dims.x = std::max(1U, output_grid_dims.x);
                output_grid_dims.y = std::max(1U, output_grid_dims.y);
                output_grid_dims.z = std::max(1U, output_grid_dims.z);

                unsigned int launch_grid_size = maps::RoundUp(output_grid_dims.x * output_grid_dims.y * output_grid_dims.z, numGPUs);

                int outdims = container->GetDimensions();

                // Get the contiguous dimension's size in blocks
                size_t cdim_blocks = maps::RoundUp((unsigned int)container->GetDimension(outdims - 1) / segmenter->ItemsPerThread(outdims - 1), 
                                                   GetContiguousBlockDimension(outdims, task.block_size));
                size_t segs = cdim_blocks / numGPUs;
                size_t remainder = segs + (cdim_blocks % numGPUs);

                unsigned int otherdims = (unsigned int)((output_grid_dims.x * output_grid_dims.y * output_grid_dims.z) / cdim_blocks);

                // Perform an uneven segmentation, based on the dimensionality of the buffer
                GridSegment gs;
                gs.block_dims = task.block_size;
                gs.total_grid_dims = output_grid_dims; // Overrides original grid dimensions

                for (unsigned int i = 0; i < numGPUs; ++i)
                {
                    gs.offset = i * (unsigned int)segs * otherdims;
                    gs.blocks = (unsigned int)((i == (numGPUs - 1)) ? remainder : segs) * otherdims;
#ifndef NDEBUG
                    if (gs.blocks == 0)
                    {
                        printf("SANITY CHECK FAILED: Zero blocks requested!\n");
                        return;
                    }
#endif
                    segmentation.push_back(gs);
                }
            }

            task.segmentation = segmentation;
        }

    }  // namespace multi
}  // namespace maps

#endif  // __MAPS_MULTI_TASK_CONSTRUCTOR_H
