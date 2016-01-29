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

#ifndef __MAPS_MULTI_COMMON_H
#define __MAPS_MULTI_COMMON_H

#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <cuda_runtime.h> // for dim3

#include "../internal/common.h"
#include "datum.h"

namespace maps
{
    namespace multi
    {
        static void * const UNALLOCATED_BUFFER = (void *)0x00000010;

        enum OutputContainerType
        {
            OCT_STRUCTURED_INJECTIVE,   ///< Singular input-to-output mapping, ordered access

            // The following need aggregation operators

            OCT_UNSTRUCTURED_INJECTIVE, ///< Singular input-to-output mapping, unordered access
            OCT_REDUCTIVE_CONSTANT, ///< Reductive, constant size
            OCT_REDUCTIVE_DYNAMIC,  ///< Reductive, dynamic size (currently only supports append aggregation)
        };

        enum ArgumentType
        {
            AT_CONSTANT,
            AT_INPUT,
            AT_OUTPUT,
        };

        enum LocationState
        {
            LS_HOST,              ///< Data is still on host memory
            LS_DEVICE,            ///< Data resides entirely on one GPU, or several copies exist on multiple GPUs
            LS_SEGMENTED,         ///< Data is segmented between several GPUs
            LS_NEEDS_AGGREGATION, ///< Data resides on several GPUs, requires aggregation to be used
        };

        /// Tells the scheduler where the datum was last output from
        struct DatumLocation
        {
            LocationState state;
            std::vector< std::pair<unsigned int, DatumSegment> > entries;

            DatumLocation() : state(LS_HOST) {}
        };

        struct GridSegment
        {
            unsigned int offset;  ///< The offset of this segment (in the 1D space)
            unsigned int blocks;  ///< The amount of blocks in this segment (in the 1D space)
            dim3 total_grid_dims; ///< The total grid dimensions
            dim3 block_dims;      ///< Block dimensions

            GridSegment() : offset(0), blocks(0), total_grid_dims(), block_dims() {}
        };

        /// @brief Helper method to get the correct dimension from a dim3
        static inline unsigned int GetContiguousBlockDimension(unsigned int dims, dim3 block_dims)
        {
            switch (dims)
            {
            default:
                return 1;
            case 1:
                return block_dims.x;
            case 2:
                return block_dims.y;
            case 3:
                return block_dims.z;
            }
        }

        typedef std::vector<GridSegment> GridSegmentation;
        typedef std::vector< std::vector<DatumSegment> > ContainerSegmentation;

        typedef uint32_t taskHandle_t;
        typedef std::vector< std::map<IDatum *, Memory> > BufferList;
        typedef std::tuple<IDatum *, DatumSegment, DatumSegment> SegmentCopy;


        /// @brief Interface for segmenting input/output task data
        class ISegmenter
        {
        public:            
            // Returns a list of memory-contiguous regions (w.r.t. boundary conditions) per grid segment.
            virtual void Segment(const GridSegmentation& segments,
                                 ContainerSegmentation& out_data_segments) const = 0;

            virtual unsigned int ItemsPerThread(unsigned int dimension) const = 0;
        };

        /// @brief Interface for creating single-GPU containers
        class IContainerFactory
        {
        public:
            // Returns a pointer to an input container, for use as a kernel parameters
            virtual std::shared_ptr<::maps::IInputContainer> 
                CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const
            {
                return nullptr;
            }

            virtual std::shared_ptr<::maps::IOutputContainer>
                CreateOutputContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const
            {
                return nullptr;
            }
        };

        /// @brief Interface for reducing multiple single-GPU containers into one composed container
        class IContainerReducer
        {
        public:
            virtual std::shared_ptr<::maps::IContainerComposition>
                ComposeContainers(const std::vector<void*>& containers, const GridSegment& grid_seg) const
            {
                return nullptr;
            }
        };

        // Common data type aliases

        template <typename T>
        using Vector = Datum<T, 1>;        

        template <typename T>
        using Matrix = Datum<T, 2>;

        struct IAggregator
        {            
            virtual void AggregateToHost(IDatum *datum, const void *src, size_t stride_bytes,
                                         void *dst) const = 0;

            // TODO (later): Aggregate to device
        };

        struct TaskInput
        {
            IDatum *datum;
            std::shared_ptr<ISegmenter> segmenter;
            std::shared_ptr<IContainerFactory> container_factory;
            ContainerSegmentation segmentation;

            TaskInput(IDatum *dat, std::shared_ptr<ISegmenter> seg, std::shared_ptr<IContainerFactory> cf) : datum(dat), segmenter(seg),
                container_factory(cf), segmentation() {}
        };

        struct TaskOutput
        {
            IDatum *datum;
            std::shared_ptr<ISegmenter> segmenter;
            std::shared_ptr<IContainerFactory> container_factory;
            OutputContainerType type;
            std::shared_ptr<IAggregator> aggregator;
            ContainerSegmentation segmentation;
            
            TaskOutput(IDatum *dat, std::shared_ptr<ISegmenter> seg, std::shared_ptr<IContainerFactory> cf,
                       OutputContainerType ot, std::shared_ptr<IAggregator> agg) :
                datum(dat), segmenter(seg), container_factory(cf), type(ot), aggregator(agg), 
                segmentation() {}
        };

        struct Task
        {
            std::vector<TaskInput> inputs;
            std::vector<TaskOutput> outputs;
            std::vector< std::shared_ptr<void> > constants; ///< Constant inputs
            std::vector<ArgumentType> argument_ordering; ///< Specifies which arguments are input, output or constants (for invocation purposes)
            const void *kernel; ///< Pointer to kernel function
            dim3 grid_size;  ///< Total (virtual) grid dimensions
            dim3 block_size; ///< Block dimensions
            size_t dsmem; ///< Dynamic shared memory size

            // Support for patterns with multiple data structures
            // (e.g. Adjacency)
            std::vector< std::tuple< int, int, std::shared_ptr<maps::multi::IContainerReducer>> > reducers;

            // Scheduler-related information

            std::vector<cudaEvent_t> deps; ///< Events to wait to before copying data to segments
            std::vector<cudaEvent_t> events;   ///< So that other kernels and copies can wait for this task to finish
            std::vector<cudaStream_t> streams; ///< So that kernels can be executed independently

            /// Determines how to segment the task. Points to the index in "outputs" array to segment
            /// with, or -1 if segmentation is performed by blocks.
            int segmentation_output_index;

            /// Contains segmentation information and individual kernel launch parameters
            GridSegmentation segmentation;
            
            Task() : kernel(nullptr), dsmem(0), segmentation_output_index(-1), grid_size(), block_size() {}
        };

        /// @brief An unmodified function to be called instead of a kernel in a task. Kernels must run on the given stream,
        /// and cudaSetDevice must not be called during this routine (unless returned to original state).
        typedef bool(*routine_t)(void *context, int deviceIdx, cudaStream_t stream,
                                 const GridSegment& task_segment,
                                 const std::vector<void *>& parameters,
                                 const std::vector<DatumSegment>& container_segments,
                                 const std::vector<DatumSegment>& container_allocation);

        static __host__ __device__ __forceinline__ 
            void GlobalBlockIdx(uint3& blockId, unsigned int realBlock, 
                                const dim3& multigridDim)
        {
            // Compute the offset of the current device
            unsigned int blockOffset = blockId.x + realBlock;

            blockId.x =   blockOffset % multigridDim.x;
            blockId.y =  (blockOffset / multigridDim.x) % multigridDim.y;
            blockId.z = ((blockOffset / multigridDim.x) / multigridDim.y);
        }

        /**
        * @brief Returns a constant parameter in unmodified routines
        */
        template<typename T>
        static inline void GetConstantParameter(void *parameter, T& value)
        {
            memcpy(&value, parameter, sizeof(T));
        }

    }  // namespace multi
}  // namespace maps

#endif  // __MAPS_MULTI_COMMON_H
