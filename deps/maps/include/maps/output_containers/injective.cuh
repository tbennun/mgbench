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

#ifndef __MAPS_INJECTIVE_OUTPUT_CUH_
#define __MAPS_INJECTIVE_OUTPUT_CUH_

#include "../internal/common.cuh"

namespace maps
{
    template <typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
              int BLOCK_DEPTH, int ITEMS_PER_THREAD = 1, 
              int ROWS_PER_THREAD = 1, ILPScheme ILP_SCHEME = ILP_CONTINUOUS, 
              bool MULTI_GPU = true>
    struct StructuredInjectiveOutput;

    // Type aliases    
    template <typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
              int BLOCK_DEPTH, int ITEMS_PER_THREAD = 1, 
              int ROWS_PER_THREAD = 1, ILPScheme ILP_SCHEME = ILP_CONTINUOUS>
    using StructuredInjectiveSingleGPU = 
        StructuredInjectiveOutput<T, DIMS, BLOCK_WIDTH, BLOCK_HEIGHT,
                                  BLOCK_DEPTH, ITEMS_PER_THREAD, 
                                  ROWS_PER_THREAD, ILP_SCHEME, false>;

    template <typename T, int BLOCK_WIDTH, int ITEMS_PER_THREAD = 1, 
              int BLOCK_HEIGHT = 1, int BLOCK_DEPTH = 1,
              ILPScheme ILP_SCHEME = ILP_CONTINUOUS, bool MULTI_GPU = true>
    using StructuredInjective1D = 
        StructuredInjectiveOutput<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT,
                                  BLOCK_DEPTH, ITEMS_PER_THREAD, 1, ILP_SCHEME,
                                  MULTI_GPU>;

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
              int ITEMS_PER_THREAD = 1, int ROWS_PER_THREAD = 1, 
              int BLOCK_DEPTH = 1, ILPScheme ILP_SCHEME = ILP_CONTINUOUS, 
              bool MULTI_GPU = true>
    using StructuredInjective2D = 
        StructuredInjectiveOutput<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT,
                                  BLOCK_DEPTH, ITEMS_PER_THREAD, 
                                  ROWS_PER_THREAD, ILP_SCHEME,
                                  MULTI_GPU>;
    
    template <typename T, int BLOCK_WIDTH, int ITEMS_PER_THREAD = 1, 
              int BLOCK_HEIGHT = 1, int BLOCK_DEPTH = 1,
              ILPScheme ILP_SCHEME = ILP_CONTINUOUS>
    using StructuredInjective1DSingleGPU = 
        StructuredInjectiveOutput<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT,
                                  BLOCK_DEPTH, ITEMS_PER_THREAD, 1, ILP_SCHEME,
                                  false>;

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
              int ITEMS_PER_THREAD = 1, int ROWS_PER_THREAD = 1, 
              int BLOCK_DEPTH = 1, ILPScheme ILP_SCHEME = ILP_CONTINUOUS>
    using StructuredInjective2DSingleGPU = 
        StructuredInjectiveOutput<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT,
                                  BLOCK_DEPTH, ITEMS_PER_THREAD, 
                                  ROWS_PER_THREAD, ILP_SCHEME,
                                  false>;

    template <typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
              int BLOCK_DEPTH, int ITEMS_PER_THREAD, int ROWS_PER_THREAD, 
              ILPScheme ILP_SCHEME, bool MULTI_GPU>
    struct StructuredInjectiveOutput : public IOutputContainer
    {
        MAPS_STATIC_ASSERT(DIMS > 0, "Dimensions must be positive.");
        MAPS_STATIC_ASSERT(DIMS <= 3, "Dimensions must not exceed 3.");

        int m_dimensions[3];
        size_t m_stride;

        dim3 grid_dims; ///< Actual grid dimensions, for block index computation
        uint3 blockId;

        enum
        {
            ELEMENTS = ITEMS_PER_THREAD * ROWS_PER_THREAD,
            SYNC_AFTER_INIT = false,
            DIRECT_TO_GLOBAL = false,
        };

        struct SharedData
        {
        };

        T m_regs[ELEMENTS];

        StructuredInjectiveOutput() : m_stride(0), grid_dims(), blockId() {}

        // Disallow copy
        __host__ __device__ void operator=(
            const StructuredInjectiveOutput&) = delete;

        // [[deprecated]]
        __device__ __forceinline__ void init()
        {
            SharedData sdata;
            init_async(sdata);
        }

        __device__ __forceinline__ void init_async(SharedData& sdata)
        {
            if (MULTI_GPU)
            {
                if (grid_dims.x + grid_dims.y + grid_dims.z > 3)
                {
                    unsigned int __realBlockIdx;
                    asm("mov.b32   %0, %ctaid.x;" : "=r"(__realBlockIdx));

                    blockId.x = __realBlockIdx % grid_dims.x;
                    blockId.y = (__realBlockIdx / grid_dims.x) % grid_dims.y;
                    blockId.z = ((__realBlockIdx / grid_dims.x) / grid_dims.y);
                }
                else
                {
                    blockId = blockIdx;
                }
            }
        }

        __device__ __forceinline__ void init(SharedData& sdata)
        {
            init_async(sdata);
        }

        __device__ __forceinline__ void init_async_postsync()
        {
        }

        __device__ __forceinline__ int Items()
        {
            switch (DIMS)
            {
            default:
                return 0;
            case 1:
            {
                int x = (BLOCK_WIDTH * (MULTI_GPU ? blockId.x : blockIdx.x) + threadIdx.x) * ITEMS_PER_THREAD;
                if (x >= m_dimensions[0])
                    return 0;
                
                return ITEMS_PER_THREAD * ROWS_PER_THREAD;
            }
            case 2:
            {
                int x = (BLOCK_WIDTH * (MULTI_GPU ? blockId.x : blockIdx.x) + threadIdx.x) * ITEMS_PER_THREAD;
                int y = (BLOCK_HEIGHT * (MULTI_GPU ? blockId.y : blockIdx.y) + threadIdx.y) * ROWS_PER_THREAD;
                if (x >= m_dimensions[0] || y >= m_dimensions[1])
                    return 0;
                
                return ITEMS_PER_THREAD * ROWS_PER_THREAD;
            }
            case 3:
            {
                int x = (BLOCK_WIDTH  * (MULTI_GPU ? blockId.x : blockIdx.x) + threadIdx.x) * ITEMS_PER_THREAD;
                int y = (BLOCK_HEIGHT * (MULTI_GPU ? blockId.y : blockIdx.y) + threadIdx.y) * ROWS_PER_THREAD;
                int z = BLOCK_DEPTH   * (MULTI_GPU ? blockId.z : blockIdx.z) + threadIdx.z;
                if (x >= m_dimensions[0] || y >= m_dimensions[1] || z >= m_dimensions[2])
                    return 0;
                
                return ITEMS_PER_THREAD * ROWS_PER_THREAD;
            }
            }
        }
        
        __device__ __forceinline__ void commit()
        {
            if (!DIRECT_TO_GLOBAL)
            {
                int xoff = (BLOCK_WIDTH  * (MULTI_GPU ? blockId.x : blockIdx.x)) * ITEMS_PER_THREAD;
                int yoff = DIMS < 2 ? 0 : (BLOCK_HEIGHT * (MULTI_GPU ? blockId.y : blockIdx.y)) * ROWS_PER_THREAD;
                int zoff = DIMS < 3 ? 0 : (BLOCK_DEPTH   * (MULTI_GPU ? blockId.z : blockIdx.z));

                // Commit register array to global memory
                ArrayToGlobal<T, DIMS, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH,
                              ITEMS_PER_THREAD, ROWS_PER_THREAD, 1>::Write(
                    m_regs, m_dimensions, m_stride, xoff, yoff, zoff, (T *)m_ptr);
            }
        }

        class iterator : public IOutputContainerIterator, 
                         public std::iterator<std::output_iterator_tag, T>
        {
        protected:
            typedef StructuredInjectiveOutput<T, DIMS, BLOCK_WIDTH, 
                                              BLOCK_HEIGHT, BLOCK_DEPTH, 
                                              ITEMS_PER_THREAD, ROWS_PER_THREAD,
                                              ILP_SCHEME, MULTI_GPU> Parent;

            Parent& m_parentData;
            int m_offset;

        public:
            __device__ iterator(Parent& parentData, int offset, int pos) :
                m_parentData(parentData), m_offset(offset)
            {
                m_pos = pos;
            }
            __device__ iterator(const iterator& other) : 
                m_parentData(other.m_parentData), m_offset(other.m_offset)
            {
                m_pos = other.m_pos;
            }

            __device__ __forceinline__ int index() const { return m_pos; }
            __device__ __forceinline__ T& operator*() 
            {
                if (DIRECT_TO_GLOBAL)
                    return *((T*)m_parentData.m_ptr + m_offset +
                             (DIMS == 1 ? m_pos : 
                              ((m_pos % ITEMS_PER_THREAD) + m_parentData.m_stride * (m_pos / ITEMS_PER_THREAD))));
                else
                    return m_parentData.m_regs[m_pos]; 
            }
            __device__  __forceinline__ iterator& operator++() // Prefix
            {
                ++m_pos;
                return *this;
            }
            __device__  __forceinline__ bool operator==(const iterator &a) const
            {
                return m_pos == a.m_pos;
            }
            __device__  __forceinline__ bool operator!=(const iterator &a) const
            {
                return m_pos != a.m_pos;
            }
        };

        __device__ __forceinline__ iterator begin()
        {
            const int x = (BLOCK_WIDTH  * (MULTI_GPU ? blockId.x : blockIdx.x) + threadIdx.x) * ITEMS_PER_THREAD;
            const int y = (DIMS < 2) ? 0 : (m_stride * (BLOCK_HEIGHT * (MULTI_GPU ? blockId.y : blockIdx.y) + threadIdx.y)) * ROWS_PER_THREAD;
            const int z = (DIMS < 3) ? 0 : (m_stride * m_dimensions[1] * (BLOCK_DEPTH  * (MULTI_GPU ? blockId.z : blockIdx.z) + threadIdx.z));
            
            return iterator(*this, DIRECT_TO_GLOBAL ? (x+y+z) : 0, 0);
        }

        __device__ __forceinline__ iterator end()
        {
            const int x = (BLOCK_WIDTH  * (MULTI_GPU ? blockId.x : blockIdx.x) + threadIdx.x) * ITEMS_PER_THREAD;
            const int y = (DIMS < 2) ? 0 : (m_stride * (BLOCK_HEIGHT * (MULTI_GPU ? blockId.y : blockIdx.y) + threadIdx.y)) * ROWS_PER_THREAD;
            const int z = (DIMS < 3) ? 0 : (m_stride * m_dimensions[1] * (BLOCK_DEPTH  * (MULTI_GPU ? blockId.z : blockIdx.z) + threadIdx.z));

            return iterator(*this, DIRECT_TO_GLOBAL ? (x + y + z) : 0, ELEMENTS);
        }
    };
}  // namespace maps

#endif  // __MAPS_INJECTIVE_OUTPUT_CUH_
