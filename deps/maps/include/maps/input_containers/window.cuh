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

#ifndef __MAPS_WINDOW_CUH_
#define __MAPS_WINDOW_CUH_

#include "../internal/common.h"
#include "internal/io_common.cuh"

namespace maps
{
    template<typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
             int BLOCK_DEPTH, int WINDOW_APRON, int IPX = 1, int IPY = 1, 
             int IPZ = 1, BorderBehavior BORDERS = WB_NOCHECKS, 
             int TEXTURE_UID = -1, GlobalReadScheme GRS = GR_DISTINCT, 
             bool MULTI_GPU = true>
    class Window;

    template<typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
             int BLOCK_DEPTH, int WINDOW_APRON, int IPX, int IPY, int IPZ, 
             BorderBehavior BORDERS, int TEXTURE_UID, GlobalReadScheme GRS, 
             bool USE_REGISTERS, int XSTRIDE, bool MULTI_GPU>
    class WindowIterator;

    // Template aliases for ease of use
    template<typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
             int BLOCK_DEPTH, int WINDOW_APRON, int IPX = 1, int IPY = 1, 
             int IPZ = 1, BorderBehavior BORDERS = WB_NOCHECKS, 
             int TEXTURE_UID = -1, GlobalReadScheme GRS = GR_DISTINCT>
    using WindowSingleGPU = Window<T, DIMS, BLOCK_WIDTH, BLOCK_HEIGHT, 
                                   BLOCK_DEPTH, WINDOW_APRON, IPX, IPY, IPZ, 
                                   BORDERS, TEXTURE_UID, GRS, false>;

    template<typename T, int BLOCK_WIDTH, int WINDOW_APRON, int IPX = 1, 
             BorderBehavior BORDERS = WB_NOCHECKS, int BLOCK_HEIGHT = 1, 
             int BLOCK_DEPTH = 1, int TEXTURE_UID = -1, 
             GlobalReadScheme GRS = GR_DISTINCT, bool MULTI_GPU = true>
    using Window1D = Window<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                            WINDOW_APRON, IPX, 1, 1, BORDERS, TEXTURE_UID, GRS,
                            MULTI_GPU>;
    template<typename T, int BLOCK_WIDTH, int WINDOW_APRON, int IPX = 1, 
             BorderBehavior BORDERS = WB_NOCHECKS, int BLOCK_HEIGHT = 1, 
             int BLOCK_DEPTH = 1, int TEXTURE_UID = -1, 
             GlobalReadScheme GRS = GR_DISTINCT>
    using Window1DSingleGPU = Window<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 
                                     BLOCK_DEPTH, WINDOW_APRON, IPX, 1, 1, 
                                     BORDERS, TEXTURE_UID, GRS, false>;

    template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int WINDOW_APRON, 
             BorderBehavior BORDERS = WB_NOCHECKS, int IPX = 1, int IPY = 1, 
             int BLOCK_DEPTH = 1, int TEXTURE_UID = -1, 
             GlobalReadScheme GRS = GR_DISTINCT, bool MULTI_GPU = true>
    using Window2D = Window<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                            WINDOW_APRON, IPX, IPY, 1, BORDERS, TEXTURE_UID, 
                            GRS, MULTI_GPU>;
    template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int WINDOW_APRON,
             BorderBehavior BORDERS = WB_NOCHECKS, int IPX = 1, int IPY = 1, 
             int BLOCK_DEPTH = 1, int TEXTURE_UID = -1, 
             GlobalReadScheme GRS = GR_DISTINCT>
    using Window2DSingleGPU = Window<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 
                                     BLOCK_DEPTH, WINDOW_APRON, IPX, IPY, 1, 
                                     BORDERS, TEXTURE_UID, GRS, false>;
    ///////////////////////////////////


    template<typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
             int BLOCK_DEPTH, int WINDOW_APRON, int IPX, int IPY, int IPZ, 
             BorderBehavior BORDERS, int TEXTURE_UID, GlobalReadScheme GRS,
             bool MULTI_GPU>
    class Window : public IInputContainer
    {
        MAPS_STATIC_ASSERT((DIMS >= 1 && DIMS <= 3), 
                           "Only the {1D,2D,3D} Window patterns are supported");
        MAPS_STATIC_ASSERT(BLOCK_WIDTH > 0, "Block width must be positive");
        MAPS_STATIC_ASSERT(BLOCK_HEIGHT > 0, "Block height must be positive");
        MAPS_STATIC_ASSERT(BLOCK_DEPTH > 0, "Block depth must be positive");
        MAPS_STATIC_ASSERT(WINDOW_APRON >= 0, 
                           "Window apron must be non-negative");
        MAPS_STATIC_ASSERT(IPX > 0, "Items per thread must be positive");

        enum
        {
            WIND_WIDTH = WINDOW_APRON * 2 + 1,

            SHARED_WIDTH = IPX * (BLOCK_WIDTH + 2 * WINDOW_APRON),
            SHARED_HEIGHT = ((DIMS < 2) ? 1 : (IPY * (BLOCK_HEIGHT + 2 * WINDOW_APRON))),
            SHARED_DEPTH  = ((DIMS < 3) ? 1 : (IPZ * (BLOCK_DEPTH + 2 * WINDOW_APRON))),

            // If this value is 1, skip shared memory and read directly to 
            // registers
            // TODO(later): More cases, such as 3x3 window
            USE_REGISTERS = (WINDOW_APRON == 0) ? 1 : 0,

            TOTAL_SHARED = USE_REGISTERS ? 1 : 
                           (SHARED_WIDTH * SHARED_HEIGHT * SHARED_DEPTH),
            TOTAL_REGISTERS = USE_REGISTERS ? 
                              ((WINDOW_APRON * 2 + IPX) * 
                               (WINDOW_APRON * 2 + IPY) * 
                               (WINDOW_APRON * 2 + IPZ)) : 1,

            XSTRIDE = USE_REGISTERS ? (WINDOW_APRON * 2 + IPX) : SHARED_WIDTH,
        };
        
        __device__ __forceinline__ unsigned int GetBeginIndex() const
        {
            switch (DIMS)
            {
            default:
            case 1:
                return threadIdx.x * IPX;
            case 2:
                return threadIdx.x * IPX + threadIdx.y * IPY * SHARED_WIDTH;
            case 3:
                return threadIdx.x * IPX + threadIdx.y * IPY * SHARED_WIDTH +
                       threadIdx.z * IPZ * SHARED_WIDTH * SHARED_HEIGHT;
            }
        }

        __device__ __forceinline__ void ToSharedArray(const uint3& offset)
        {
            // Load data to shared memory
            GlobalToShared<T, DIMS, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                           SHARED_WIDTH, SHARED_WIDTH, SHARED_HEIGHT, 
                           SHARED_DEPTH, true, BORDERS, 
                           ((TEXTURE_UID >= 0) ? GR_TEXTURE : GRS), 
                TEXTURE_UID>::Read((T *)m_ptr, m_dimensions, m_stride, 
                                   offset.x, offset.y, offset.z,
                                   m_sdata, 0, 1);
        }

        __device__ __forceinline__ void ToRegArray(const uint3& offset)
        {
            // Load data to internal registers                
            GlobalToArray<T, DIMS, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                          USE_REGISTERS ? (WINDOW_APRON * 2 + IPX) : 1,
                          USE_REGISTERS ? (WINDOW_APRON * 2 + IPY) : 1, 
                          USE_REGISTERS ? (WINDOW_APRON * 2 + IPZ) : 1,
                          BORDERS, ((TEXTURE_UID >= 0) ? GR_TEXTURE : GRS), 
                TEXTURE_UID>::Read((T *)m_ptr, m_dimensions, m_stride, 
                                   offset.x + IPX * threadIdx.x, 
                                   offset.y + IPY * threadIdx.y, offset.z,
                                   m_regs, 0, 1);
        }

        // Avoiding "subscript out of range" warnings
        __device__ __forceinline__ const T& internal_at_1D(
            const int *index_array, int offx) const
        {
            const unsigned int OFFSETX = (USE_REGISTERS) ? 0 : threadIdx.x * IPX;

            if (USE_REGISTERS)
                return m_regs[(OFFSETX + index_array[0] + WINDOW_APRON + offx)];
            else
                return m_sdata[(OFFSETX + index_array[0] + WINDOW_APRON + offx)];
        }

        __device__ __forceinline__ const T& internal_at_2D(
            const int *index_array, int offx, int offy) const
        {
            const unsigned int OFFSETX = (USE_REGISTERS) ? 0 : threadIdx.x * IPX;
            const unsigned int OFFSETY = (USE_REGISTERS) ? 0 : threadIdx.y * IPY;

            
            if (USE_REGISTERS)
                return m_regs[(OFFSETX + index_array[0] + WINDOW_APRON + offx) +
                              (OFFSETY + index_array[1] + WINDOW_APRON + offy) *
                              SHARED_WIDTH];
            else
                return m_sdata[(OFFSETX + index_array[0] + 
                                WINDOW_APRON + offx) +
                  (OFFSETY + index_array[1] + WINDOW_APRON + offy) * 
                  SHARED_WIDTH];
        }

        __device__ __forceinline__ const T& internal_at_3D(
            const int *index_array, int offx, int offy, int offz) const
        {
            const unsigned int OFFSETX = (USE_REGISTERS) ? 0 : threadIdx.x * IPX;
            const unsigned int OFFSETY = (USE_REGISTERS) ? 0 : threadIdx.y * IPY;
            const unsigned int OFFSETZ = (USE_REGISTERS) ? 0 : threadIdx.z * IPZ;

            if (USE_REGISTERS)
                return m_regs[(OFFSETX + index_array[0] + WINDOW_APRON + offx) +
                              (OFFSETY + index_array[1] + WINDOW_APRON + offy) * SHARED_WIDTH +
                              (OFFSETZ + index_array[2] + WINDOW_APRON + offz) * SHARED_WIDTH * SHARED_HEIGHT];
            else
                return m_sdata[(OFFSETX + index_array[0] + WINDOW_APRON + offx) +
                               (OFFSETY + index_array[1] + WINDOW_APRON + offy) * SHARED_WIDTH +
                               (OFFSETZ + index_array[2] + WINDOW_APRON + offz) * SHARED_WIDTH * SHARED_HEIGHT];
        }

    public:
        enum
        {
            ELEMENTS = Power<WIND_WIDTH, DIMS>::value,
            SYNC_AFTER_INIT = !USE_REGISTERS,
        };

        // Shared data
        struct SharedData
        {
            /// The data loaded onto shared memory
            T m_sdata[TOTAL_SHARED];
        };

        // Data
        T m_regs[TOTAL_REGISTERS];        

        // Single-GPU parameters
        int m_dimensions[DIMS];
        int m_stride;
        T *m_sdata;

        // Multi-GPU parameters
        bool m_containsApron;
        int block_offset;
        int m_gridWidth;        

        // Define iterator classes
        typedef WindowIterator<T, DIMS, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH,
                               WINDOW_APRON, IPX, IPY, IPZ, BORDERS, 
                               TEXTURE_UID, GRS, USE_REGISTERS, XSTRIDE, MULTI_GPU> iterator;
        typedef iterator const_iterator;
  
        __host__ __device__ Window() : block_offset(0), m_gridWidth(0), 
                                       m_containsApron(false), m_stride(0)
        {
            m_dimensions[0] = m_dimensions[1] = 0;
        }

        /**
         * @brief Initializes the container.
         * @param[in] sdata SharedData structure (allocated on shared memory).
         */
        __device__ __forceinline__ void init(SharedData& sdata)
        {
            init_async(sdata);
            
            if (!USE_REGISTERS)
                __syncthreads();
        }        

        /**
         * @brief Initializes the container.
         * @param[in] sdata SharedData structure (allocated on shared memory).
         */
        __device__ __forceinline__ void init_async(SharedData& sdata)
        {
            uint3 offset;

            if (MULTI_GPU)
            {
                if (m_gridWidth > 0)
                {
                    unsigned int __realBlockIdx;
                    asm("mov.b32   %0, %ctaid.x;" : "=r"(__realBlockIdx));
                    __realBlockIdx += block_offset;

                    offset.x = IPX * BLOCK_WIDTH * 
                        (__realBlockIdx % m_gridWidth) - WINDOW_APRON;

                    if (DIMS == 2)
                    {
                        if (m_containsApron)
                            offset.y = IPY * BLOCK_HEIGHT * 
                                (__realBlockIdx / m_gridWidth);
                        else
                            offset.y = IPY * BLOCK_HEIGHT * 
                                (__realBlockIdx / m_gridWidth) - WINDOW_APRON;
                    }
                    else if (DIMS > 2)
                        offset.y = IPY * BLOCK_HEIGHT * 
                            (__realBlockIdx / m_gridWidth) - WINDOW_APRON;
                    else if (DIMS < 2)
                        offset.y = 0;

                    // TODO(later): 3D windows
                    offset.z = 0;
                    //offset.z = IPZ * BLOCK_DEPTH  * blockIdx.z - WINDOW_APRON;
                }
                else
                {
                    offset.x = IPX * BLOCK_WIDTH  * blockIdx.x - WINDOW_APRON;
                    offset.y = (DIMS < 2) ? 0 : 
                        (IPY * BLOCK_HEIGHT * blockIdx.y - WINDOW_APRON);
                    offset.z = (DIMS < 3) ? 0 : 
                        (IPZ * BLOCK_DEPTH  * blockIdx.z - WINDOW_APRON);
                }
            }
            else
            {
                offset.x = IPX * BLOCK_WIDTH  * blockIdx.x - WINDOW_APRON;
                offset.y = (DIMS < 2) ? 0 : 
                    (IPY * BLOCK_HEIGHT * blockIdx.y - WINDOW_APRON);
                offset.z = (DIMS < 3) ? 0 :
                    (IPZ * BLOCK_DEPTH  * blockIdx.z - WINDOW_APRON);
            }

            if (!USE_REGISTERS)
            {
                m_sdata = sdata.m_sdata;       
            }
            else
            {
                // NOTE: Turns out that the following line 
                // (pointers to local arrays) compiles HORRIBLY, even with 
                // inlining.
                //m_sdata = m_regs;
            }

            // Load the global data either to shared memory or registers
            if (USE_REGISTERS)
                ToRegArray(offset);
            else
                ToSharedArray(offset);            
        }

        __device__ __forceinline__ void init_async_postsync()
        {
        }

        /**
         * @brief Returns the value at the thread-relative index in the range 
         * [-APRON, APRON].
         */
        template<typename... Index>
        __device__ __forceinline__ const T& at(Index... indices) const
        {
            static_assert(sizeof...(indices) == DIMS, 
                          "Input must agree with container dimensions");
            int index_array[] = { (int)indices... };

            switch (DIMS)
            {
            default:
            case 1:
                return internal_at_1D(index_array, 0);
            case 2:
                return internal_at_2D(index_array, 0, 0);
            case 3:
                return internal_at_3D(index_array, 0, 0, 0);
            }
        }

        /**
         * @brief Returns the value at the thread-relative index in the range 
         * [-APRON, APRON].
         */
        template<typename... Index>
        __device__ __forceinline__ const T& aligned_at(
            IOutputContainerIterator& oiter, Index... indices) const
        {
            static_assert(sizeof...(indices) == DIMS, 
                          "Input must agree with container dimensions");
            int index_array[] = { (int)indices... };

            switch (DIMS)
            {
            default:
            case 1:
                return internal_at_1D(index_array, oiter.m_pos);
            case 2:
                return internal_at_2D(index_array, oiter.m_pos % IPX, 
                                      oiter.m_pos / IPX);
            case 3:
                return internal_at_3D(index_array, oiter.m_pos % IPX, 
                                      (oiter.m_pos / IPX) % IPY, 
                                      (oiter.m_pos / IPX) / IPY);
            }
        }

        /**
         * @brief Creates a thread-level iterator that points to the beginning 
         * of the current chunk.
         * @return Thread-level iterator.
         */
        __device__ __forceinline__ iterator begin() const
        {
            return iterator((USE_REGISTERS) ? 0 : GetBeginIndex(), *this);
        }

        /**
         * @brief Creates a thread-level iterator that points to the end of the
         * current chunk.
         * @return Thread-level iterator.
         */
        __device__ __forceinline__ iterator end() const
        {
            return iterator(((USE_REGISTERS) ? 0 : GetBeginIndex()) + ELEMENTS,
                            *this);
        }
  
        /**
         * @brief Creates a thread-level iterator that points to the beginning
         * of the current chunk.
         * @return Thread-level iterator.
         */
        __device__ __forceinline__ iterator align(
            IOutputContainerIterator& oiter) const
        {
            return iterator(((USE_REGISTERS) ? 0 : GetBeginIndex()) + 
                            oiter.m_pos % IPX + (oiter.m_pos / IPX) * XSTRIDE, *this);
        }

        /**
         * @brief Creates a thread-level iterator that points to the end of the
         * current chunk.
         * @return Thread-level iterator.
         */
        __device__ __forceinline__ iterator end_aligned(
            IOutputContainerIterator& oiter) const
        {
            return iterator(((USE_REGISTERS) ? 0 : GetBeginIndex()) + 
                            ELEMENTS + oiter.m_pos % IPX + (oiter.m_pos / IPX) * XSTRIDE, *this);
        }

        /**
         * @brief Progresses to process the next chunk (does nothing).
         */
        __device__ __forceinline__ void nextChunk() { }

        /**
         * Progresses to process the next chunk without calling __syncthreads()
         * (does nothing).
         * @note This is an advanced function that should be used carefully.
         */
        __device__ __forceinline__ void nextChunkAsync() { }

        /**
         * @brief Returns false if there are more chunks to process.
         */
        __device__ __forceinline__ bool isDone() { return true; }

        /**
        * @brief Returns the total number of chunks, or 0 for dynamic chunks.
        */
        __device__ __forceinline__ int chunks() { return 1; }
    };    
}

// Iterator implementation
#include "window/window_iterator.inl"

// ILP specializations
#include "window/window2D_ilp.inl"

#endif  // __MAPS_WINDOW_CUH_
