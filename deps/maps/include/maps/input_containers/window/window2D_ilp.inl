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

#ifndef __MAPS_WINDOW2D_ILP_INL_
#define __MAPS_WINDOW2D_ILP_INL_

#include <cstdint>

// Don't include this file directly. 
// DEPRECATED: File will be removed in a future version.

namespace maps
{
    enum
    {
        MAPS_32C4_IPT = 4,
        MAPS_32C8_IPT = 8,
        MAPS_SM_AP = 1,
    };
 
////////////////////////////////////////////////////////////////////////////////

    // Specialization for small window aprons that does not use shared memory
    template<int BLOCK_WIDTH, int BLOCK_HEIGHT>
    class Window<uint8_t, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 1, 4, 2, 1, WB_WRAP,
                 -1, GR_DISTINCT, true> : public IInputContainer
    {
        enum
        {
            ITEMS_PER_THREAD = MAPS_32C8_IPT,
            WINDOW_APRON = MAPS_SM_AP,
            NUM_ITEMS_X  = MAPS_32C4_IPT,
            NUM_ITEMS_Y = MAPS_32C8_IPT / MAPS_32C4_IPT,
            DIMS = 2,
        };

        // Static assertions on block width, height and apron size
        MAPS_STATIC_ASSERT(BLOCK_WIDTH > 0, "Block width must be positive");
        MAPS_STATIC_ASSERT(BLOCK_HEIGHT > 0, "Block height must be positive");
        MAPS_STATIC_ASSERT(WINDOW_APRON > 0, "Window apron must be positive");
        MAPS_STATIC_ASSERT(BLOCK_WIDTH >= 2 * WINDOW_APRON, 
                           "Block width must be at least twice the size of the "
                           "apron");
        MAPS_STATIC_ASSERT(BLOCK_HEIGHT >= 2 * WINDOW_APRON, 
                           "Block height must be at least twice the size of the"
                           " apron");

        // Static assertions on items per thread
        MAPS_STATIC_ASSERT(WINDOW_APRON <= ITEMS_PER_THREAD, 
                           "The window apron must be less or equal to the "
                           "items per thread");

        enum
        {
            XSHARED32 = BLOCK_WIDTH + 2 * WINDOW_APRON,
            YSHARED = BLOCK_HEIGHT + WINDOW_APRON * 2,
            WIND_WIDTH = 2 * WINDOW_APRON + 1,
            XOFFSET = NUM_ITEMS_X - WINDOW_APRON,
            X_STRIDE = WIND_WIDTH * NUM_ITEMS_X,
        };

    public:
        enum
        {
            ELEMENTS = (WINDOW_APRON * 2 + 1) * (WINDOW_APRON * 2 + 1),
            SYNC_AFTER_INIT = false,
        };

        struct SharedData
        {
            // No data loaded onto shared memory
        };

        int m_dimensions[DIMS];
        size_t m_stride;

        bool m_containsApron; /// Signals the container to start from y + APRON
        int m_gridWidth; /// Grid width, in blocks
        int block_offset;

    protected:
        uchar4 m_data[2 * WINDOW_APRON + NUM_ITEMS_Y][WIND_WIDTH];

    public:

        /// @brief Internal Window 2D iterator class
        class iterator : public std::iterator<std::input_iterator_tag, uint8_t>
        {
        protected:
            unsigned int m_pos;
            int m_id;
            const uint8_t *m_lParentData;
            int m_initialOffset;
   
            __device__  __forceinline__ void next()
            {
                ++m_id;
                m_pos = m_initialOffset + (m_id % WIND_WIDTH) + 
                    ((m_id / WIND_WIDTH) * X_STRIDE);
            }
        public:
            __device__ iterator(unsigned int pos, const Window *parent)
            {
                m_pos = pos;
                m_lParentData = (uint8_t *)parent->m_data;
                m_id = 0;
                m_initialOffset = pos;
            }

            __device__ iterator(const iterator& other)
            {
                m_pos = other.m_pos;
                m_lParentData = other.m_lParentData;
                m_id = other.m_id;
                m_initialOffset = other.m_initialOffset;
            }

            __device__  __forceinline__ void operator=(const iterator &a)
            {
                m_id = a.m_id;
                m_pos = a.m_pos;
                m_initialOffset = a.m_initialOffset;
                m_lParentData = a.m_lParentData;
            }

            __device__ __forceinline__ int index() const
            {
                return m_id;
            }

            __device__ __forceinline__ const uint8_t& operator*() const
            {
                return m_lParentData[m_pos];
            }

            __device__  __forceinline__ iterator& operator++() // Prefix
            {
                next();
                return *this;
            }

            __device__  __forceinline__ iterator operator++(int) // Postfix
            {
                iterator temp(*this);
                next();
                return temp;
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

        // Every iterator is a const iterator
        typedef iterator const_iterator;

        __host__ __device__ Window() : m_containsApron(false), m_gridWidth(0), 
            m_stride(0) 
        {
            m_dimensions[0] = m_dimensions[1] = 0;
        }


        /**
         * @brief Initializes only the container's shared data.
         * @param[in] sdata Reference to the shared data
         */
        __device__ __forceinline__ void init_async(SharedData& sdata)
        {
            m_stride /= NUM_ITEMS_X;
            m_dimensions[0] /= NUM_ITEMS_X;
   
            // Load data to shared memory

            // Calculate begin and end indices
            int y = 0;
            int x = 0;

            if (m_gridWidth == 0)
            {
                y = (BLOCK_HEIGHT * blockIdx.y + threadIdx.y) * NUM_ITEMS_Y;
                x = BLOCK_WIDTH * blockIdx.x + threadIdx.x;
            }
            else // if (m_gridWidth > 0)
            {
#if __CUDA_ARCH__
                unsigned int __realBlockIdx;    
                asm("mov.b32   %0, %ctaid.x;" : "=r"(__realBlockIdx));
                __realBlockIdx += block_offset;
    

                y = (BLOCK_HEIGHT * (__realBlockIdx / m_gridWidth) + 
                     threadIdx.y) * NUM_ITEMS_Y;
                x = BLOCK_WIDTH  * (__realBlockIdx % m_gridWidth) + threadIdx.x;
#endif
            }

            if (m_containsApron)
                y += WINDOW_APRON;

            #pragma unroll
            for (int y_ind = 0; y_ind < 2 + NUM_ITEMS_Y; ++y_ind)
            {
                m_data[y_ind][0] = ((uchar4*)m_ptr)[m_stride*Wrap((y + y_ind - 1), m_dimensions[1]) + Wrap((x - 1), m_dimensions[0])];
                m_data[y_ind][1] = ((uchar4*)m_ptr)[m_stride*Wrap((y + y_ind - 1), m_dimensions[1]) + Wrap((x), m_dimensions[0])];
                m_data[y_ind][2] = ((uchar4*)m_ptr)[m_stride*Wrap((y + y_ind - 1), m_dimensions[1]) + Wrap((x + 1), m_dimensions[0])];
            }
        }

        /**
         * @brief Initializes only the container's shared data.
         * @param[in] sdata Reference to the shared data
         */
        __device__ __forceinline__ void init(SharedData& sdata)
        {
            init_async(sdata);
            // No need to synchronize
        }

        __device__ __forceinline__ void init_async_postsync()
        {
        }

        __device__ __forceinline__ const uint8_t& aligned_at(
            IOutputContainerIterator& oiter, int x, int y) const
        {
            return ((uint8_t *)m_data)[(XOFFSET + (oiter.m_pos % NUM_ITEMS_X) +
                x) + ((oiter.m_pos / NUM_ITEMS_X) + y) * X_STRIDE];
        }

        /**
         * @brief Creates a thread-level iterator that points to the beginning 
         * of the current chunk.
         * @return Thread-level iterator.
         */
        __device__ __forceinline__ iterator align(
            IOutputContainerIterator& oiter) const
        {
            return iterator((XOFFSET + (oiter.m_pos % NUM_ITEMS_X)) + 
                (oiter.m_pos / NUM_ITEMS_X) * X_STRIDE, this);
        }

        /**
         * @brief Creates a thread-level iterator that points to the end of the
         * current chunk.
         * @return Thread-level iterator.
         */
        __device__ __forceinline__ iterator end_aligned(
            IOutputContainerIterator& oiter) const
        {
            return iterator((XOFFSET + (oiter.m_pos % NUM_ITEMS_X)) + 
                (WIND_WIDTH + (oiter.m_pos / NUM_ITEMS_X)) * 
                (WIND_WIDTH * NUM_ITEMS_X), this);
        }

        // VERSIONS WITHOUT ILP BELOW:
        /**
         * @brief Creates a thread-level iterator that points to the beginning 
         * of the current chunk.
         * @return Thread-level iterator.
         */
        __device__ __forceinline__ iterator begin() const
        {
            return iterator(XOFFSET, this);
        }

        /**
         * @brief Creates a thread-level iterator that points to the end of the 
         * current chunk.
         * @return Thread-level iterator.
         */
        __device__ __forceinline__ iterator end() const
        {
            return iterator(XOFFSET + (WIND_WIDTH)* (WIND_WIDTH * NUM_ITEMS_X),
                            this);
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

    };
    
}  // namespace maps

#endif  // __MAPS_WINDOW2D_ILP_INL_
