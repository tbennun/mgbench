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

#ifndef __MAPS_BLOCK_CUH_
#define __MAPS_BLOCK_CUH_

#include "../internal/common.h"
#include "internal/io_common.cuh"

namespace maps
{
    template<typename T, int DIMS, int PRINCIPAL_DIM, int BLOCK_WIDTH, 
             int BLOCK_HEIGHT, int BLOCK_DEPTH, int IPX = 1, int IPY = 1, 
             int IPZ = 1, int CHUNKX = BLOCK_WIDTH, int CHUNKY = BLOCK_HEIGHT,
             int CHUNKZ = BLOCK_DEPTH, BorderBehavior BORDERS = WB_NOCHECKS, 
             int TEXTURE_UID = -1, GlobalReadScheme GRS = GR_DISTINCT, 
             bool MULTI_GPU = true>
    class Block;

    template<typename T, int DIMS, int PRINCIPAL_DIM, int BLOCK_WIDTH, 
             int BLOCK_HEIGHT, int BLOCK_DEPTH, int IPX = 1, int IPY = 1, 
             int IPZ = 1, int CHUNKX = BLOCK_WIDTH, int CHUNKY = BLOCK_HEIGHT,
             int CHUNKZ = BLOCK_DEPTH, BorderBehavior BORDERS = WB_NOCHECKS,
             int TEXTURE_UID = -1, GlobalReadScheme GRS = GR_DISTINCT, 
             bool MULTI_GPU = true>
    using BlockSingleGPU = Block<T, DIMS, PRINCIPAL_DIM, BLOCK_WIDTH, 
                                 BLOCK_HEIGHT, BLOCK_DEPTH, IPX, IPY, IPZ, 
                                 CHUNKX, CHUNKY, CHUNKZ,
                                 BORDERS, TEXTURE_UID, GRS, false>;


    template<typename T, int DIMS, int PRINCIPAL_DIM, int BLOCK_WIDTH, 
             int BLOCK_HEIGHT, int BLOCK_DEPTH, int IPX, int IPY, int IPZ, 
             int CHUNKX, int CHUNKY, int CHUNKZ, int XSTRIDE,
             BorderBehavior BORDERS, int TEXTURE_UID, GlobalReadScheme GRS, 
             bool MULTI_GPU>
    class BlockIterator;

    
    template<typename T, int DIMS, int PRINCIPAL_DIM, int BLOCK_WIDTH, 
             int BLOCK_HEIGHT, int BLOCK_DEPTH, int IPX, int IPY, int IPZ, 
             int CHUNKX, int CHUNKY, int CHUNKZ,
             BorderBehavior BORDERS, int TEXTURE_UID, GlobalReadScheme GRS, 
             bool MULTI_GPU>
    class Block : public IInputContainer
    {
        MAPS_STATIC_ASSERT((DIMS >= 1 && DIMS <= 2), 
                           "Only Block 1D and 2D patterns are supported");
        MAPS_STATIC_ASSERT(PRINCIPAL_DIM < DIMS, 
                           "Invalid principal dimension");
        MAPS_STATIC_ASSERT(PRINCIPAL_DIM >= 0, 
                           "Principal dimension must be non-negative");
        MAPS_STATIC_ASSERT(BLOCK_WIDTH > 0, "Block width must be positive");
        MAPS_STATIC_ASSERT(BLOCK_HEIGHT > 0, "Block height must be positive");
        MAPS_STATIC_ASSERT(BLOCK_DEPTH > 0, "Block depth must be positive");
        MAPS_STATIC_ASSERT(IPX > 0, "Items per thread must be positive");
        // TODO: More static assertions
        MAPS_STATIC_ASSERT(PRINCIPAL_DIM == 0 || BLOCK_WIDTH == CHUNKX, "Chunk X dimension must be equal to block width");
        MAPS_STATIC_ASSERT(DIMS < 2 || (PRINCIPAL_DIM == 1 || BLOCK_HEIGHT == CHUNKY), "Chunk Y dimension must be equal to block height");
        MAPS_STATIC_ASSERT(DIMS < 3 || (PRINCIPAL_DIM == 2 || BLOCK_DEPTH  == CHUNKZ), "Chunk Z dimension must be equal to block depth");

        enum
        {
            SHARED_WIDTH = (PRINCIPAL_DIM == 0) ? CHUNKX : CHUNKX * IPX,
            SHARED_HEIGHT = (PRINCIPAL_DIM == 1) ? CHUNKY : CHUNKY * IPY,
            SHARED_DEPTH = (PRINCIPAL_DIM == 2) ? CHUNKZ : CHUNKZ * IPZ,

#if __CUDA_ARCH__ < 300
            // For bank conflict-free transposed read to shared memory.
            SHARED_STRIDE = (DIMS == 2 && PRINCIPAL_DIM == 1) ? 
                            (SHARED_WIDTH + 1) : SHARED_WIDTH,
#else
            // 128-bit shared loads result in a better speed gain than bank 
            // conflicts reduce.
            SHARED_STRIDE = SHARED_WIDTH,
#endif

            TOTAL_SHARED = SHARED_STRIDE * SHARED_HEIGHT * SHARED_DEPTH,

            // OPTIMIZATION: Double-buffer shared memory (read next buffer 
            // while operating on current, conserves one __syncthreads() 
            // per chunk)
            USE_SMEM_DOUBLE_BUFFERING = false,
        };

    public:
        enum
        {
            PRINCIPAL_BLOCK_DIM = ((PRINCIPAL_DIM == 0) ? SHARED_WIDTH :
                                   ((PRINCIPAL_DIM == 1) ? SHARED_HEIGHT :
                                    SHARED_DEPTH)),

            ELEMENTS = PRINCIPAL_BLOCK_DIM,
            SYNC_AFTER_NEXTCHUNK = !USE_SMEM_DOUBLE_BUFFERING,
        };

        struct SharedData
        {
            /// The data loaded onto shared memory
            T m_sdata[USE_SMEM_DOUBLE_BUFFERING ? (2*TOTAL_SHARED) : 
                      TOTAL_SHARED];
        };

        int m_dimensions[DIMS];
        int m_stride;
        T *m_sdata;
        int m_blockInd;
        int m_blocks;

        // Multi-GPU parameters
        //int block_offset;
        uint3 blockId;
        dim3 grid_dims;

        // Define iterator classes
        typedef BlockIterator<T, DIMS, PRINCIPAL_DIM, BLOCK_WIDTH, 
                              BLOCK_HEIGHT, BLOCK_DEPTH, IPX, IPY, IPZ, 
                              CHUNKX, CHUNKY, CHUNKZ, SHARED_STRIDE,
                              BORDERS, TEXTURE_UID, GRS, MULTI_GPU> iterator;
        typedef iterator const_iterator;
       
        // Define block loaders
        typedef GlobalToShared<T, DIMS, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                               SHARED_WIDTH, SHARED_STRIDE, SHARED_HEIGHT, SHARED_DEPTH, false, BORDERS,
                               ((TEXTURE_UID >= 0) ? GR_TEXTURE : GRS), TEXTURE_UID> LoadSync;
        typedef GlobalToShared<T, DIMS, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                               SHARED_WIDTH, SHARED_STRIDE, SHARED_HEIGHT, SHARED_DEPTH, true, BORDERS,
                               ((TEXTURE_UID >= 0) ? GR_TEXTURE : GRS), TEXTURE_UID> LoadAsync;

        /**
         * @brief Initializes the container.
         * @param[in] sdata SharedData structure (allocated on shared memory).
         */
        __device__ __forceinline__ void init(SharedData& sdata)
        {
            init_async(sdata);
            
            __syncthreads();
            
            init_async_postsync();
        }

        /**
         * @brief Initializes the container without synchronizing. (Call 
         * init_async_postsync() after __syncthreads())
         * @param[in] sdata SharedData structure (allocated on shared memory).
         */
        __device__ __forceinline__ void init_async(SharedData& sdata)
        {
            m_blockInd = 0;
            m_blocks = maps::RoundUp(m_dimensions[PRINCIPAL_DIM], 
                                     PRINCIPAL_BLOCK_DIM);

            if (MULTI_GPU)
            {
                unsigned int __realBlockIdx;
                asm("mov.b32   %0, %ctaid.x;" : "=r"(__realBlockIdx));
                
                blockId.x = __realBlockIdx % grid_dims.x;
                blockId.y = (__realBlockIdx / grid_dims.x) % grid_dims.y;
                blockId.z = ((__realBlockIdx / grid_dims.x) / grid_dims.y);
            }
                      
            m_sdata = sdata.m_sdata;
            
            // TODO(later): Offset using block index (avoiding partition camping)?
            
            // Load data to shared memory
            LoadAsync::Read((T *)m_ptr, m_dimensions, m_stride,
                            (PRINCIPAL_DIM == 0) ? 0 : (SHARED_WIDTH  * (MULTI_GPU ? blockId.x : blockIdx.x)),
                            DIMS < 2 ? 0 : (PRINCIPAL_DIM == 1) ? 0 : (SHARED_HEIGHT * (MULTI_GPU ? blockId.y : blockIdx.y)),
                            DIMS < 3 ? 0 : (PRINCIPAL_DIM == 2) ? 0 : (SHARED_DEPTH  * (MULTI_GPU ? blockId.z : blockIdx.z)),
                            m_sdata, 0, m_blocks);
        }

        /**
         * @brief Finishes container asynchronous initialization after calling __syncthreads()
         */
        __device__ __forceinline__ void init_async_postsync()
        {
            // If double-buffered, start loading the next batch to shared already
            if (USE_SMEM_DOUBLE_BUFFERING)
            {
                LoadAsync::Read((T *)m_ptr, m_dimensions, m_stride,
                                (PRINCIPAL_DIM == 0) ? (SHARED_WIDTH  * 1) : (SHARED_WIDTH  * (MULTI_GPU ? blockId.x : blockIdx.x)),
                                DIMS < 2 ? 0 : (PRINCIPAL_DIM == 1) ? (SHARED_HEIGHT * 1) : (SHARED_HEIGHT * (MULTI_GPU ? blockId.y : blockIdx.y)),
                                DIMS < 3 ? 0 : (PRINCIPAL_DIM == 2) ? (SHARED_DEPTH  * 1) : (SHARED_DEPTH  * (MULTI_GPU ? blockId.z : blockIdx.z)),
                                m_sdata + TOTAL_SHARED, 1, m_blocks);
            }
        }


        /**
         * @brief Returns the value at the thread-relative index in the range [0, CHUNK_DIM).
         */
        template<typename... Index>
        __device__ __forceinline__ const T& at(Index... indices) const
        {
            static_assert(sizeof...(indices) == DIMS, 
                          "Input must agree with container dimensions");
            size_t index_array[] = { (size_t)indices... };
                
            const unsigned int OFFSETX = (PRINCIPAL_DIM == 0 ? 0 : threadIdx.x * IPX) + 
                                         ((USE_SMEM_DOUBLE_BUFFERING && (m_blockInd % 2 == 1)) ? TOTAL_SHARED : 0);
            const unsigned int OFFSETY = (PRINCIPAL_DIM == 1 ? 0 : threadIdx.y * IPY);
            const unsigned int OFFSETZ = (PRINCIPAL_DIM == 2 ? 0 : threadIdx.z * IPZ);

            switch (DIMS)
            {
            default:
            case 1:
                return m_sdata[(OFFSETX + index_array[0])];
            case 2:
                return m_sdata[(OFFSETX + index_array[0]) +
                                (OFFSETY + index_array[1]) * SHARED_STRIDE];
            case 3:
                return m_sdata[(OFFSETX + index_array[0]) + 
                                (OFFSETY + index_array[1]) * SHARED_STRIDE + 
                                (OFFSETZ + index_array[2]) * SHARED_STRIDE * SHARED_HEIGHT];
            }
        }

        /**
         * @brief Returns the value at the thread-relative index in the range [0, CHUNK_DIM).
         */
        template<typename... Index>
        __device__ __forceinline__ const T& aligned_at(IOutputContainerIterator& oiter, Index... indices) const
        {
            static_assert(sizeof...(indices) == DIMS,
                          "Input must agree with container dimensions");
            size_t index_array[] = { (size_t)indices... };

            const unsigned int OFFSETX = (PRINCIPAL_DIM == 0 ? 0 : threadIdx.x * IPX) + 
                                         ((USE_SMEM_DOUBLE_BUFFERING && (m_blockInd % 2 == 1)) ? TOTAL_SHARED : 0);
            const unsigned int OFFSETY = (PRINCIPAL_DIM == 1 ? 0 : threadIdx.y * IPY);
            const unsigned int OFFSETZ = (PRINCIPAL_DIM == 2 ? 0 : threadIdx.z * IPZ);
            
            switch (DIMS)
            {
            default:
            case 1:
                return m_sdata[(OFFSETX + index_array[0] + oiter.m_pos)];
            case 2:
                return m_sdata[(OFFSETX + index_array[0] + (oiter.m_pos % IPX)) +
                               (OFFSETY + index_array[1] + (oiter.m_pos / IPX)) * SHARED_STRIDE];
            case 3:
                return m_sdata[(OFFSETX + index_array[0] + (oiter.m_pos % IPX)) +
                               (OFFSETY + index_array[1] + ((oiter.m_pos / IPX) % IPY)) * SHARED_STRIDE +
                               (OFFSETZ + index_array[2] + ((oiter.m_pos / IPX) / IPY)) * SHARED_STRIDE * SHARED_HEIGHT];
            }
        }

        /**
         * @brief Creates a thread-level iterator that points to the beginning of the current chunk.
         * @return Thread-level iterator.
         */
        __device__ __forceinline__ iterator begin() const
        {
            const unsigned int OFFSETX = (PRINCIPAL_DIM == 0 ? 0 : threadIdx.x * IPX) +
                ((USE_SMEM_DOUBLE_BUFFERING && (m_blockInd % 2 == 1)) ? TOTAL_SHARED : 0);
            const unsigned int OFFSETY = (PRINCIPAL_DIM == 1 ? 0 : threadIdx.y * IPY);
            const unsigned int OFFSETZ = (PRINCIPAL_DIM == 2 ? 0 : threadIdx.z * IPZ);


            switch (DIMS)
            {
            default:
            case 1:
                return iterator(0, OFFSETX, *this);

            case 2:
                return iterator(0, OFFSETX + OFFSETY * SHARED_STRIDE, *this);

            case 3:
                return iterator(0, OFFSETX + OFFSETY * SHARED_STRIDE + 
                                OFFSETZ * SHARED_STRIDE * SHARED_HEIGHT, *this);
            }
        }
        
        /**
         * @brief Creates a thread-level iterator that points to the end of the current chunk.
         * @return Thread-level iterator.
         */
        __device__ __forceinline__ iterator end() const
        {
            const unsigned int OFFSETX = (PRINCIPAL_DIM == 0 ? 0 : threadIdx.x * IPX) +
                ((USE_SMEM_DOUBLE_BUFFERING && (m_blockInd % 2 == 1)) ? TOTAL_SHARED : 0);
            const unsigned int OFFSETY = (PRINCIPAL_DIM == 1 ? 0 : threadIdx.y * IPY);
            const unsigned int OFFSETZ = (PRINCIPAL_DIM == 2 ? 0 : threadIdx.z * IPZ);


            switch (DIMS)
            {
            default:
            case 1:
                return iterator(ELEMENTS, OFFSETX, *this);

            case 2:
                return iterator(ELEMENTS, OFFSETX + OFFSETY * SHARED_STRIDE, *this);

            case 3:
                return iterator(ELEMENTS, OFFSETX + OFFSETY * SHARED_STRIDE +
                                OFFSETZ * SHARED_STRIDE * SHARED_HEIGHT, *this);
            }
        }
        
        /**
         * @brief Creates a thread-level iterator that points to the beginning of the current chunk.
         * @return Thread-level iterator.
         */
        __device__ __forceinline__ iterator align(IOutputContainerIterator& oiter) const
        {
            const unsigned int ILPOFFX = (DIMS == 1 ? oiter.m_pos :
                                          oiter.m_pos % IPX);
            const unsigned int ILPOFFY = (DIMS == 1 ? 0 :
                                          DIMS == 2 ? oiter.m_pos / IPX :
                                          (oiter.m_pos / IPX) % IPY);
            const unsigned int ILPOFFZ = (DIMS < 3 ? 0 : oiter.m_pos / IPX / IPY);

            const unsigned int OFFSETX = (PRINCIPAL_DIM == 0 ? 0 : (threadIdx.x * IPX + ILPOFFX)) +
                ((USE_SMEM_DOUBLE_BUFFERING && (m_blockInd % 2 == 1)) ? TOTAL_SHARED : 0);
            const unsigned int OFFSETY = (PRINCIPAL_DIM == 1 ? 0 : (threadIdx.y * IPY + ILPOFFY));
            const unsigned int OFFSETZ = (PRINCIPAL_DIM == 2 ? 0 : (threadIdx.z * IPZ + ILPOFFZ));


            switch (DIMS)
            {
            default:
            case 1:
                return iterator(0, OFFSETX, *this);

            case 2:
                return iterator(0, OFFSETX + OFFSETY * SHARED_STRIDE, *this);

            case 3:
                return iterator(0, OFFSETX + OFFSETY * SHARED_STRIDE +
                                OFFSETZ * SHARED_STRIDE * SHARED_HEIGHT, *this);
            }
        }

        /**
         * @brief Creates a thread-level iterator that points to the end of the current chunk.
         * @return Thread-level iterator.
         */
        __device__ __forceinline__ iterator end_aligned(IOutputContainerIterator& oiter) const
        {
            const unsigned int ILPOFFX = (DIMS == 1 ? oiter.m_pos :
                                          oiter.m_pos % IPX);
            const unsigned int ILPOFFY = (DIMS == 1 ? 0 :
                                          DIMS == 2 ? oiter.m_pos / IPX :
                                          (oiter.m_pos / IPX) % IPY);
            const unsigned int ILPOFFZ = (DIMS < 3 ? 0 : oiter.m_pos / IPX / IPY);

            const unsigned int OFFSETX = (PRINCIPAL_DIM == 0 ? 0 : (threadIdx.x * IPX + ILPOFFX)) +
                ((USE_SMEM_DOUBLE_BUFFERING && (m_blockInd % 2 == 1)) ? TOTAL_SHARED : 0);
            const unsigned int OFFSETY = (PRINCIPAL_DIM == 1 ? 0 : (threadIdx.y * IPY + ILPOFFY));
            const unsigned int OFFSETZ = (PRINCIPAL_DIM == 2 ? 0 : (threadIdx.z * IPZ + ILPOFFZ));


            switch (DIMS)
            {
            default:
            case 1:
                return iterator(ELEMENTS, OFFSETX, *this);

            case 2:
                return iterator(ELEMENTS, OFFSETX + OFFSETY * SHARED_STRIDE, *this);

            case 3:
                return iterator(ELEMENTS, OFFSETX + OFFSETY * SHARED_STRIDE +
                                OFFSETZ * SHARED_STRIDE * SHARED_HEIGHT, *this);
            }
        }

        /**
         * @brief Progresses to process the next chunk.
         */
        __device__ __forceinline__ void nextChunk() 
        {
            ++m_blockInd;
            
            if (USE_SMEM_DOUBLE_BUFFERING)
            {
                if (m_blockInd < m_blocks)
                    __syncthreads();

                if (m_blockInd < (m_blocks - 1))
                {                    
                    // Prefetch the other double-buffered block asynchronously
                    LoadAsync::Read((T *)m_ptr, m_dimensions, m_stride, 
                                    (PRINCIPAL_DIM == 0) ? (SHARED_WIDTH  * (m_blockInd + 1)) : (SHARED_WIDTH  * (MULTI_GPU ? blockId.x : blockIdx.x)),
                                    DIMS < 2 ? 0 : (PRINCIPAL_DIM == 1) ? (SHARED_HEIGHT * (m_blockInd + 1)) : (SHARED_HEIGHT * (MULTI_GPU ? blockId.y : blockIdx.y)),
                                    DIMS < 3 ? 0 : (PRINCIPAL_DIM == 2) ? (SHARED_DEPTH  * (m_blockInd + 1)) : (SHARED_DEPTH  * (MULTI_GPU ? blockId.z : blockIdx.z)),
                                    m_sdata + ((USE_SMEM_DOUBLE_BUFFERING && (m_blockInd % 2 == 0)) ? TOTAL_SHARED : 0),
                                    m_blockInd + 1, m_blocks);
                }
            }
            else
            {
                if (m_blockInd < m_blocks)
                {
                    __syncthreads();
                               
                    // Load the next block synchronously
                    LoadSync::Read((T *)m_ptr, m_dimensions, m_stride, 
                                   (PRINCIPAL_DIM == 0) ? (SHARED_WIDTH  * m_blockInd) : (SHARED_WIDTH  * (MULTI_GPU ? blockId.x : blockIdx.x)),
                                   DIMS < 2 ? 0 : (PRINCIPAL_DIM == 1) ? (SHARED_HEIGHT * m_blockInd) : (SHARED_HEIGHT * (MULTI_GPU ? blockId.y : blockIdx.y)),
                                   DIMS < 3 ? 0 : (PRINCIPAL_DIM == 2) ? (SHARED_DEPTH  * m_blockInd) : (SHARED_DEPTH  * (MULTI_GPU ? blockId.z : blockIdx.z)),
                                   m_sdata, m_blockInd, m_blocks);
                }
            }
        }

        /**
         * @brief Progresses to process the next chunk without calling __syncthreads().
         * @note This is an advanced function that should be used carefully.
         */
        __device__ __forceinline__ void nextChunkAsync() 
        {
            ++m_blockInd;

            if (USE_SMEM_DOUBLE_BUFFERING)
            {
                if (m_blockInd < (m_blocks - 1))
                {
                    // Prefetch the other double-buffered block asynchronously
                   LoadAsync::Read((T *)m_ptr, m_dimensions, m_stride,
                                   (PRINCIPAL_DIM == 0) ? (SHARED_WIDTH  * (m_blockInd + 1)) : (SHARED_WIDTH  * (MULTI_GPU ? blockId.x : blockIdx.x)),
                                   DIMS < 2 ? 0 : (PRINCIPAL_DIM == 1) ? (SHARED_HEIGHT * (m_blockInd + 1)) : (SHARED_HEIGHT * (MULTI_GPU ? blockId.y : blockIdx.y)),
                                   DIMS < 3 ? 0 : (PRINCIPAL_DIM == 2) ? (SHARED_DEPTH  * (m_blockInd + 1)) : (SHARED_DEPTH  * (MULTI_GPU ? blockId.z : blockIdx.z)),
                                   m_sdata + ((USE_SMEM_DOUBLE_BUFFERING && (m_blockInd % 2 == 0)) ? TOTAL_SHARED : 0), m_blockInd + 1, m_blocks);
                }
            }
            else
            {
                if (m_blockInd < m_blocks)
                {
                    // Load the next block asynchronously
                    LoadAsync::Read((T *)m_ptr, m_dimensions, m_stride,
                                    (PRINCIPAL_DIM == 0) ? (SHARED_WIDTH  * m_blockInd) : (SHARED_WIDTH  * (MULTI_GPU ? blockId.x : blockIdx.x)),
                                    DIMS < 2 ? 0 : (PRINCIPAL_DIM == 1) ? (SHARED_HEIGHT * m_blockInd) : (SHARED_HEIGHT * (MULTI_GPU ? blockId.y : blockIdx.y)),
                                    DIMS < 3 ? 0 : (PRINCIPAL_DIM == 2) ? (SHARED_DEPTH  * m_blockInd) : (SHARED_DEPTH  * (MULTI_GPU ? blockId.z : blockIdx.z)),
                                    m_sdata, m_blockInd, m_blocks);
                }
            }
        }

        /**
         * @brief Returns false if there are more chunks to process.
         */
        __device__ __forceinline__ bool isDone() { return (m_blockInd >= m_blocks); }

        /**
         * @brief Returns the total number of chunks, or 0 for dynamic chunks.
         */
        __device__ __forceinline__ int chunks() { return m_blocks; }
    };

}  // namespace maps

// Iterator implementation
#include "block/block_iterator.inl"

#endif  // __MAPS_BLOCK_CUH_
