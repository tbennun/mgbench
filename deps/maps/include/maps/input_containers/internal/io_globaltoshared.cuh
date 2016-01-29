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

#ifndef __MAPS_IO_GLOBALTOSHARED_CUH_
#define __MAPS_IO_GLOBALTOSHARED_CUH_

#include "../../internal/common.cuh"
#include "../../internal/cuda_utils.hpp"
#include "../../internal/type_traits.hpp"

#include "io_common.cuh"

namespace maps
{ 
    ////////////////////////////////////////////////////////////////////////
    // Detect optimal bit read preference

    // TODO (later): Verify
    enum
    {
#if __CUDA_ARCH__ >= 500
        PREFERRED_GREAD_SIZE = 128 / 8, // 128-bit
        PREFERRED_SWRITE_SIZE = 128 / 8, // 128-bit
#elif __CUDA_ARCH__ >= 300
        PREFERRED_GREAD_SIZE = 128 / 8, // 128-bit
        PREFERRED_SWRITE_SIZE = 64 / 8, // 64-bit
#elif __CUDA_ARCH__ >= 130
        PREFERRED_GREAD_SIZE = 64 / 8, // 64-bit
        PREFERRED_SWRITE_SIZE = 32 / 8, // 32-bit
#else
        PREFERRED_GREAD_SIZE = 32 / 8, // Default to 32-bit loads
        PREFERRED_SWRITE_SIZE = 32 / 8, // 32-bit
#endif
    };



    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
              int XSHARED, bool ASYNC, BorderBehavior BORDERS, 
              GlobalReadScheme GRS = GR_DISTINCT, int TEXTURE_UID = -1>
    static __device__ __forceinline__ bool GlobalToShared1D(
        const T *ptr, int width, int offset, T *smem, int chunkID, 
        int num_chunks)
    {
        // TODO(later): Vectorized reads
        
        //typedef typename BytesToType<PREFERRED_GREAD_SIZE>::type GReadType;
        typedef T GReadType;
        typedef T SWriteType;
        MAPS_STATIC_ASSERT(sizeof(T) <= sizeof(GReadType), 
                           "Base element size must be less than or equal to "
                           "global read type size");

        bool result = true;

        // Load memory in sizes that may be different than sizeof(T)
        const GReadType *p = (const GReadType *)ptr;
        SWriteType *s = (SWriteType *)smem;

        GReadType tmp;
        const SWriteType *tmp_stype = (const SWriteType *)&tmp;

        enum
        {
            BLOCK_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * BLOCK_DEPTH,

            SWRITES_PER_GREAD = sizeof(GReadType) / sizeof(SWriteType),
            TOTAL_BYTES = XSHARED * sizeof(T),
            TOTAL_GREADS_IN_XSHARED = TOTAL_BYTES / sizeof(GReadType),
            TOTAL_READS = TOTAL_GREADS_IN_XSHARED / BLOCK_SIZE,
            REMAINDER_GREADS = (TOTAL_GREADS_IN_XSHARED % BLOCK_SIZE),
        };

        // Compute linear thread index
        int tid = (BLOCK_DEPTH > 1 ? (threadIdx.z * BLOCK_WIDTH * BLOCK_HEIGHT) : 0) +
                  (BLOCK_HEIGHT > 1 ? (threadIdx.y * BLOCK_WIDTH) : 0) +
                  threadIdx.x;

        #pragma unroll
        for (int i = 0; i < TOTAL_READS; ++i)
        {
            // Read from global memory
            result &= IndexedRead<GReadType, BORDERS, BLOCK_WIDTH, BLOCK_HEIGHT,
                                  BLOCK_DEPTH, GRS, TEXTURE_UID>::Read1D(
                p,
                offset + BLOCK_SIZE * i + tid, width,
                tmp);

            // Write to shared memory
            #pragma unroll
            for (int j = 0; j < SWRITES_PER_GREAD; ++j)
            {
                s[BLOCK_SIZE * i + SWRITES_PER_GREAD * tid + j] = tmp_stype[j];
            }
        }
        
        // Remainder
        if (REMAINDER_GREADS != 0 && tid < REMAINDER_GREADS)
        {
            enum
            {
                REMAINDER_START_INDEX = TOTAL_READS * BLOCK_SIZE,
            };

            // Read from global memory
            result &= IndexedRead<GReadType, BORDERS, BLOCK_WIDTH, BLOCK_HEIGHT,
                                  BLOCK_DEPTH, GRS, TEXTURE_UID>::Read1D(
                p, 
                offset + REMAINDER_START_INDEX + tid, width,
                tmp);

            // Write to shared memory
            #pragma unroll
            for (int j = 0; j < SWRITES_PER_GREAD; ++j)
            {
                s[REMAINDER_START_INDEX + SWRITES_PER_GREAD * tid + j] = 
                    tmp_stype[j];
            }
        }

        // Special case: The total size is less than a single GReadType
        if (XSHARED * sizeof(T) < sizeof(GReadType))
        {
            if (tid < XSHARED)
            {
                // Read T-sized elements, write directly to shared
                result &= IndexedRead<T, BORDERS, BLOCK_WIDTH, BLOCK_HEIGHT, 
                                      BLOCK_DEPTH, GRS, TEXTURE_UID>::Read1D(
                    ptr,
                    offset + tid, width,
                    tmp);

                smem[tid] = tmp;
            }
        }
        
        if (!ASYNC)
            __syncthreads();

        return result;
    }

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
              int XSHARED, int XSTRIDE, int YSHARED, bool ASYNC, 
              BorderBehavior BORDERS, GlobalReadScheme GRS = GR_DISTINCT, 
              int TEXTURE_UID = -1>
    static __device__ __forceinline__ bool GlobalToShared2D(
        const T *ptr, int width, int stride, int xoffset, int height, 
        int yoffset, T *smem, int chunkID, int num_chunks)
    {      
        // TODO(later): Vectorized reads
        typedef T GReadType;
        typedef T SWriteType;
        MAPS_STATIC_ASSERT(sizeof(T) <= sizeof(GReadType), 
                           "Base element size must be less than or equal to "
                           "global read type size");
        
        bool result = true;

        const GReadType *p = (const GReadType *)ptr;
        SWriteType *s = (SWriteType *)smem;

        GReadType tmp;
        const SWriteType *tmp_stype = (const SWriteType *)&tmp;

        enum
        {
            SWRITES_PER_GREAD = sizeof(GReadType) / sizeof(SWriteType),
            
            BLOCK_SIZE = BLOCK_HEIGHT * BLOCK_DEPTH,

            TOTAL_BYTES_X = XSHARED * sizeof(T),
            TOTAL_GREADS_IN_XSHARED = TOTAL_BYTES_X / sizeof(GReadType),
            TOTAL_READS_X = TOTAL_GREADS_IN_XSHARED / BLOCK_WIDTH,
            REMAINDER_READS_X = TOTAL_GREADS_IN_XSHARED % BLOCK_WIDTH,

            TOTAL_READS_Y = YSHARED / BLOCK_SIZE,
            REMAINDER_READS_Y = YSHARED % BLOCK_SIZE,

            XREMAINDER_START_INDEX = TOTAL_READS_X * BLOCK_WIDTH,
            YREMAINDER_START_INDEX = TOTAL_READS_Y * BLOCK_SIZE,
        };
       
        // Compute linear thread (x,y) indices
        int tidx = threadIdx.x;
        int tidy = (BLOCK_DEPTH > 1 ? (threadIdx.z * BLOCK_HEIGHT) : 0) +
                   threadIdx.y;

        // Loop over Y read chunks
        #pragma unroll
        for (int i = 0; i < TOTAL_READS_Y; ++i)
        {
            // Loop over X read chunks
            #pragma unroll
            for (int j = 0; j < TOTAL_READS_X; ++j)
            {
                // Read from global memory
                result &= IndexedRead<GReadType, BORDERS, BLOCK_WIDTH, 
                                      BLOCK_HEIGHT, BLOCK_DEPTH, GRS, 
                                      TEXTURE_UID>::Read2D(
                    p,
                    xoffset + BLOCK_WIDTH * j + tidx, width,
                    stride,
                    yoffset + BLOCK_SIZE * i + tidy, height,
                    tmp);

                // Write to shared memory
                #pragma unroll
                for (int k = 0; k < SWRITES_PER_GREAD; ++k)
                {
                    s[(i * BLOCK_SIZE + tidy) * XSTRIDE + j * BLOCK_WIDTH + 
                      SWRITES_PER_GREAD * tidx + k] = tmp_stype[k];
                }
            }

            // Remainder X reads
            if (REMAINDER_READS_X != 0 && tidx < REMAINDER_READS_X)
            {
                // Read from global memory
                result &= IndexedRead<GReadType, BORDERS, BLOCK_WIDTH, 
                                      BLOCK_HEIGHT, BLOCK_DEPTH, GRS, 
                                      TEXTURE_UID>::Read2D(
                    p,
                    xoffset + XREMAINDER_START_INDEX + tidx, width,
                    stride,
                    yoffset + BLOCK_SIZE * i + tidy, height,
                    tmp);

                // Write to shared memory
                #pragma unroll
                for (int k = 0; k < SWRITES_PER_GREAD; ++k)
                {
                    s[(i * BLOCK_SIZE + tidy) * XSTRIDE + 
                      XREMAINDER_START_INDEX + SWRITES_PER_GREAD * tidx + k] = 
                        tmp_stype[k];
                }
            }
        }

        // Remainder Y reads
        if (REMAINDER_READS_Y != 0 && tidy < REMAINDER_READS_Y)
        {
            // Loop over X read chunks
            #pragma unroll
            for (int j = 0; j < TOTAL_READS_X; ++j)
            {
                // Read from global memory
                result &= IndexedRead<GReadType, BORDERS, BLOCK_WIDTH, 
                                      BLOCK_HEIGHT, BLOCK_DEPTH, GRS, 
                                      TEXTURE_UID>::Read2D(
                    p,
                    xoffset + BLOCK_WIDTH * j + tidx, width,
                    stride,
                    yoffset + YREMAINDER_START_INDEX + tidy, height,
                    tmp);

                // Write to shared memory
                #pragma unroll
                for (int k = 0; k < SWRITES_PER_GREAD; ++k)
                {
                    s[(YREMAINDER_START_INDEX + tidy) * XSTRIDE + 
                      j * BLOCK_WIDTH + SWRITES_PER_GREAD * tidx + k] = 
                        tmp_stype[k];
                }
            }

            // Remainder X reads (inside remainder Y reads)
            if (REMAINDER_READS_X != 0 && tidx < REMAINDER_READS_X)
            {
                // Read from global memory
                result &= IndexedRead<GReadType, BORDERS, BLOCK_WIDTH, 
                                      BLOCK_HEIGHT, BLOCK_DEPTH, GRS, 
                                      TEXTURE_UID>::Read2D(
                    p,
                    xoffset + XREMAINDER_START_INDEX + tidx, width,
                    stride,
                    yoffset + YREMAINDER_START_INDEX + tidy, height,
                    tmp);

                // Write to shared memory
                #pragma unroll
                for (int k = 0; k < SWRITES_PER_GREAD; ++k)
                {
                    s[(YREMAINDER_START_INDEX + tidy) * XSTRIDE + 
                      XREMAINDER_START_INDEX + SWRITES_PER_GREAD * tidx + k] =
                        tmp_stype[k];
                }
            }
        }

        if (!ASYNC)
            __syncthreads();

        return result;
    }

    template <typename T, int PRINCIPAL_DIM, int BLOCK_WIDTH, int BLOCK_HEIGHT,
              int BLOCK_DEPTH, int XSHARED, int XSTRIDE, int YSHARED, 
              int ZSHARED, bool ASYNC, BorderBehavior BORDERS, 
              GlobalReadScheme GRS = GR_DISTINCT, int TEXTURE_UID = -1>
    static __device__ __forceinline__ bool GlobalToShared3D(
        const T *ptr, int width, int stride, int xoffset,
        int height, int yoffset, int depth, int zoffset,
        T *smem, int chunkID, int num_chunks)
    {
        // TODO: 3D reads
        return false;
    }

    /////////////////////////////////////////////////////////////////////////

    // Template-to-function multiplexers

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
              int XSHARED, int XSTRIDE, int YSHARED, int ZSHARED, bool ASYNC, 
              BorderBehavior BORDERS, GlobalReadScheme GRS, int TEXTURE_UID>
    struct GlobalToShared<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, XSHARED,
                          XSTRIDE, YSHARED, ZSHARED, ASYNC, BORDERS, GRS, 
                          TEXTURE_UID>
    {
        static __device__ __forceinline__ bool Read(
            const T *ptr, int dimensions[1], int stride, int xoffset,
            int yoffset, int zoffset, T *smem, int chunkID, int num_chunks)
        {
            return GlobalToShared1D<T, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                                    XSHARED, ASYNC, BORDERS, GRS, TEXTURE_UID>(
              ptr, dimensions[0], xoffset, smem, chunkID, num_chunks);
        }
    };

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
              int XSHARED, int XSTRIDE, int YSHARED, int ZSHARED, bool ASYNC, 
              BorderBehavior BORDERS, GlobalReadScheme GRS, int TEXTURE_UID>
    struct GlobalToShared<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, XSHARED,
                          XSTRIDE, YSHARED, ZSHARED, ASYNC, BORDERS, GRS, 
                          TEXTURE_UID>
    {
        static __device__ __forceinline__ bool Read(
            const T *ptr, int dimensions[2], int stride, int xoffset,
            int yoffset, int zoffset, T *smem, int chunkID, int num_chunks)
        {
            return GlobalToShared2D<T, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH,
                                    XSHARED, XSTRIDE, YSHARED, ASYNC, BORDERS, 
                                    GRS, TEXTURE_UID>(
                ptr, dimensions[0], stride, xoffset, dimensions[1], yoffset, 
                smem, chunkID, num_chunks);
        }
    };

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
              int XSHARED, int XSTRIDE, int YSHARED, int ZSHARED, bool ASYNC, 
              BorderBehavior BORDERS, GlobalReadScheme GRS, int TEXTURE_UID>
    struct GlobalToShared<T, 3, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, XSHARED,
                          XSTRIDE, YSHARED, ZSHARED, ASYNC, BORDERS, GRS, 
                          TEXTURE_UID>
    {
        static __device__ __forceinline__ bool Read(
            const T *ptr, int dimensions[3], int stride, int xoffset,
            int yoffset, int zoffset, T *smem, int chunkID, int num_chunks)
        {
            return GlobalToShared3D<T, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                                    XSHARED, XSTRIDE, YSHARED, ZSHARED,
                                    ASYNC, BORDERS, GRS, TEXTURE_UID>(
                ptr, dimensions[0], stride, xoffset, dimensions[1], yoffset, 
                dimensions[2], zoffset, smem, chunkID, num_chunks);
        }
    };

}  // namespace maps

#endif  // __MAPS_IO_GLOBALTOSHARED_CUH_
