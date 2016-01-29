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

#ifndef __MAPS_IO_GLOBALTOARRAY_CUH_
#define __MAPS_IO_GLOBALTOARRAY_CUH_

#include "../../internal/common.cuh"
#include "../../internal/cuda_utils.hpp"
#include "../../internal/type_traits.hpp"

#include "io_common.cuh"

namespace maps
{
    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
              int DIMX, int DIMY, int DIMZ, BorderBehavior BORDERS, 
              GlobalReadScheme GRS, int TEXTURE_UID>
    struct GlobalToArray<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, DIMX, 
                         DIMY, DIMZ, BORDERS, GRS, TEXTURE_UID>
    {
        static __device__ __forceinline__ bool Read(
            const T *ptr, int dimensions[1], int stride, int xoffset,
            int yoffset, int zoffset, T (&regs)[DIMX*DIMY*DIMZ],
            int chunkID, int num_chunks)
        {
            // TODO (later): Vectorized reads

            bool result = true;
            
            #pragma unroll
            for (int i = 0; i < DIMX; ++i)
            {
                result &= IndexedRead<T, BORDERS, BLOCK_WIDTH, BLOCK_HEIGHT,
                                      BLOCK_DEPTH, GRS, TEXTURE_UID>::Read1D(
                    ptr, xoffset + i, dimensions[0], regs[i]);
            }

            return result;
        }
    };

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
              int DIMX, int DIMY, int DIMZ, BorderBehavior BORDERS, 
              GlobalReadScheme GRS, int TEXTURE_UID>
    struct GlobalToArray<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, DIMX, 
                         DIMY, DIMZ, BORDERS, GRS, TEXTURE_UID>
    {
        static __device__ __forceinline__ bool Read(
            const T *ptr, int dimensions[2], int stride, int xoffset,
            int yoffset, int zoffset, T (&regs)[DIMX*DIMY*DIMZ],
            int chunkID, int num_chunks)
        {
            // TODO (later): Vectorized reads

            bool result = true;

            #pragma unroll
            for (int y = 0; y < DIMY; ++y)
            {
                #pragma unroll
                for (int x = 0; x < DIMX; ++x)
                {
                    result &= IndexedRead<T, BORDERS, BLOCK_WIDTH, BLOCK_HEIGHT,
                                          BLOCK_DEPTH, GRS, 
                                          TEXTURE_UID>::Read2D(
                      ptr, xoffset + x, dimensions[0], stride, yoffset + y, 
                      dimensions[1], regs[y * DIMX + x]);
                }
            }

            return result;
        }
    };

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
              int DIMX, int DIMY, int DIMZ, BorderBehavior BORDERS, 
              GlobalReadScheme GRS, int TEXTURE_UID>
    struct GlobalToArray<T, 3, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, DIMX, 
                         DIMY, DIMZ, BORDERS, GRS, TEXTURE_UID>
    {
        static __device__ __forceinline__ bool Read(
            const T *ptr, int dimensions[3], int stride, int xoffset,
            int yoffset, int zoffset, T (&regs)[DIMX*DIMY*DIMZ],
            int chunkID, int num_chunks)
        {
            // TODO: 3D reads
            return false;
        }
    };

    //////////////////////////////////////////////////////////////////////////

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
              int DIMX, int DIMY, int DIMZ>
    struct ArrayToGlobal<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, DIMX, 
                         DIMY, DIMZ>
    {
        static __device__ __forceinline__ bool Write(
            const T (&regs)[DIMX*DIMY*DIMZ], int dimensions[1], int stride, 
            int xoffset, int yoffset, int zoffset, T *ptr)
        {
            // TODO (later): Vectorized writes

            #pragma unroll
            for (int i = 0; i < DIMX; ++i)
            {
                GlobalWrite<T>::Write(ptr, xoffset + threadIdx.x * DIMX + i, 
                                      regs[i]);
            }

            return true;
        }
    };

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
              int DIMX, int DIMY, int DIMZ>
    struct ArrayToGlobal<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, DIMX, 
                         DIMY, DIMZ>
    {
        static __device__ __forceinline__ bool Write(
            const T (&regs)[DIMX*DIMY*DIMZ], int dimensions[2], int stride, 
                int xoffset, int yoffset, int zoffset, T *ptr)
        {
            // TODO (later): Vectorized reads
            #pragma unroll
            for (int y = 0; y < DIMY; ++y)
            {
                #pragma unroll
                for (int x = 0; x < DIMX; ++x)
                {
                    GlobalWrite<T>::Write(ptr, 
                      xoffset + threadIdx.x * DIMX + x + 
                      (yoffset + threadIdx.y * DIMY + y) * stride, 
                      regs[y * DIMX + x]);
                }
            }

            return true;
        }
    };

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
              int DIMX, int DIMY, int DIMZ>
    struct ArrayToGlobal<T, 3, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, DIMX, 
                         DIMY, DIMZ>
    {
        static __device__ __forceinline__ bool Write(
            const T (&regs)[DIMX*DIMY*DIMZ], int dimensions[3], int stride, 
            int xoffset, int yoffset, int zoffset, T *ptr)
        {
            // TODO (later): Vectorized writes
            #pragma unroll
            for (int z = 0; z < DIMZ; ++z)
            {
                #pragma unroll
                for (int y = 0; y < DIMY; ++y)
                {
                    #pragma unroll
                    for (int x = 0; x < DIMX; ++x)
                    {
                        GlobalWrite<T>::Write(
                          ptr, 
                          xoffset + threadIdx.x * DIMX + x + 
                          (yoffset + threadIdx.y * DIMY + y) * stride + 
                          (zoffset + threadIdx.z * DIMZ + z) * stride * dimensions[1], 
                          regs[z * DIMY * DIMX + y * DIMX + x]);
                    }
                }
            }

            return true;            
        }
    };

}  // namespace maps

#endif  // __MAPS_IO_GLOBALTOARRAY_CUH_
