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

#ifndef __MAPS_IO_COMMON_CUH_
#define __MAPS_IO_COMMON_CUH_

#include <type_traits>
#include <cuda_runtime.h>

#include "../../internal/common.cuh"
#include "../../internal/cuda_utils.hpp"
#include "../../internal/type_traits.hpp"

// NOTE: MAKE SURE THAT THERE ARE NO "blockIdx" REFERENCES IN THIS FILE.
//       IT IS OVERWRITTEN IN MAPS-MULTI FOR MULTI-GPU PURPOSES.

namespace maps
{
    
    ////////////////////////////////////////////////////////////////////////
    // Global to register reads
    
    template <typename T, GlobalReadScheme GRS = GR_DIRECT, 
              int TEXTURE_UID = -1>
    struct GlobalRead
    {
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      T& value);
        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int stride, int offy, 
                                                      T& value);
    };

    // Indexed global read
    template <typename T, BorderBehavior WB, int BLOCK_WIDTH, int BLOCK_HEIGHT,
              int BLOCK_DEPTH, GlobalReadScheme GRS = GR_DIRECT, 
              int TEXTURE_UID = -1>
    struct IndexedRead
    {
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      int width, T& value);
        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int width, int stride, 
                                                      int offy, int height, 
                                                      T& value);
    };

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
              GlobalReadScheme GRS, int TEXTURE_UID>
    struct IndexedRead<T, WB_NOCHECKS, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                       GRS, TEXTURE_UID>
    {
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      int width, T& value)
        {
            return GlobalRead<T, GRS, TEXTURE_UID>::Read1D(ptr, offset, value);
        }

        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int width, int stride, 
                                                      int offy, int height, 
                                                      T& value)
        {
            return GlobalRead<T, GRS, TEXTURE_UID>::Read2D(ptr, offx, stride, 
                                                           offy, value);
        }
    };

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
              GlobalReadScheme GRS, int TEXTURE_UID>
    struct IndexedRead<T, WB_ZERO, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, GRS,
                       TEXTURE_UID>
    {
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      int width, T& value)
        {
            if (offset < 0 || offset >= width)
            {
                value = T(0);
                return true;
            }

            return GlobalRead<T, GRS, TEXTURE_UID>::Read1D(ptr, offset, value);
        }

        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int width, int stride, 
                                                      int offy, int height, 
                                                      T& value)
        {
            if (offx < 0 || offy < 0 || offx >= width || offy >= height)
            {
                value = T(0);
                return true;
            }
            return GlobalRead<T, GRS, TEXTURE_UID>::Read2D(ptr, offx, stride, 
                                                           offy, value);
        }
    };

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
              GlobalReadScheme GRS, int TEXTURE_UID>
    struct IndexedRead<T, WB_COPY, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, GRS,
                       TEXTURE_UID>
    {
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      int width, T& value)
        {
            return GlobalRead<T, GRS, TEXTURE_UID>::Read1D(ptr, 
              Clamp(offset, 0, width - 1), value);
        }

        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int width, int stride, 
                                                      int offy, int height, 
                                                      T& value)
        {
            return GlobalRead<T, GRS, TEXTURE_UID>::Read2D(ptr, 
              Clamp(offx, 0, width - 1), stride, Clamp(offy, 0, height - 1), 
              value);
        }
    };

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
              GlobalReadScheme GRS, int TEXTURE_UID>
    struct IndexedRead<T, WB_WRAP, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, GRS,
                       TEXTURE_UID>
    {
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      int width, T& value)
        {
            return GlobalRead<T, GRS, TEXTURE_UID>::Read1D(ptr, 
              Wrap(offset, width), value);
        }

        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int width, int stride,
                                                      int offy, int height, 
                                                      T& value)
        {
            return GlobalRead<T, GRS, TEXTURE_UID>::Read2D(ptr, 
              Wrap(offx, width), stride, Wrap(offy, height), value);
        }
    };

    ////////////////////////////////////////////////////////////////////////
    // Global write from register

    template <typename T>
    struct GlobalWrite
    {
        static __device__ __forceinline__ void Write(T *ptr, int offset, 
                                                     const T& value)
        {
            *(ptr + offset) = value;
        }
    };

    ////////////////////////////////////////////////////////////////////////
    // Global to shared reads

    template <typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
              int BLOCK_DEPTH, int XSHARED, int XSTRIDE, int YSHARED, 
              int ZSHARED, bool ASYNC, BorderBehavior BORDERS, 
              GlobalReadScheme GRS = GR_DISTINCT, int TEXTURE_UID = -1>
    struct GlobalToShared
    {
        static __device__ __forceinline__ bool Read(const T *ptr, 
                                                    int dimensions[DIMS], 
                                                    int stride, int xoffset,
                                                    int yoffset, int zoffset, 
                                                    T *smem, int chunkID, 
                                                    int num_chunks);
    };

    ////////////////////////////////////////////////////////////////////////
    // Global to register array reads

    template <typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT,
              int BLOCK_DEPTH, int DIMX, int DIMY, int DIMZ,
              BorderBehavior BORDERS = WB_NOCHECKS,
              GlobalReadScheme GRS = GR_DISTINCT, int TEXTURE_UID = -1>
    struct GlobalToArray
    {
        static __device__ __forceinline__ bool Read(const T *ptr, 
                                                    int dimensions[DIMS], 
                                                    int stride, int xoffset,
                                                    int yoffset, int zoffset, 
                                                    T (&regs)[DIMX*DIMY*DIMZ], 
                                                    int chunkID, 
                                                    int num_chunks);
    };

    //////////////////////////////////////////////////////////////////////////
    // Register array to global write + helper structure

    template <typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
              int BLOCK_DEPTH, int DIMX, int DIMY, int DIMZ>
    struct ArrayToGlobal
    {
        static __device__ __forceinline__ bool Write(
          const T (&regs)[DIMX*DIMY*DIMZ], int dimensions[DIMS], int stride, 
          int xoffset, int yoffset, int zoffset, T *ptr);
    };

}  // namespace maps

#endif  // __MAPS_IO_COMMON_CUH_



