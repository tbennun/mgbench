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

#ifndef __MAPS_IO_GLOBALREAD_CUH_
#define __MAPS_IO_GLOBALREAD_CUH_

#include "../../internal/common.cuh"
#include "../../internal/cuda_utils.hpp"
#include "../../internal/texref.cuh"
#include "../../internal/type_traits.hpp"

#include "io_common.cuh"

namespace maps
{

    template <typename T>
    struct GlobalRead<T, GR_DIRECT, -1>
    {
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      T& value)
        {
            value = *(ptr + offset);
            return true;
        }

        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int stride, int offy, 
                                                      T& value)
        {
            value = *(ptr + offy * stride + offx);
            return true;
        }
    };

    template <typename T>
    struct GlobalRead<T, GR_DISTINCT, -1>
    {
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      T& value)
        {
#if __CUDA_ARCH__ >= 320
            value = __ldg(ptr + offset);
#else
            value = *(ptr + offset);
#endif
            return true;
        }

        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int stride, int offy, 
                                                      T& value)
        {
#if __CUDA_ARCH__ >= 320
            value = __ldg(ptr + offy * stride + offx);
#else
            value = *(ptr + offy * stride + offx);
#endif            
            return true;
        }
    };

    template <typename T, int TEXTURE_UID>
    struct GlobalRead<T, GR_TEXTURE, TEXTURE_UID>
    {
        MAPS_STATIC_ASSERT(TEXTURE_UID >= 0, "Texture UID cannot be negative");

        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      T& value)
        {
            typedef typename UniqueTexRef1D<T>::template TexId<TEXTURE_UID> 
                TexId;
            value = TexId::read(offset + 0.5f);
            return true;
        }

        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int stride, int offy, 
                                                      T& value)
        {
            typedef typename UniqueTexRef2D<T>::template TexId<TEXTURE_UID> 
                TexId;
            value = TexId::read(offx + 0.5f, offy + 0.5f);
            return true;
        }
    };

}  // namespace maps

#endif  // __MAPS_IO_GLOBALREAD_CUH_
