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

#ifndef __MAPS_IRREGULAR_INPUT_CUH_
#define __MAPS_IRREGULAR_INPUT_CUH_

namespace maps
{
    template<typename T, int DIMS>
    class IrregularInput : public IInputContainer
    {
        // Static assertions
        MAPS_STATIC_ASSERT(DIMS > 0, "Dimensions must be positive");

    public:
        enum
        {
            SYNC_AFTER_INIT = false,
            SYNC_AFTER_NEXTCHUNK = false,
        };

        int m_dims[DIMS];
        size_t m_stride;        

        struct SharedData
        {
        };

        __host__ __device__ IrregularInput() : m_stride(0) {}

        __host__ __device__ T* GetTypedPtr() { return (T *)m_ptr; }

        __device__ __forceinline__ void init(const T *ptr, size_t width, 
                                             size_t stride, size_t height)
        {
            m_ptr = ptr;
            m_dims[0] = width;
            m_dims[1] = height;
            m_stride = stride;
        }

        __device__ __forceinline__ void init() { }
        __device__ __forceinline__ void init_async(SharedData& sdata) { }
        __device__ __forceinline__ void init_async_postsync() { }
        __device__ __forceinline__ void nextChunk() { }
        __device__ __forceinline__ void nextChunkAsync() { }
        __device__ __forceinline__ bool isDone() { return true; }
        __device__ __forceinline__ int chunks() { return 1; }
    };

}  // namespace maps

#endif  // __MAPS_IRREGULAR_INPUT_CUH_
