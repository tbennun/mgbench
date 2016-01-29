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

#ifndef __MAPS_COMMON_CUH_
#define __MAPS_COMMON_CUH_

#include <cuda_runtime.h>
#include <iterator>
#include "common.h"

namespace maps
{
    template <typename T>
    static __device__ __forceinline__ T LDG(T *ptr)
    {
#if (__CUDA_ARCH__ >= 320) // Kepler-based devices and above
        return __ldg(ptr);
#else
        return *ptr;
#endif
    }
    
    /// @brief Dynamic shared memory wrapper for general use.
    template<typename T>
    struct DynamicSharedMemory
    {
        ptrdiff_t m_offset;
        
        /**
         * @brief Initializes dynamic shared memory pointer with offset.
         * @param offset The offset (in bytes) where this buffer starts.
         */
        __device__ __forceinline__ void init(ptrdiff_t offset)
        {
            m_offset = offset;
        }

        __device__ __forceinline__ T *ptr()
        {
            extern __shared__ unsigned char __smem[];
            return (T *)(__smem + m_offset);
        }

        __device__ __forceinline__ const T *ptr() const
        {
            extern __shared__ unsigned char __smem[];
            return (const T *)(__smem + m_offset);
        }
    };

    /**
     * @brief A shared-memory array wrapper that can allocate both static 
     *        and dynamic shared memory. 
     * 
     * @note The struct must be designated "__shared__" on declaration,
     *       or part of a shared class.
     */
    template<typename T, size_t ARRAY_SIZE = DYNAMIC_SMEM>
    struct SharedMemoryArray
    {
        T smem[ARRAY_SIZE];

        __device__ __forceinline__ void init(ptrdiff_t offset = 0)
        {
            // Do nothing
        }
    };

    // Specialization for dynamic shared memory
    template<typename T>
    struct SharedMemoryArray<T, 0>
    {        
        T *smem;

        __device__ __forceinline__ void init(ptrdiff_t offset = 0)
        {
            extern __shared__ unsigned char __smem[];
            smem = (T *)(__smem + offset);
        }
    };

    namespace internal
    {
        static __device__ __forceinline__ bool __NextChunkAsync()
        {
            return false;
        }

        template <typename T>
        static __device__ __forceinline__ bool __NextChunkAsync(T& container)
        {
            container.nextChunkAsync();
            return T::SYNC_AFTER_NEXTCHUNK;
        }

        template <typename First, typename... Rest>
        static __device__ __forceinline__ bool __NextChunkAsync(First& first, Rest&... rest)
        {
            first.nextChunkAsync();
            return __NextChunkAsync(rest...) || First::SYNC_AFTER_NEXTCHUNK;
        }
    }  // namespace internal

    template <typename... Args>
    static __device__ __forceinline__ void NextChunkAll(Args&... args) {
        __syncthreads();
        bool bSync = internal::__NextChunkAsync(args...);
        if(bSync)
            __syncthreads();
    }    


    // Helper macros for MAPS_INIT
    #define _MI0()
  
    #define _MI1(arg1)                                                   \
            __shared__ typename decltype(arg1)::SharedData arg1##_sdata; \
            arg1.init_async(arg1##_sdata);                               \
            if(decltype(arg1)::SYNC_AFTER_INIT)                          \
                __syncthreads();                                         \
            arg1.init_async_postsync();

    #define _MI2(arg1, arg2)                                             \
            __shared__ typename decltype(arg1)::SharedData arg1##_sdata; \
            __shared__ typename decltype(arg2)::SharedData arg2##_sdata; \
            arg1.init_async(arg1##_sdata);                               \
            arg2.init_async(arg2##_sdata);                               \
            if(decltype(arg1)::SYNC_AFTER_INIT || decltype(arg2)::SYNC_AFTER_INIT) \
                __syncthreads();                                         \
            arg1.init_async_postsync();                                  \
            arg2.init_async_postsync();

    #define _MI3(arg1, arg2, arg3)                                       \
            __shared__ typename decltype(arg1)::SharedData arg1##_sdata; \
            __shared__ typename decltype(arg2)::SharedData arg2##_sdata; \
            __shared__ typename decltype(arg3)::SharedData arg3##_sdata; \
            arg1.init_async(arg1##_sdata);                               \
            arg2.init_async(arg2##_sdata);                               \
            arg3.init_async(arg3##_sdata);                               \
            if(decltype(arg1)::SYNC_AFTER_INIT || decltype(arg2)::SYNC_AFTER_INIT || decltype(arg3)::SYNC_AFTER_INIT) \
                __syncthreads();                                         \
            arg1.init_async_postsync();                                  \
            arg2.init_async_postsync();                                  \
            arg3.init_async_postsync();

    #define _MI4(arg1, arg2, arg3, arg4)                                 \
            __shared__ typename decltype(arg1)::SharedData arg1##_sdata; \
            __shared__ typename decltype(arg2)::SharedData arg2##_sdata; \
            __shared__ typename decltype(arg3)::SharedData arg3##_sdata; \
            __shared__ typename decltype(arg4)::SharedData arg4##_sdata; \
            arg1.init_async(arg1##_sdata);                               \
            arg2.init_async(arg2##_sdata);                               \
            arg3.init_async(arg3##_sdata);                               \
            arg4.init_async(arg4##_sdata);                               \
            if(decltype(arg1)::SYNC_AFTER_INIT || decltype(arg2)::SYNC_AFTER_INIT || decltype(arg3)::SYNC_AFTER_INIT || decltype(arg4)::SYNC_AFTER_INIT) \
                __syncthreads();                                         \
            arg1.init_async_postsync();                                  \
            arg2.init_async_postsync();                                  \
            arg3.init_async_postsync();                                  \
            arg4.init_async_postsync();

    #define _MI5(arg1, arg2, arg3, arg4, arg5)                           \
            __shared__ typename decltype(arg1)::SharedData arg1##_sdata; \
            __shared__ typename decltype(arg2)::SharedData arg2##_sdata; \
            __shared__ typename decltype(arg3)::SharedData arg3##_sdata; \
            __shared__ typename decltype(arg4)::SharedData arg4##_sdata; \
            __shared__ typename decltype(arg5)::SharedData arg5##_sdata; \
            arg1.init_async(arg1##_sdata);                               \
            arg2.init_async(arg2##_sdata);                               \
            arg3.init_async(arg3##_sdata);                               \
            arg4.init_async(arg4##_sdata);                               \
            arg5.init_async(arg5##_sdata);                               \
            if(decltype(arg1)::SYNC_AFTER_INIT || decltype(arg2)::SYNC_AFTER_INIT || decltype(arg3)::SYNC_AFTER_INIT || decltype(arg4)::SYNC_AFTER_INIT || decltype(arg5)::SYNC_AFTER_INIT) \
                __syncthreads();                                         \
            arg1.init_async_postsync();                                  \
            arg2.init_async_postsync();                                  \
            arg3.init_async_postsync();                                  \
            arg4.init_async_postsync();                                  \
            arg5.init_async_postsync();

    #define EXPAND(x) x
    #define GET_MACRO(_1,_2,_3,_4,_5,NAME,...) NAME
    #define MAPS_INIT(...) EXPAND(GET_MACRO(__VA_ARGS__, _MI5, _MI4, _MI3, _MI2, _MI1, _MI0)(__VA_ARGS__))
        

}  // namespace maps

#endif  // __MAPS_COMMON_CUH_
