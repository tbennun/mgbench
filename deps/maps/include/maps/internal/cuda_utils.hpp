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

#ifndef __MAPS_CUDA_UTILS_HPP_
#define __MAPS_CUDA_UTILS_HPP_

#include <vector>

#include <cuda_runtime.h>
#include <vector_types.h>

namespace maps
{
    static void HandleError(const char *file, int line, cudaError_t err)
    {
        printf("ERROR in %s:%d: %s (%d)\n", file, line, 
               cudaGetErrorString(err), err);
        exit(1);
    }

    // CUDA assertions
#define MAPS_CUDA_CHECK(err) do {                                       \
    cudaError_t errr = (err);                                           \
    if(errr != cudaSuccess)                                             \
    {                                                                   \
        ::maps::HandleError(__FILE__, __LINE__, errr);                  \
    }                                                                   \
} while(0)

    static inline __host__ __device__ unsigned int RoundUp(
      unsigned int numerator, unsigned int denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }

    static inline void CudaAlloc(void** d_ptr, unsigned int size)
    {
        MAPS_CUDA_CHECK(cudaMalloc(d_ptr, size));
    }

    static inline void CudaAllocAndMemsetToZero(void** d_ptr, unsigned int size)
    {
        MAPS_CUDA_CHECK(cudaMalloc(d_ptr, size));
        MAPS_CUDA_CHECK(cudaMemset(*d_ptr, 0, size));
    }
    
    static inline void CudaAllocAndCopy(void** d_ptr, void* h_ptr, 
                                        unsigned int size)
    {
        CudaAlloc(d_ptr,size);
        MAPS_CUDA_CHECK(cudaMemcpy(*d_ptr,h_ptr,size, cudaMemcpyHostToDevice));
    }

    static inline void CudaSafeFree(void* d_ptr)
    {
        if (d_ptr)
            MAPS_CUDA_CHECK(cudaFree(d_ptr));
    }

}  // namespace maps

#endif  // __MAPS_CUDA_UTILS_HPP_
