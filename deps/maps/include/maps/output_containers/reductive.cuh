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

#ifndef __MAPS_REDUCTIVE_OUTPUT_CUH_
#define __MAPS_REDUCTIVE_OUTPUT_CUH_

#include "../internal/common.cuh"

namespace maps
{
    template <typename T, int LENGTH, int BLOCK_WIDTH, int ITEMS_PER_THREAD = 1>
    struct ReductiveStaticOutput : public IOutputContainer
    {
        MAPS_STATIC_ASSERT(LENGTH > 0, "Amount of items must be positive.");
        
        enum
        {
            ELEMENTS = ITEMS_PER_THREAD,
            SYNC_AFTER_INIT = true,
        };
        
        struct SharedData
        {
            T data[LENGTH];
        };

        // A shared copy of the array
        T *m_sdata;

        enum
        {
            // Round up LEN/BW
            LOOPS = ((LENGTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH),
            // Round down LEN/BW
            LEN_ROUND_DOWN = LENGTH / BLOCK_WIDTH,
            // Check if BW is an integral multiple of LEN
            IS_MULTIPLE = (LOOPS == LEN_ROUND_DOWN),
        };
        
        // Disallow copy
        __host__ __device__ void operator=(
            const ReductiveStaticOutput&) = delete;
        
        __device__ __forceinline__ void init(SharedData& sdata)
        {
            init_async(sdata);
            __syncthreads();
        }

        __device__ __forceinline__ void init_async(SharedData& sdata)
        {
            m_sdata = sdata.data;            
            
            if (threadIdx.y == 0)
            {
                // If block width is a multiple of length, the loop is simpler
                if (IS_MULTIPLE)
                {
                    #pragma unroll
                    for (int i = 0; i < LOOPS; ++i)
                        m_sdata[threadIdx.x + i * BLOCK_WIDTH] = 0;
                }
                else // Otherwise, do some checks
                {
                    #pragma unroll
                    for (int i = 0; i < LOOPS; ++i)
                    {
                        if ((threadIdx.x + i * BLOCK_WIDTH) < LENGTH)
                            m_sdata[threadIdx.x + i * BLOCK_WIDTH] = 0;
                    }
                }
            }
        }

        __device__ __forceinline__ void init_async_postsync()
        {
        }

        __device__ __forceinline__ int Items()
        {
            return ITEMS_PER_THREAD;
        }

        __device__ __forceinline__ void commit()
        {
            __syncthreads();

            if (threadIdx.y == 0)
            {
                // If block width is a multiple of length, the loop is simpler
                if (IS_MULTIPLE)
                {
                    #pragma unroll
                    for (int i = 0; i < LOOPS; ++i)
                        atomicAdd((T *)m_ptr + threadIdx.x + i * BLOCK_WIDTH, 
                                  m_sdata[threadIdx.x + i * BLOCK_WIDTH]);
                }
                else // Otherwise, do some checks
                {
                    #pragma unroll
                    for (int i = 0; i < LOOPS; ++i)
                    {
                        if ((threadIdx.x + i * BLOCK_WIDTH) < LENGTH)
                            atomicAdd((T *)m_ptr + threadIdx.x + 
                                      i * BLOCK_WIDTH, 
                                      m_sdata[threadIdx.x + i * BLOCK_WIDTH]);
                    }
                }
            }
        }

        __device__ __forceinline__ void commitAsync()
        {
            if (threadIdx.y == 0)
            {
                // If block width is a multiple of length, the loop is simpler
                if (IS_MULTIPLE)
                {
                    #pragma unroll
                    for (int i = 0; i < LOOPS; ++i)
                        atomicAdd((T *)m_ptr + threadIdx.x + i * BLOCK_WIDTH, 
                                  m_sdata[threadIdx.x + i * BLOCK_WIDTH]);
                }
                else // Otherwise, do some checks
                {
                    #pragma unroll
                    for (int i = 0; i < LOOPS; ++i)
                    {
                        if ((threadIdx.x + i * BLOCK_WIDTH) < LENGTH)
                            atomicAdd((T *)m_ptr + threadIdx.x + 
                                      i * BLOCK_WIDTH, 
                                      m_sdata[threadIdx.x + i * BLOCK_WIDTH]);
                    }
                }
            }
        }

        class item
        {
        protected:
            T *m_sdata;
        public:
            __device__ item(T *sdata) : m_sdata(sdata) {}
            __device__  __forceinline__ T operator+=(const T& other)
            {            
                return atomicAdd(m_sdata, other);
            }
            __device__  __forceinline__ T operator++() // Prefix
            {
                return atomicAdd(m_sdata, 1);
            }
            __device__  __forceinline__ T operator++(int) // Postfix
            {
                T res = *m_sdata;
                atomicAdd(m_sdata, 1);
                return res;
            }
            __device__  __forceinline__ T operator-=(const T& other)
            {
                return atomicSub(m_sdata, other);
            }
            __device__  __forceinline__ T operator--() // Prefix
            {
                return atomicSub(m_sdata, 1);
            }
            __device__  __forceinline__ T operator--(int) // Postfix
            {
                T res = *m_sdata;
                atomicSub(m_sdata, 1);
                return res;
            }
        };

        class iterator : public IOutputContainerIterator, 
                         public std::iterator<std::output_iterator_tag, T>
        {
        protected:
            ReductiveStaticOutput<T, LENGTH, BLOCK_WIDTH, 
                                  ITEMS_PER_THREAD>& m_parent;

        public:
            __device__ iterator(ReductiveStaticOutput<T, LENGTH, BLOCK_WIDTH, 
                                ITEMS_PER_THREAD>& parent, int ind) : 
                m_parent(parent) { m_pos = ind; }
            
            __device__ iterator(const iterator& other) : 
                m_parent(other.m_parent) { m_pos = other.m_pos; }

            __device__ __forceinline__ int index() const { return m_pos; }
            __device__ __forceinline__ item operator[](unsigned int index) {
                return item(m_parent.m_sdata + index);
            }
            __device__  __forceinline__ iterator& operator++() // Prefix
            {
                ++m_pos;
                return *this;
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

        __device__ __forceinline__ iterator begin()
        {
            return iterator(*this, 0);
        }

        __device__ __forceinline__ iterator end()
        {
            return iterator(*this, ITEMS_PER_THREAD);
        }
    };

}  // namespace maps

#endif  // __MAPS_REDUCTIVE_OUTPUT_CUH_
