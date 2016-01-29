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

#ifndef __MAPS_MULTI_ALLOCATOR_H
#define __MAPS_MULTI_ALLOCATOR_H

#include <vector>
#include <map>
#include <cuda_runtime.h>

#include "../internal/cuda_utils.hpp"
#include "common.h"

namespace maps
{
    namespace multi
    {

        class IAllocator
        {
        public:
            virtual void DisablePitchedAllocation() = 0;
            virtual bool PitchedAllocationEnabled() const = 0;

            virtual void *Allocate(unsigned int gpuID, const DatumSegment& segment, size_t elementSize,
                                   size_t& strideBytes) = 0;
            virtual bool Deallocate(unsigned int gpuID, void *datum) = 0;
        };

        /**
        * @brief A simple CUDA allocator
        */
        class SimpleAllocator : public IAllocator
        {
        protected:
            bool m_bPitchedAllocation;

            /// Allocated buffers, ordered by device ID
            std::map< unsigned int, std::vector<void *> > m_allocatedBuffers;

        public:
            SimpleAllocator() : m_bPitchedAllocation(true) {}

            virtual void DisablePitchedAllocation() override
            {
                m_bPitchedAllocation = false;
            }

            virtual bool PitchedAllocationEnabled() const override
            {
                return m_bPitchedAllocation;
            }

            virtual void *Allocate(unsigned int gpuID, const DatumSegment& segment, size_t elementSize,
                                   size_t& strideBytes)
            {
                MAPS_CUDA_CHECK(cudaSetDevice((int)gpuID));

                void *ret = nullptr;
                strideBytes = 0;

                if (segment.GetDimensions() == 1)
                {
                    MAPS_CUDA_CHECK(cudaMalloc(&ret, elementSize * segment.GetDimension(0)));

                    if (ret)
                    {
                        m_allocatedBuffers[gpuID].push_back(ret);
                        strideBytes = elementSize * segment.GetDimension(0);
                    }
                    return ret;
                }
                else if (segment.GetDimensions() > 1)
                {
                    size_t otherDimensions = 1;
                    for (unsigned int i = 1; i < segment.GetDimensions(); ++i)
                        otherDimensions *= segment.GetDimension(i);

                    if (m_bPitchedAllocation)
                        MAPS_CUDA_CHECK(cudaMallocPitch(&ret, &strideBytes, elementSize * segment.GetDimension(0), otherDimensions));
                    else
                    {
                        MAPS_CUDA_CHECK(cudaMalloc(&ret, elementSize * segment.GetDimension(0) * otherDimensions));
                        strideBytes = elementSize * segment.GetDimension(0);
                    }

                    if (ret)
                        m_allocatedBuffers[gpuID].push_back(ret);
                    return ret;
                }
                else
                {
                    printf("Zero dimensions are not allowed\n");
                    return nullptr;
                }
            }

            virtual bool Deallocate(unsigned int gpuID, void *datum)
            {
                MAPS_CUDA_CHECK(cudaSetDevice((int)gpuID));
                MAPS_CUDA_CHECK(cudaFree(datum));

                // Remove from allocated buffer list
                for (auto iter = m_allocatedBuffers[gpuID].begin(),
                     end = m_allocatedBuffers[gpuID].end(); iter != end; ++iter)
                {
                    if (*iter == datum)
                    {
                        m_allocatedBuffers[gpuID].erase(iter);
                        break;
                    }
                }
                return true;
            }

            virtual ~SimpleAllocator()
            {
                // Free all buffers
                for (auto&& dev : m_allocatedBuffers)
                {
                    MAPS_CUDA_CHECK(cudaSetDevice((int)dev.first));
                    for (const auto& datum : dev.second)
                    {
                        MAPS_CUDA_CHECK(cudaFree(datum));
                    }
                    dev.second.clear();
                }
            }
        };

    } // namespace multi

} // namespace maps

#endif // __MAPS_MULTI_ALLOCATOR_H
