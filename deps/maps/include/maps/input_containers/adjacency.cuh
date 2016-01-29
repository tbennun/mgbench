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

#ifndef __MAPS_ADJACENCY_CUH_
#define __MAPS_ADJACENCY_CUH_

#include "../internal/common.h"
#include "internal/io_common.cuh"
#include "internal/io_globalread.cuh"

namespace maps
{
    template <typename TWeight, typename TValue>
    class Adjacency : public IContainerComposition
    {
    public:
        uint2*            m_nodeEdgesCountOffsetMap;
        maps::TypedInputContainer<TWeight>        m_edgesWeightList;
        TValue*            m_nodesValueList;

        uint2*            m_blockEdgesCountOffsetMap;
        maps::TypedInputContainer<unsigned int>    m_blockEdgesIndexList;
        int*            m_nodeNodesSMemIndexList;

    private:
        unsigned int m_numNodesRoundUp;
        unsigned int m_maxNumOfEdgesInBlock;
        size_t m_maxNodeRank;
        size_t m_dsmemOffset; // dynamic shared memory offset

        uint2 m_edge_count_offset_pair;
        TValue *m_sdata;

    public:
        enum
        {
            SYNC_AFTER_INIT = true,
        };

        typedef SharedMemoryArray <TValue, DYNAMIC_SMEM> SharedData;

        dim3 grid_dims; // Actual grid dimensions, for block index computation
        unsigned int block_offset;
        uint3 blockId;

        __host__ __device__ Adjacency(
          uint2 *nodeEdgesCountOffsetMap,
          const TypedInputContainer<TWeight>& edgesWeightList,
          TValue *nodesValueList,
          uint2 *blockEdgesCountOffsetMap,
          const TypedInputContainer<unsigned int>& blockEdgesIndexList,
          int *nodeNodesSMemIndexList,
          unsigned int numNodesRoundUp,
          unsigned int maxNumOfEdgesInBlock,
          size_t maxNodeRank, 
          size_t dsmemOffset) 
            :
          m_nodeEdgesCountOffsetMap(nodeEdgesCountOffsetMap),
          m_edgesWeightList(edgesWeightList),
          m_nodesValueList(nodesValueList),
          
          m_blockEdgesCountOffsetMap(blockEdgesCountOffsetMap),
          m_blockEdgesIndexList(blockEdgesIndexList),
          m_nodeNodesSMemIndexList(nodeNodesSMemIndexList),
          
          m_numNodesRoundUp(numNodesRoundUp),
          m_maxNumOfEdgesInBlock(maxNumOfEdgesInBlock),
          m_maxNodeRank(maxNodeRank),
          m_dsmemOffset(dsmemOffset)
        {

        }

        __device__ __forceinline__ void init_async(SharedData& sdata)
        {
            if (block_offset > 0)
            {
                unsigned int __realBlockIdx;
                asm("mov.b32   %0, %ctaid.x;" : "=r"(__realBlockIdx));
                __realBlockIdx += block_offset;
              
                blockId.x = __realBlockIdx % grid_dims.x;
                blockId.y = (__realBlockIdx / grid_dims.x) % grid_dims.y;
                blockId.z = ((__realBlockIdx / grid_dims.x) / grid_dims.y);
            }
            else
                blockId = blockIdx;
            
            sdata.init(m_dsmemOffset);
            m_sdata = sdata.smem;
            
            // Load nodes values according to pre-processed info
            loadGraphDataToSharedMem();    
          }

          __device__ __forceinline__ void init_async_postsync()
          {
              // TODO: Operate w.r.t. 3D blocks
              // Load edge info
              m_edge_count_offset_pair = 
                m_nodeEdgesCountOffsetMap[threadIdx.x + blockIdx.x*blockDim.x];
          }

          __device__ __forceinline__ void init(SharedData& sdata)
          {
              init_async(sdata);
              __syncthreads();
              init_async_postsync();
          }

          __device__ __forceinline__ void loadGraphDataToSharedMem()
          {
              // First load all needed data to shared memory and then use them
              int loadIndex = threadIdx.x;

              uint2 numOfItemsForBlock = m_blockEdgesCountOffsetMap[blockIdx.x];
              
              while (loadIndex < numOfItemsForBlock.x)
              {

                  int gmemIndex;
                  GlobalRead<int, GR_DISTINCT>::Read1D(
                      (int *)m_blockEdgesIndexList.GetTypedPtr(),  
                      -m_blockEdgesIndexList.GetOffset() + 
                      numOfItemsForBlock.y + loadIndex, gmemIndex);
                  GlobalRead<TValue, GR_DISTINCT>::Read1D(
                      m_nodesValueList, gmemIndex, m_sdata[loadIndex]);

                  loadIndex += blockDim.x;
              }
          }

          struct edge_data
          {
              TWeight edge_weight;
              TValue adjacent_node_value;
          };

          class iterator
          {
          private:
            TWeight* m_edgesWeightListPtr;
            int64_t m_edgesWeightListOffset;
            unsigned int m_index;
            
            unsigned int m_gmemIndex;
            const int* m_nodeNodesSMemIndexList;
            
            TValue *m_sdata;
            
          public:
            __device__ iterator() { };
            
            __device__ __forceinline__ iterator(const Adjacency *adjacency, 
                                                const unsigned int index)
                : m_index(index)
            {
                m_edgesWeightListPtr = adjacency->m_edgesWeightList.GetTypedPtr();
                m_edgesWeightListOffset = adjacency->m_edgesWeightList.GetOffset();

                m_gmemIndex = blockIdx.x * blockDim.x * adjacency->m_maxNodeRank + threadIdx.x;
                m_nodeNodesSMemIndexList = adjacency->m_nodeNodesSMemIndexList;

                m_sdata = adjacency->m_sdata;
            }

            __device__ __forceinline__ void next()
            {
                m_gmemIndex += blockDim.x;
                ++m_index;
            }

            __device__ __forceinline__ edge_data operator* ()
            {
                edge_data edgeData;

                // Load shared memory index from the preprocessed global memory
                int smemIndex;
                GlobalRead<int, GR_DISTINCT>::Read1D(
                    m_nodeNodesSMemIndexList, m_gmemIndex, smemIndex);

                // Load the adjacent node value from smem
                edgeData.adjacent_node_value = m_sdata[smemIndex];

                // Load the edge weight from global memory
                edgeData.edge_weight = m_edgesWeightListPtr[m_index - m_edgesWeightListOffset];

                return edgeData;
            }

            __device__ __forceinline__ iterator operator++() // Prefix
            {
                next();
                return *this;
            }

            __device__ __forceinline__ iterator operator++(int) // Postfix
            {
                iterator temp(*this);
                next();
                return temp;
            }

            __device__ __forceinline__ void operator=(const iterator &a)
            {
                m_edgesWeightListPtr = a.m_edgesWeightListPtr;
                m_edgesWeightListOffset = a.m_edgesWeightListOffset;
                m_index = a.m_index;

                m_gmemIndex = a.m_gmemIndex;
                m_nodeNodesSMemIndexList = a.m_nodeNodesSMemIndexList;

                m_sdata = a.m_sdata;
            }

            __device__ __forceinline__ bool operator==(const iterator &a)
            {
                return m_index == a.m_index;
            }

            __device__ __forceinline__ bool operator!=(const iterator &a)
            {
                return m_index != a.m_index;
            }

        }; // end of class iterator

        /**
        * @brief Creates a thread-level iterator that points to the beginning 
        * of the current chunk.
        * @return Thread-level iterator.
        */
        __device__ __forceinline__ iterator begin() const
        {
            return iterator(this, m_edge_count_offset_pair.y);
        }

        /**
        * @brief Creates a thread-level iterator that points to the end of the 
        * current chunk.
        * @return Thread-level iterator.
        */
        __device__ __forceinline__ iterator end() const
        {
            return iterator(this, m_edge_count_offset_pair.y + 
                                  m_edge_count_offset_pair.x);
        }

        // TODO(later): aligned iterators


        /**
        * @brief Progresses to process the next chunk (does nothing).
        */
        __device__ __forceinline__ void nextChunk() { }

        /**
        * Progresses to process the next chunk without calling __syncthreads()
        * (does nothing).
        * @note This is an advanced function that should be used carefully.
        */
        __device__ __forceinline__ void nextChunkAsync() { }

        /**
        * @brief Returns false if there are more chunks to process.
        */
        __device__ __forceinline__ bool isDone() { return true; }

        /**
        * @brief Returns the total number of chunks, or 0 for dynamic chunks.
        */
        __device__ __forceinline__ int chunks() { return 1; }
    };

}  // namespace maps

#endif  // __MAPS_ADJACENCY_CUH_
