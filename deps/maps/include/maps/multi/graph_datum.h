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

#ifndef __MAPS_MULTI_GRAPH_DATUM_H
#define __MAPS_MULTI_GRAPH_DATUM_H

#include <cuda_runtime.h>
#include <memory>
#include <vector>

#include "graph.h"
#include <unordered_set>

namespace maps
{
    namespace multi
    {
        /**
        * @brief Represents a virtual Datum which is actually composed of other real Datum's
        */
        struct IVirtualDatum
        {
            virtual std::vector< std::shared_ptr<IDatum> > GetDatumObjects() = 0;
        };

        enum GraphPattern
        {
            ADJACENCY, /*BFS, DFS*/
        };

        template <typename TWeight, typename TValue, int BLOCK_SIZE, GraphPattern TPattern>
        class GraphDatum;

        // Forward declaration
        template <typename TWeight, typename TValue, int BLOCK_SIZE>
        class Adjacency;

        template <typename TWeight, typename TValue, int BLOCK_SIZE>
        class GraphDatum<TWeight, TValue, BLOCK_SIZE, ADJACENCY> : public IVirtualDatum
        {
        private:
            friend class Adjacency<TWeight, TValue, BLOCK_SIZE>;

            Graph<TWeight, TValue> *m_graph;
            bool m_preprocessingRequired;

            // Graph CSR arrays
            std::vector<uint2>    m_nodeEdgesCountOffsetMap;
            std::vector<int>    m_edgesIndexList;

            // Graph edge weight and node value lists
            std::vector<TWeight>        m_edgesWeightList;
            std::vector<TValue>            m_nodesValueList;

            // Per block pre-processed data
            std::vector<uint2>    m_blockEdgesCountOffsetMap;
            std::vector<unsigned int>    m_blockEdgesIndexList;
            std::vector<int>    m_nodeNodesSMemIndexList;

            // Graph CSR datum objects
            std::shared_ptr< maps::multi::Vector<uint2> >    m_nodeEdgesCountOffsetMapDatum;
            std::shared_ptr< maps::multi::Vector<int> >        m_edgesIndexListDatum;
            std::shared_ptr< maps::multi::Vector<TWeight> >        m_edgesWeightListDatum;
            std::shared_ptr< maps::multi::Vector<TValue> >        m_nodesValueListDatum;

            // Per block pre-processed datum objects
            std::shared_ptr< maps::multi::Vector<uint2>    >            m_blockEdgesCountOffsetMapDatum;
            std::shared_ptr< maps::multi::Vector<unsigned int> >    m_blockEdgesIndexListDatum;
            std::shared_ptr< maps::multi::Vector<int> >                m_nodeNodesSMemIndexListDatum;

            unsigned int m_numNodesRoundUp;
            size_t m_maxNodeRank;
            unsigned int m_maxNumOfEdgesInBlock; // Used for calculating the dynamic shared memory size

            void LoadCSRArrays()
            {
                // Reset CSR arrays

                m_nodeEdgesCountOffsetMap.clear();
                m_edgesIndexList.clear();
                m_edgesWeightList.clear();
                m_nodesValueList.clear();

                // Reset max rank
                m_maxNodeRank = 0;

                // Build the CSR representation

                auto& nodeList = m_graph->GetNodeList();
                auto& edgeList = m_graph->GetEdgeList();

                m_nodeEdgesCountOffsetMap.resize(nodeList.size());
                unsigned int count = 0;

                // Compose a list of edges for each node
                for (unsigned int nodeIndex = 0; nodeIndex < nodeList.size(); nodeIndex++)
                {
                    auto& node = nodeList[nodeIndex];
                    m_nodesValueList.push_back(node._value);

                    if (node._exclude == true)
                    {
                        m_nodeEdgesCountOffsetMap[nodeIndex].x = 0;
                        m_nodeEdgesCountOffsetMap[nodeIndex].y = count;
                    }
                    else
                    {
                        m_maxNodeRank = std::max(m_maxNodeRank, node._edgeIndexList.size());

                        m_nodeEdgesCountOffsetMap[nodeIndex].x = node._edgeIndexList.size();
                        m_nodeEdgesCountOffsetMap[nodeIndex].y = count;

                        count += node._edgeIndexList.size();

                        for (unsigned int edgeIndex = 0; edgeIndex < node._edgeIndexList.size(); edgeIndex++)
                        {
                            int ind = node._edgeIndexList[edgeIndex];

                            m_edgesIndexList.push_back(edgeList[ind]._node2Index);
                            m_edgesWeightList.push_back(edgeList[ind]._weight);
                        }
                    }
                }

                m_nodeEdgesCountOffsetMapDatum = std::make_shared< maps::multi::Vector<uint2> >(m_nodeEdgesCountOffsetMap.size());
                m_edgesIndexListDatum = std::make_shared< maps::multi::Vector<int> >(m_edgesIndexList.size());
                m_edgesWeightListDatum = std::make_shared< maps::multi::Vector<TWeight> >(m_edgesWeightList.size());
                m_nodesValueListDatum = std::make_shared< maps::multi::Vector<TValue> >(m_nodesValueList.size());

                m_nodeEdgesCountOffsetMapDatum->Bind(&m_nodeEdgesCountOffsetMap[0]);
                m_edgesIndexListDatum->Bind(&m_edgesIndexList[0]);
                m_edgesWeightListDatum->Bind(&m_edgesWeightList[0]);
                m_nodesValueListDatum->Bind(&m_nodesValueList[0]);
            }

            void PreprocessData()
            {
                auto& nodeList = m_graph->GetNodeList();
                auto& edgeList = m_graph->GetEdgeList();

                unsigned int nodeCount = nodeList.size();

                unsigned int blockSize = BLOCK_SIZE;
                unsigned int numBlocks = RoundUp(nodeCount, blockSize);

                unsigned int maxOverlap = 0;
                unsigned int totalOverLap = 0;
                unsigned int overlapCount = 0;
                unsigned int offset = 0;

                m_maxNumOfEdgesInBlock = 0;
                m_numNodesRoundUp = RoundUp(nodeCount, blockSize) * blockSize;

                // clear and reset
                m_blockEdgesIndexList.clear();
                m_blockEdgesCountOffsetMap.resize(numBlocks);
                m_nodeNodesSMemIndexList.resize(m_numNodesRoundUp * m_maxNodeRank);

                struct side_1_edge_info
                {
                    unsigned int node_1_index;
                    unsigned int node_1_edge_index;

                    side_1_edge_info(
                        unsigned int node_1_index,
                        unsigned int node_1_edge_index) : node_1_index(node_1_index), node_1_edge_index(node_1_edge_index)
                    { }
                };

                struct edge_info {
                    unsigned int node_2_index;
                    mutable std::vector< side_1_edge_info > side_1_edges;

                    edge_info(unsigned int node_2_index) : node_2_index(node_2_index), side_1_edges()
                    { }
                };

                struct edge_info_hash {
                    size_t operator()(const edge_info& info) const
                    {
                        std::hash<unsigned int> h;
                        return h(info.node_2_index);
                    }
                };

                struct edge_info_equal_to
                {    
                    bool operator()(const edge_info& _Left, const edge_info& _Right) const
                    {    
                        return (_Left.node_2_index == _Right.node_2_index);
                    }
                };

                for (unsigned int nodeId = 0, blockId = 0; blockId < numBlocks; blockId++)
                {
                    std::unordered_set<
                        edge_info, 
                        edge_info_hash, 
                        edge_info_equal_to > blockNodesSet;

                    unsigned int overlap = 0;
                    unsigned int blockStartNodeId = nodeId;

                    for (; nodeId < (blockId + 1) * blockSize; ++nodeId)
                    {
                        if (nodeId < nodeCount)
                        {
                            auto& node = nodeList[nodeId];
                            if (node._exclude != true)
                            {
                                unsigned int i = 0;
                                for (auto edgeIndex : node._edgeIndexList)
                                {
                                    unsigned int node2Index = abs(edgeList[edgeIndex]._node2Index);
                                    auto res = blockNodesSet.insert(edge_info(node2Index));
                                    if (!res.second) ++overlap;

                                    res.first->side_1_edges.emplace_back(nodeId, i);
                                    ++i;
                                }
                            }
                        }
                    } // end of block loop

                    m_maxNumOfEdgesInBlock = std::max(blockNodesSet.size(), (size_t) m_maxNumOfEdgesInBlock);
                    maxOverlap = std::max(overlap, maxOverlap);
                    totalOverLap += overlap;
                    overlapCount++;

                    m_blockEdgesCountOffsetMap[blockId].x = blockNodesSet.size();
                    m_blockEdgesCountOffsetMap[blockId].y = offset;

                    offset += blockNodesSet.size();
                    m_blockEdgesIndexList.reserve(m_blockEdgesIndexList.size() + blockNodesSet.size());

                    unsigned int e = 0;
                    for (auto& edgeInfo : blockNodesSet) 
                    {
                        m_blockEdgesIndexList.push_back(edgeInfo.node_2_index);

                        // Prepare internal shared memory indexes for the constraints connected to a particle
                        for (auto& edge_side_1 : edgeInfo.side_1_edges)
                        {
                            unsigned int threadIndex = edge_side_1.node_1_index - blockStartNodeId;
                            unsigned int blockStart = blockId * blockSize * m_maxNodeRank;

                            m_nodeNodesSMemIndexList[blockStart + edge_side_1.node_1_edge_index* blockSize + threadIndex] = e;
                        }

                        ++e;
                    }
                }

                m_blockEdgesCountOffsetMapDatum = std::make_shared< maps::multi::Vector<uint2> >(m_blockEdgesCountOffsetMap.size());
                m_blockEdgesIndexListDatum = std::make_shared< maps::multi::Vector<unsigned int> >(m_blockEdgesIndexList.size());
                m_nodeNodesSMemIndexListDatum = std::make_shared< maps::multi::Vector<int> >(m_nodeNodesSMemIndexList.size());

                m_blockEdgesCountOffsetMapDatum->Bind(&m_blockEdgesCountOffsetMap[0]);
                m_blockEdgesIndexListDatum->Bind(&m_blockEdgesIndexList[0]);
                m_nodeNodesSMemIndexListDatum->Bind(&m_nodeNodesSMemIndexList[0]);
            }

            void InvalidatePreprocessing()
            {
                if (!m_preprocessingRequired) return;

                LoadCSRArrays();
                PreprocessData();

                m_preprocessingRequired = false;
            }

        public:
            explicit GraphDatum(Graph<TWeight, TValue>& dat) : m_graph(&dat), m_preprocessingRequired(true), m_maxNodeRank(0)
            {
                InvalidatePreprocessing();
            }

            void Invalidate()
            {
                InvalidatePreprocessing();
            }

            size_t RequiredDynamicSMemSize() const
            {
                return sizeof(TValue) * m_maxNumOfEdgesInBlock;
            }

            virtual std::vector< std::shared_ptr<IDatum> > GetDatumObjects() override
            {
                std::vector< std::shared_ptr<IDatum> > data;
                data.reserve(7);

                data.push_back(m_nodeEdgesCountOffsetMapDatum);
                data.push_back(m_edgesIndexListDatum);
                data.push_back(m_edgesWeightListDatum);
                data.push_back(m_nodesValueListDatum);

                data.push_back(m_blockEdgesCountOffsetMapDatum);
                data.push_back(m_blockEdgesIndexListDatum);
                data.push_back(m_nodeNodesSMemIndexListDatum);

                return std::move(data);
            }
        };

    } // namespace multi

} // namespace maps

#endif // __MAPS_MULTI_GRAPH_DATUM_H
