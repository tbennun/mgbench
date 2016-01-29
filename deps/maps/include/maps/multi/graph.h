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

#ifndef __MAPS_MULTI_GRAPH_H
#define __MAPS_MULTI_GRAPH_H

#include <vector>
#include <memory>
#include <algorithm>

#include <cuda_runtime.h>
#include "../internal/common.h"

namespace maps
{
    namespace multi
    {
        /**
        * @brief Represents a simple pure graph entity
        *        TWeight: the type edge weight
        *        TValue: the type of value attached to each node
        *        
        * TODO (later): both template args should support also void types (through specialization) 
        */
        template<typename TWeight, typename TValue>
        class Graph
        {
            // Holds a general representation of a Graph

        public:
            class Node;
            class Edge;

        private:
            bool m_symmetric;

            std::vector<Node> m_nodeList;
            std::vector<Edge> m_edgeList;

        public:
            explicit Graph(bool symmetric = false) : m_symmetric(symmetric)
            {
            }

            bool IsSymmetric() { return m_symmetric; }

            const std::vector<Node>& GetNodeList() { return m_nodeList; }
            const std::vector<Edge>& GetEdgeList() { return m_edgeList; }

            size_t AddNode(const TValue& val)
            {
                Node node((unsigned int)m_nodeList.size(), val);
                m_nodeList.push_back(node);
                return m_nodeList.size() - 1;
            }

            void AddNodes(const std::vector<TValue>& values)
            {
                for (auto& val : values)
                    AddNode(val);
            }

            void AddNodes(unsigned int nodes)
            {
                for (unsigned int i = 0; i < nodes; ++i)
                    AddNode(TValue(0));
            }

            /**
            * adds the option of excluding specific nodes from being
            * processed
            */
            void ExcludeNode(const unsigned int nodeInd)
            {
                m_nodeList[nodeInd]._exclude = true;
            }

            /**
            * This function connects to nodes in the graph with an edge
            * Calling this function in the user code were he builds his
            * data can be an easy way to build the topology
            */
            int AddEdge(unsigned int node1Ind, unsigned int node2Ind, const TWeight& val)
            {
                // We increment edge count in the symmetric case before we get the index
                // to support negative edges (e.g. 1,-1)

                int edgeInd = (int)m_edgeList.size();

                Edge edge(node1Ind, node2Ind, edgeInd, val);
                m_edgeList.push_back(edge);
                m_nodeList[node1Ind].addEdge(edgeInd);

                return edgeInd;
            }

            class Edge
            {
            public:
                int _node1Index;
                int _node2Index;
                int _index;
                TWeight _weight;

                Edge(unsigned int node1, unsigned int node2, unsigned int index, const TWeight& weight)
                {
                    _node1Index = node1;
                    _node2Index = node2;
                    _index = index;
                    _weight = weight; // copy weight
                }
            };

            class Node
            {
            public:
                std::vector<int> _edgeIndexList;
                int _index;
                bool _exclude;
                TValue _value;

                Node(unsigned int index, const TValue& val)
                {
                    _index = index;
                    _exclude = false;
                    _value = val; // copy val
                }

                void addEdge(const int edgeInd)
                {
                    _edgeIndexList.push_back(edgeInd);
                }
            };
        };

    } // namespace multi
}  // namespace maps

#endif  // __MAPS_MULTI_GRAPH_H
