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

#ifndef __MAPS_MULTI_GRAPH_CONTAINERS_H
#define __MAPS_MULTI_GRAPH_CONTAINERS_H

#include <cuda_runtime.h>
#include <iterator>
#include <iostream>
#include <memory>
#include <vector>

#include "graph_datum.h"
#include "input_containers.h"
#include "../input_containers/adjacency.cuh"

namespace maps
{
    namespace multi
    {
        //////////////////////////////////////////////////////////////////////////////////////////////
        // ADJACENCY INPUT PATTERN
        //////////////////////////////////////////////////////////////////////////////////////////////

        template <typename TWeight, typename TValue, int BLOCK_SIZE>
        class Adjacency
        {
        private:
            GraphDatum<TWeight, TValue, BLOCK_SIZE, ADJACENCY> *m_datum;

            enum SegmentationUnit
            {
                BLOCK, THREAD
            };

            /**
            * @brief Distributes a fixed number of input items to each unit (block / thread)
            */
            template<typename T, SegmentationUnit UNIT>
            class InjectiveInput
            {
            private:
                int m_itemsPerUnit;

            public:
                Datum<T, 1> *datum;

                InjectiveInput(Datum<T, 1>& dat, int itemsPerUnit = 1) : m_itemsPerUnit(itemsPerUnit), datum(&dat) {}

                class Segmenter : public ISegmenter
                {
                private:
                    int m_itemsPerUnit;

                public:
                    Datum<T, 1> *datum;
                    Segmenter(Datum<T, 1> *inDatum, int itemsPerUnit) : m_itemsPerUnit(itemsPerUnit), datum(inDatum) {}

                    virtual unsigned int ItemsPerThread(unsigned int dimension) const override
                    {
                        // not supported for block unit, 
                        // seems like function is not called for Input containers

                        switch (dimension)
                        {
                        default:
                            return UNIT == THREAD ? 1 : 0;
                        case 0:
                            return UNIT == THREAD ? m_itemsPerUnit : 0;
                        }
                    }

                    virtual void Segment(const std::vector<GridSegment>& segments,
                        std::vector< std::vector<DatumSegment> >& out_data_segments) const override
                    {
                        out_data_segments.reserve(segments.size());

                        for (const auto& seg : segments)
                        {
                            // Compute 1D offset
                            size_t xoff_begin = seg.offset % seg.total_grid_dims.x;
                            size_t xoff_end = (seg.offset + seg.blocks - 1) % seg.total_grid_dims.x;

                            // The list is split to memory-contiguous regions (for boundary conditions)
                            std::vector<DatumSegment> datasegs;

                            // Prepare the main data segment:
                            DatumSegment dataseg(1);
                            dataseg.m_borders = WB_NOCHECKS;

                            dataseg.m_offset[0] = 
                                (int64_t)xoff_begin * 
                                (UNIT == THREAD ? seg.block_dims.x : 1) * 
                                m_itemsPerUnit;

                            dataseg.m_dimensions[0] = 
                                (xoff_end - xoff_begin + 1) * 
                                (UNIT == THREAD ? seg.block_dims.x : 1) * 
                                m_itemsPerUnit;

                            // Multiply dimensions by m_itemsPerUnit
                            // (they are divided in segmentation)
                            if (datum->GetDataDimension(0) % m_itemsPerUnit != 0)
                            {
                                printf("WARNING: Width is not a multiple of items per unit\n");
                            }

                            // Avoid unnecessary overflows due to block size
                            for (int d = 0; d < 1; ++d)
                            {
                                if (dataseg.m_offset[d] + dataseg.m_dimensions[d] > datum->GetDataDimension(d))
                                {
                                    dataseg.m_dimensions[d] = datum->GetDataDimension(d) - dataseg.m_offset[d];
                                }
                            }

                            datasegs.push_back(dataseg);
                            out_data_segments.push_back(datasegs);
                        }
                    }
                };

                class ContainerFactory : public IContainerFactory
                {
                public:
                    virtual std::shared_ptr<::maps::IInputContainer>
                        CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg,
                        const GridSegment& grid_seg) const override
                    {
                        typedef maps::TypedInputContainer<T> ContainerType;
                        ContainerType *cont = new ContainerType(seg.m_offset[0]);
                        cont->m_ptr = buff.ptr;

                        return std::shared_ptr<::maps::IInputContainer>(cont);
                    }
                };

                std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum, m_itemsPerUnit)); }
                std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }
            };

            /**
            * @brief Distributes the input items based on per unit (block / thread) mapping
            */
            template<typename T, SegmentationUnit UNIT>
            class MappedInput
            {
            private:
                const uint2* m_perUnitMapping; // x -> count, y -> offset

            public:
                Datum<T, 1> *datum;

                MappedInput(Datum<T, 1>& dat, const uint2* mapping) : m_perUnitMapping(mapping), datum(&dat) {}

                class Segmenter : public ISegmenter
                {
                private:
                    const uint2* m_perUnitMapping; // x -> count, y -> offset

                public:
                    Datum<T, 1> *datum;
                    Segmenter(Datum<T, 1> *inDatum, const uint2* mapping) : m_perUnitMapping(mapping), datum(inDatum) {}

                    virtual unsigned int ItemsPerThread(unsigned int dimension) const override
                    {
                        return 1;
                    }

                    virtual void Segment(const std::vector<GridSegment>& segments,
                        std::vector< std::vector<DatumSegment> >& out_data_segments) const override
                    {
                        out_data_segments.reserve(segments.size());

                        for (const auto& seg : segments)
                        {
                            // Compute 1D block offset
                            size_t begin_block = seg.offset % seg.total_grid_dims.x;
                            size_t end_block = (seg.offset + seg.blocks - 1) % seg.total_grid_dims.x;

                            // mapping size should match the grid dims (according to UNIT)

                            size_t begin = m_perUnitMapping[begin_block * (UNIT == THREAD ? seg.block_dims.x : 1)].y;
                            size_t end = // excluded index
                                (end_block + 1) >= seg.total_grid_dims.x
                                ? datum->GetDataDimension(0)
                                : m_perUnitMapping[(end_block + 1) * (UNIT == THREAD ? seg.block_dims.x : 1)].y;

                            // The list is split to memory-contiguous regions (for boundary conditions)
                            std::vector<DatumSegment> datasegs;

                            // Prepare the main data segment:
                            DatumSegment dataseg(1);
                            dataseg.m_borders = WB_NOCHECKS;

                            dataseg.m_offset[0] = (int64_t) begin;
                            dataseg.m_dimensions[0] = (int64_t)(end - begin);

                            datasegs.push_back(dataseg);
                            out_data_segments.push_back(datasegs);
                        }
                    }
                };

                class ContainerFactory : public IContainerFactory
                {
                public:
                    virtual std::shared_ptr<::maps::IInputContainer>
                        CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg,
                        const GridSegment& grid_seg) const override
                    {
                        typedef maps::TypedInputContainer<T> ContainerType;
                        ContainerType *cont = new ContainerType(seg.m_offset[0]);
                        cont->m_ptr = buff.ptr;

                        return std::shared_ptr<::maps::IInputContainer>(cont);
                    }
                };

                std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum, m_perUnitMapping)); }
                std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }
            };

            template<typename TContainer>
            static inline void AppendInputArgToTask(Task& task, const TContainer& arg)
            {
                task.inputs.push_back(TaskInput(arg.datum, arg.CreateSegmenter(), arg.CreateContainerFactory()));
                task.argument_ordering.push_back(AT_INPUT);
            }

        public:
            explicit Adjacency(GraphDatum<TWeight, TValue, BLOCK_SIZE, ADJACENCY>& dat) : m_datum(&dat)
            {
            }

            inline void AppendArgsToTask(Task& task) const
            {
                size_t dsmemOffset = task.dsmem;
                task.dsmem += m_datum->RequiredDynamicSMemSize();

                InjectiveInput<uint2, THREAD>    nodeEdgesCountOffsetMapInput(*m_datum->m_nodeEdgesCountOffsetMapDatum);
                MappedInput<TWeight, THREAD>        edgesWeightListInput(*m_datum->m_edgesWeightListDatum, (const uint2*)m_datum->m_nodeEdgesCountOffsetMapDatum->HostPtr());
                maps::multi::IrregularInput<TValue, 1>        nodesValueListInput(*m_datum->m_nodesValueListDatum);

                InjectiveInput<uint2, BLOCK>    blockEdgesCountOffsetMapInput(*m_datum->m_blockEdgesCountOffsetMapDatum);
                MappedInput<unsigned int, BLOCK>    blockEdgesIndexListInput(*m_datum->m_blockEdgesIndexListDatum, (const uint2*)m_datum->m_blockEdgesCountOffsetMapDatum->HostPtr());
                InjectiveInput<int, THREAD>        nodeNodesSMemIndexListInput(*m_datum->m_nodeNodesSMemIndexListDatum, m_datum->m_maxNodeRank);

                int index = (int)task.argument_ordering.size();

                AppendInputArgToTask(task, nodeEdgesCountOffsetMapInput);
                AppendInputArgToTask(task, edgesWeightListInput);
                AppendInputArgToTask(task, nodesValueListInput);

                AppendInputArgToTask(task, blockEdgesCountOffsetMapInput);
                AppendInputArgToTask(task, blockEdgesIndexListInput);
                AppendInputArgToTask(task, nodeNodesSMemIndexListInput);

                int count = (int)task.argument_ordering.size() - index;
                task.reducers.push_back(
                    std::make_tuple(index, count, 
                    std::static_pointer_cast<maps::multi::IContainerReducer>(std::make_shared<ContainerReducer>(
                        m_datum->m_numNodesRoundUp, 
                        m_datum->m_maxNumOfEdgesInBlock, 
                        m_datum->m_maxNodeRank,
                        dsmemOffset
                        )
                        )));
            }

            class ContainerReducer : public IContainerReducer
            {
            private:
                unsigned int m_numNodesRoundUp;
                unsigned int m_maxNumOfEdgesInBlock;
                size_t m_maxNodeRank;
                size_t m_dsmemOffset; // dynamic shared memory offset

            public:
                ContainerReducer(unsigned int numNodesRoundUp, unsigned int maxNumOfEdgesInBlock, size_t maxNodeRank, size_t dsmemOffset)
                    : m_numNodesRoundUp(numNodesRoundUp), m_maxNumOfEdgesInBlock(maxNumOfEdgesInBlock), m_maxNodeRank(maxNodeRank), m_dsmemOffset(dsmemOffset)
                { }

                virtual std::shared_ptr<::maps::IContainerComposition>
                    ComposeContainers(const std::vector<void*>& containers, const GridSegment& grid_seg) const override
                {
                    // keep synced with code above

                    maps::TypedInputContainer<uint2>*    nodeEdgesCountOffsetMapContainer = (maps::TypedInputContainer<uint2>*) containers[0];
                    maps::TypedInputContainer<TWeight>*        edgesWeightListContainer = (maps::TypedInputContainer<TWeight>*) containers[1];
                    maps::IrregularInput<TValue, 1>*        nodesValueListContainer            = (maps::IrregularInput<TValue, 1>*) containers[2];

                    maps::TypedInputContainer<uint2>*            blockEdgesCountOffsetMapContainer = (maps::TypedInputContainer<uint2>*) containers[3];
                    maps::TypedInputContainer<unsigned int>*    blockEdgesIndexListContainer = (maps::TypedInputContainer<unsigned int>*) containers[4];
                    maps::TypedInputContainer<int>*            nodeNodesSMemIndexListContainer = (maps::TypedInputContainer<int>*) containers[5];

                    auto adjacency = new maps::Adjacency<TWeight, TValue>(
                        nodeEdgesCountOffsetMapContainer->GetTypedPtr(),
                        *edgesWeightListContainer,
                        nodesValueListContainer->GetTypedPtr(),

                        blockEdgesCountOffsetMapContainer->GetTypedPtr(),
                        *blockEdgesIndexListContainer,
                        nodeNodesSMemIndexListContainer->GetTypedPtr(),

                        m_numNodesRoundUp,
                        m_maxNumOfEdgesInBlock,
                        m_maxNodeRank,
                        m_dsmemOffset
                        );

                    adjacency->grid_dims = grid_seg.total_grid_dims;
                    adjacency->block_offset = grid_seg.offset;

                    return std::shared_ptr<::maps::IContainerComposition>(adjacency);
                }
            };
        };

    } // namespace multi

} // namespace maps

#endif // __MAPS_MULTI_GRAPH_CONTAINERS_H
