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

#ifndef __MAPS_MULTI_OUTPUT_CONTAINERS_H
#define __MAPS_MULTI_OUTPUT_CONTAINERS_H

#include <cuda_runtime.h>
#include <iterator>
#include <iostream>
#include <memory>
#include <vector>

#include "scheduler.h"
#include "../internal/common.cuh"
#include "aggregators.h"

#include "../output_containers/injective.cuh"
#include "../output_containers/reductive.cuh"

namespace maps
{
    namespace multi
    {
        template <typename T, int DIMS, int ITEMS_PER_THREAD = 1, int ROWS_PER_THREAD = 1, ::maps::ILPScheme ILP_SCHEME = ::maps::ILP_CONTINUOUS>
        struct StructuredInjectiveOutput
        {
            MAPS_STATIC_ASSERT(DIMS >= 1 && DIMS <= 4, "A structured injective array can only exist in 1 to 4 dimensions");

            IDatum *datum;
            StructuredInjectiveOutput(Datum<T, DIMS>& dat) : datum(&dat) {}
            StructuredInjectiveOutput(ColVector<T>& dat) : datum(&dat) {}

            class Segmenter : public ISegmenter
            {
            protected:
                IDatum *datum;
            public:                
                Segmenter(IDatum *inDatum) : datum(inDatum) {}

                virtual unsigned int ItemsPerThread(unsigned int dimension) const override
                {
                    switch (dimension)
                    {
                    default:
                        return 1;
                    case 0:
                        return ITEMS_PER_THREAD;
                    case 1:
                        return ROWS_PER_THREAD;
                    }
                }

                virtual void Segment(const std::vector<GridSegment>& segments,
                                     std::vector< std::vector<DatumSegment> >& out_data_segments) const override
                {                    
                    out_data_segments.reserve(segments.size());

                    for (const auto& seg : segments)
                    {
                        std::vector<DatumSegment> datasegs (1);

                        DatumSegment dataseg(DIMS);
                        dataseg.m_borders = WB_NOCHECKS;

                        if (seg.blocks == 0)
                        {
                            printf("SANITY CHECK FAILED: Zero blocks requested!\n");
                            return;
                        }

                        // Compute start and end positions
                        switch (DIMS)
                        {
                        default:
                            break;
                        case 1:
                            dataseg.m_offset[0] = seg.offset * seg.block_dims.x * ITEMS_PER_THREAD;
                            dataseg.m_dimensions[0] = seg.blocks * seg.block_dims.x * ITEMS_PER_THREAD;
                            break;
                        case 2:
                            {
                                unsigned int xoff_begin = seg.offset % seg.total_grid_dims.x;
                                unsigned int yoff_begin = seg.offset / seg.total_grid_dims.x;
                                unsigned int xoff_end   = (seg.offset + seg.blocks - 1) % seg.total_grid_dims.x;
                                unsigned int yoff_end   = (seg.offset + seg.blocks - 1) / seg.total_grid_dims.x;
                                if (xoff_begin != 0 || xoff_end != (seg.total_grid_dims.x - 1))
                                {
                                    printf("ERROR: Uneven segmentation of matrix to rows\n");
                                    out_data_segments.clear();
                                    return;
                                }

                                dataseg.m_offset[0] = 0;
                                dataseg.m_offset[1] = yoff_begin * seg.block_dims.y * ROWS_PER_THREAD;
                                dataseg.m_dimensions[0] = seg.block_dims.x * seg.total_grid_dims.x * ITEMS_PER_THREAD;
                                dataseg.m_dimensions[1] = std::max(seg.block_dims.y * ROWS_PER_THREAD, 
                                                                   (yoff_end - yoff_begin + 1) * seg.block_dims.y * ROWS_PER_THREAD);
                                break;
                            }
                        case 3:
                            {
                                unsigned int xoff_begin =   seg.offset % seg.total_grid_dims.x;
                                unsigned int yoff_begin =  (seg.offset / seg.total_grid_dims.x) % seg.total_grid_dims.y;
                                unsigned int zoff_begin =  (seg.offset / seg.total_grid_dims.x) / seg.total_grid_dims.y;
                                unsigned int xoff_end   =  (seg.offset + seg.blocks - 1) % seg.total_grid_dims.x;
                                unsigned int yoff_end   = ((seg.offset + seg.blocks - 1) / seg.total_grid_dims.x) % seg.total_grid_dims.y;
                                unsigned int zoff_end   = ((seg.offset + seg.blocks - 1) / seg.total_grid_dims.x) / seg.total_grid_dims.y;
                                if (xoff_begin != 0 || xoff_end != (seg.total_grid_dims.x - 1) ||
                                    yoff_begin != 0 || yoff_end != (seg.total_grid_dims.y - 1))
                                {
                                    printf("ERROR: Uneven segmentation of tensor to slices\n");
                                    out_data_segments.clear();
                                    return;
                                }

                                dataseg.m_offset[0] = 0;
                                dataseg.m_offset[1] = 0;
                                dataseg.m_offset[2] = zoff_begin * seg.block_dims.z;
                                dataseg.m_dimensions[0] = seg.block_dims.x * seg.total_grid_dims.x * ITEMS_PER_THREAD;
                                dataseg.m_dimensions[1] = seg.block_dims.y * seg.total_grid_dims.y * ROWS_PER_THREAD;
                                dataseg.m_dimensions[2] = std::max(seg.block_dims.z, 
                                                                   (zoff_end - zoff_begin + 1) * seg.block_dims.z);
                                break;
                            }
                        case 4:
                            {
                                int blocks_x = (int)datum->GetDataDimension(0) / seg.block_dims.x / ITEMS_PER_THREAD;
                                int blocks_y = (int)datum->GetDataDimension(1) / seg.block_dims.y / ROWS_PER_THREAD;
                                int blocks_z = (int)datum->GetDataDimension(2) / seg.block_dims.z;

                                unsigned int xoff_begin = seg.offset % blocks_x;
                                unsigned int yoff_begin = (seg.offset / blocks_x) % blocks_y;
                                unsigned int zoff_begin = ((seg.offset / blocks_x) / blocks_y) % blocks_z;
                                unsigned int woff_begin = ((seg.offset / blocks_x) / blocks_y) / blocks_z;
                                unsigned int xoff_end = (seg.offset + seg.blocks - 1) % blocks_x;
                                unsigned int yoff_end = ((seg.offset + seg.blocks - 1) / blocks_x) % blocks_y;
                                unsigned int zoff_end = (((seg.offset + seg.blocks - 1) / blocks_x) / blocks_y) % blocks_z;
                                unsigned int woff_end = (((seg.offset + seg.blocks - 1) / blocks_x) / blocks_y) / blocks_z;
                                if (xoff_begin != 0 || xoff_end != (blocks_x - 1) ||
                                    yoff_begin != 0 || yoff_end != (blocks_y - 1) ||
                                    zoff_begin != 0 || zoff_end != (blocks_z - 1))
                                {
                                    printf("ERROR: Uneven segmentation of tensor to slices\n");
                                    out_data_segments.clear();
                                    return;
                                }

                                dataseg.m_offset[0] = 0;
                                dataseg.m_offset[1] = 0;
                                dataseg.m_offset[2] = 0;
                                dataseg.m_offset[3] = woff_begin;
                                dataseg.m_dimensions[0] = seg.block_dims.x * blocks_x * ITEMS_PER_THREAD;
                                dataseg.m_dimensions[1] = seg.block_dims.y * blocks_y * ROWS_PER_THREAD;
                                dataseg.m_dimensions[2] = seg.block_dims.z * blocks_z;
                                dataseg.m_dimensions[3] = std::max(1U, (woff_end - woff_begin + 1));
                                break;
                            }
                        }

                        // Multiply dimensions by ITEMS PER THREAD and ROWS PER THREAD 
                        // (they are divided in segmentation)
                        if (datum->GetDataDimension(0) % ITEMS_PER_THREAD != 0)
                        {
                            printf("WARNING: Width is not a multiple of items per thread\n");
                        }

                        if (DIMS > 1)
                        {
                            if (datum->GetDataDimension(1) % ROWS_PER_THREAD != 0)
                            {
                                printf("WARNING: Height is not a multiple of rows per thread\n");
                            }
                        }

                        // Avoid unnecessary overflows due to block size
                        for (int d = 0; d < DIMS; ++d)
                        {
                            if (dataseg.m_offset[d] + dataseg.m_dimensions[d] >= datum->GetDataDimension(d))
                            {
                                dataseg.m_dimensions[d] = datum->GetDataDimension(d) - dataseg.m_offset[d];
                            }
                        }
                        
                        datasegs[0] = dataseg;
                        out_data_segments.push_back(datasegs);
                    }
                }
            };

            class ContainerFactory : public IContainerFactory
            {
            public:
                virtual std::shared_ptr<::maps::IOutputContainer> 
                    CreateOutputContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const override
                {
#ifndef NDEBUG
                    if (!datum || datum->GetDataDimensions() != DIMS)
                    {
                        printf("Invalid datum dimensions for structured injective array (dimensions: %d, expected %d)\n", 
                               datum->GetDataDimensions(), DIMS);
                        return nullptr;
                    }
#endif
                    enum
                    {
                        CONTAINER_DIMS = (DIMS > 3) ? 3 : DIMS,
                    };
                    typedef ::maps::StructuredInjectiveOutput<T, CONTAINER_DIMS, 1,1,1, ITEMS_PER_THREAD, ROWS_PER_THREAD, ILP_SCHEME> ContainerType;
                    ContainerType *cont = new ContainerType();
                    cont->m_ptr = (T *)buff.ptr;                    
                    cont->m_stride = buff.stride_bytes / datum->GetElementSize();
                    cont->grid_dims = grid_seg.total_grid_dims;
                    for (unsigned int i = 0; i < CONTAINER_DIMS; ++i)
                        cont->m_dimensions[i] = (int)seg.GetDimension(i);

                    // The rest of the dimensions are located at Z
                    for (unsigned int i = 3; i < DIMS; ++i)
                        cont->m_dimensions[2] *= (int)seg.GetDimension(i);

                    return std::shared_ptr<::maps::IOutputContainer>(cont);
                }
            };

            std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum)); }
            std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }
        };

        template <typename T, int ITEMS_PER_THREAD = 1, int ROWS_PER_THREAD = 1, ::maps::ILPScheme ILP_SCHEME = ::maps::ILP_CONTINUOUS>
        using StructuredInjectiveVectorO = StructuredInjectiveOutput<T, 1, ITEMS_PER_THREAD, ROWS_PER_THREAD, ILP_SCHEME>;

        template <typename T, int ITEMS_PER_THREAD = 1, int ROWS_PER_THREAD = 1, ::maps::ILPScheme ILP_SCHEME = ::maps::ILP_CONTINUOUS>
        using StructuredInjectiveMatrixO = StructuredInjectiveOutput<T, 2, ITEMS_PER_THREAD, ROWS_PER_THREAD, ILP_SCHEME>;

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Reductive-static
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////

        template <typename T, int LENGTH, int BLOCK_WIDTH, int ITEMS_PER_THREAD = 1, typename Agg = PlusAggregator<T> >
        struct ReductiveStaticOutput
        {
            MAPS_STATIC_ASSERT(LENGTH > 0, "Reductive (static) output must have a positive, fixed size");
            MAPS_STATIC_ASSERT(ITEMS_PER_THREAD > 0, "Reductive (static) output must have a positive amount of items per thread");

            IDatum *datum;
            ReductiveStaticOutput(Vector<T>& dat) : datum(&dat) {}
            ReductiveStaticOutput(ColVector<T>& dat) : datum(&dat) {}

            class Segmenter : public ISegmenter
            {
                IDatum *datum;
            public:
                Segmenter(IDatum *inDatum) : datum(inDatum) {}

                virtual unsigned int ItemsPerThread(unsigned int dimension) const override
                {
                    switch (dimension)
                    {
                    default:
                        return 1;
                    case 0:
                        return ITEMS_PER_THREAD;
                    }
                }

                virtual void Segment(const std::vector<GridSegment>& segments,
                                     std::vector< std::vector<DatumSegment> >& out_data_segments) const override
                {                    
                    out_data_segments.reserve(segments.size());

                    for (const auto& seg : segments)
                    {
                        std::vector<DatumSegment> datasegs (1);

                        DatumSegment dataseg(1);
                        dataseg.m_borders = WB_NOCHECKS;

                        dataseg.m_offset[0] = 0;
                        dataseg.m_dimensions[0] = datum->GetDataDimension(0);
                        
                        datasegs[0] = dataseg;
                        out_data_segments.push_back(datasegs);
                    }
                }
            };

            class ContainerFactory : public IContainerFactory
            {
            public:
                virtual std::shared_ptr<::maps::IOutputContainer> 
                    CreateOutputContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const override
                {
#ifndef NDEBUG
                    if (!datum || datum->GetDataDimensions() != 1)
                    {
                        printf("Invalid datum dimensions for reductive static output (dimensions: %d, expected %d)\n", 
                               datum->GetDataDimensions(), 1);
                        return nullptr;
                    }
#endif
                    typedef ::maps::ReductiveStaticOutput<T, LENGTH, BLOCK_WIDTH, ITEMS_PER_THREAD> ContainerType;
                    ContainerType *cont = new ContainerType();
                    cont->m_ptr = (T *)buff.ptr;
                    
                    return std::shared_ptr<::maps::IOutputContainer>(cont);
                }
            };

            std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum)); }
            std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }
            std::shared_ptr<IAggregator> CreateAggregator() const { return std::shared_ptr<IAggregator>(new Agg()); }
        };

    } // namespace multi

} // namespace maps

#endif // __MAPS_MULTI_OUTPUT_CONTAINERS_H
