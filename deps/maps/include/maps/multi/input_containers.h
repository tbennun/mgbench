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

#ifndef __MAPS_MULTI_INPUT_CONTAINERS_H
#define __MAPS_MULTI_INPUT_CONTAINERS_H

#include <cuda_runtime.h>
#include <iterator>
#include <iostream>
#include <memory>
#include <vector>

#include "common.h"
#include "scheduler.h"
#include "../internal/common.cuh"
#include "../input_containers/block.cuh"
#include "../input_containers/window.cuh"
#include "../input_containers/irregular.cuh"

namespace maps
{
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    namespace multi
    {
        template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int WINDOW_APRON,
                  BorderBehavior BORDER = WB_ZERO, int ITEMS_PER_THREAD = 1, int ROWS_PER_THREAD = 1>
        class Window2D
        {
        public:
            MAPS_STATIC_ASSERT(BORDER != WB_NOCHECKS, "No boundary checking is not supported in multi-GPU mode");

            IDatum *datum;

            Window2D(Matrix<T>& mat) : datum(&mat) {}            

            class Segmenter : public ISegmenter
            {
            protected:
                IDatum *datum;
            public:
                Segmenter(IDatum *dat) : datum(dat) {}

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
                        // Compute 2D Y position (assuming 2D kernel, due to Window2D input)
                        // of beginning and end
                        unsigned int yoff_begin = seg.offset / seg.total_grid_dims.x;
                        unsigned int yoff_end = ::maps::RoundUp((seg.offset + seg.blocks), seg.total_grid_dims.x);
                    
                        // The list is split to memory-contiguous regions (for boundary conditions)
                        std::vector<DatumSegment> datasegs;

                        // Prepare the main data segment, the tile contains the two (top, bottom) aprons as boundary conditions
                        DatumSegment dataseg(2);
                        dataseg.m_borders = BORDER;
                        dataseg.m_offset[0] = 0;
                        dataseg.m_offset[1] = (int64_t)yoff_begin * BLOCK_HEIGHT * ROWS_PER_THREAD - WINDOW_APRON;
                        dataseg.m_dimensions[0] = ITEMS_PER_THREAD * BLOCK_WIDTH * seg.total_grid_dims.x;
                        dataseg.m_dimensions[1] = std::max((unsigned int)(BLOCK_HEIGHT * ROWS_PER_THREAD) + 2 * WINDOW_APRON,
                                                           ((yoff_end - yoff_begin) * BLOCK_HEIGHT * ROWS_PER_THREAD) + 2 * WINDOW_APRON);

                        if (datum->GetDataDimension(0) % ITEMS_PER_THREAD != 0)
                        {
                            printf("WARNING: Window2D Width is not a multiple of items per thread\n");
                        }
                        if (datum->GetDataDimension(1) % ROWS_PER_THREAD != 0)
                        {
                            printf("WARNING: Window2D Height is not a multiple of rows per thread\n");
                        }

                        // If there is only one segment, just copy it
                        if (segments.size() == 1)
                        {
                            dataseg.m_offset[0] = 0;
                            dataseg.m_offset[1] = 0;
                            dataseg.m_dimensions[0] = ITEMS_PER_THREAD * BLOCK_WIDTH * seg.total_grid_dims.x;                            
                            dataseg.m_dimensions[1] = ROWS_PER_THREAD * BLOCK_HEIGHT * seg.total_grid_dims.y;
                            
                            // Avoid unnecessary overflows due to block size
                            for (int d = 0; d < 2; ++d)
                            {
                                if ((dataseg.m_offset[d] + dataseg.m_dimensions[d]) >= datum->GetDataDimension(d))
                                {
                                    dataseg.m_dimensions[d] = datum->GetDataDimension(d) - dataseg.m_offset[d];
                                }
                            }

                            datasegs.push_back(dataseg);
                            out_data_segments.push_back(datasegs);
                            continue;
                        }

                        // Avoid unnecessary overflows due to block size
                        for (int d = 0; d < 2; ++d)
                        {
                            if ((dataseg.m_offset[d] + dataseg.m_dimensions[d]) >= (datum->GetDataDimension(d) + WINDOW_APRON))
                            {
                                dataseg.m_dimensions[d] = datum->GetDataDimension(d) + WINDOW_APRON - dataseg.m_offset[d];
                            }
                        }


                           DatumSegment borderseg;

                        // If this is the topmost segment, add an additional border
                        if (dataseg.m_offset[1] < 0)
                        {
                            borderseg = dataseg;

                            dataseg.m_offset[1] = 0;
                            dataseg.m_dimensions[1] -= WINDOW_APRON;

                            borderseg.m_dimensions[1] = WINDOW_APRON;
                        }
                        // If this is the bottom segment, add an additional border
                        else if ((dataseg.m_offset[1] + dataseg.m_dimensions[1]) >= (ROWS_PER_THREAD * BLOCK_HEIGHT * seg.total_grid_dims.y))
                        {
                            borderseg = dataseg;

                            dataseg.m_dimensions[1] -= WINDOW_APRON;

                            borderseg.m_offset[1] = (ROWS_PER_THREAD * BLOCK_HEIGHT * seg.total_grid_dims.y);
                            borderseg.m_dimensions[1] = WINDOW_APRON;
                        }

                        // Determine what to do with the border segment, if exists
                        if (borderseg.GetDimensions() > 0 && BORDER == WB_ZERO)
                        {
                            borderseg.m_bFill = true;
                            borderseg.m_fillValue = 0;
                        }

                        datasegs.push_back(dataseg);
                        if (borderseg.GetDimensions() > 0)
                            datasegs.push_back(borderseg);

                        out_data_segments.push_back(datasegs);
                    }
                }
            };

            class ContainerFactory : public IContainerFactory
            {
            public:
                virtual std::shared_ptr<::maps::IInputContainer> 
                    CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const override
                {
#ifndef NDEBUG
                    if (!datum || datum->GetDimensions() != 2)
                    {
                        printf("Invalid datum dimensions for 2D window (dimensions: %d, expected two)\n", datum->GetDimensions());
                        return nullptr;
                    }
#endif
                    typedef ::maps::Window<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, WINDOW_APRON, ITEMS_PER_THREAD, ROWS_PER_THREAD, 1, BORDER, -1, ::maps::GR_DISTINCT> ContainerType;
                    ContainerType *cont = new ContainerType();
                    cont->m_ptr = (T *)buff.ptr;
                    cont->m_stride = (int)(buff.stride_bytes / datum->GetElementSize());
                    cont->m_dimensions[0] = (int)seg.GetDimension(0);
                    cont->m_dimensions[1] = (int)seg.GetDimension(1);
                    cont->m_containsApron = !seg.Covers(datum);
                    cont->m_gridWidth = maps::RoundUp((int)seg.GetDimension(0), BLOCK_WIDTH * ITEMS_PER_THREAD);

                    // Offset block ID for irregular grids
                    cont->block_offset = grid_seg.offset % grid_seg.total_grid_dims.x;

                    return std::shared_ptr<::maps::IInputContainer>(cont);
                }
            };


            std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum)); }
            std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }

        };

        template <typename T, int WINDOW_APRON, BorderBehavior BORDER = WB_ZERO>
        class Window2DUnmodified
        {
        public:
            MAPS_STATIC_ASSERT(BORDER != WB_NOCHECKS, "No boundary checking is not supported in multi-GPU mode");

            Matrix<T> *datum;

            Window2DUnmodified(Matrix<T>& mat) : datum(&mat) {}

            class Segmenter : public ISegmenter
            {
            public:
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
                        // Compute 2D Y position (assuming 2D kernel, due to Window2D input)
                        // of beginning and end
                        unsigned int yoff_begin = seg.offset / seg.total_grid_dims.x;
                        unsigned int yoff_end = (seg.offset + seg.blocks) / seg.total_grid_dims.x;
                    
                        // The list is split to memory-contiguous regions (for boundary conditions)
                        std::vector<DatumSegment> datasegs;

                        // Prepare the main data segment, the tile contains the two (top, bottom) aprons as boundary conditions
                        DatumSegment dataseg(2);
                        dataseg.m_borders = BORDER;
                        dataseg.m_offset[0] = 0;
                        dataseg.m_offset[1] = (int64_t)yoff_begin * 1 - WINDOW_APRON;
                        dataseg.m_dimensions[0] = 1 * seg.total_grid_dims.x;
                        dataseg.m_dimensions[1] = std::max((unsigned int)1 + 2 * WINDOW_APRON,
                                                           ((yoff_end - yoff_begin) * 1) + 2 * WINDOW_APRON);

                        // If there is only one segment, just copy it
                        if (segments.size() == 1)
                        {
                            dataseg.m_offset[0] = 0;
                            dataseg.m_offset[1] = 0;
                            dataseg.m_dimensions[0] = 1 * seg.total_grid_dims.x;
                            dataseg.m_dimensions[1] = 1 * seg.total_grid_dims.y;

                            datasegs.push_back(dataseg);
                            out_data_segments.push_back(datasegs);
                            continue;
                        }

                        DatumSegment borderseg;

                        // If this is the topmost segment, add an additional border
                        if (dataseg.m_offset[1] < 0)
                        {
                            borderseg = dataseg;

                            dataseg.m_offset[1] = 0;
                            dataseg.m_dimensions[1] -= WINDOW_APRON;

                            borderseg.m_dimensions[1] = WINDOW_APRON;
                        }
                        // If this is the bottom segment, add an additional border
                        else if ((dataseg.m_offset[1] + dataseg.m_dimensions[1]) >= (1 * seg.total_grid_dims.y))
                        {
                            borderseg = dataseg;

                            dataseg.m_dimensions[1] -= WINDOW_APRON;

                            borderseg.m_offset[1] = (1 * seg.total_grid_dims.y);
                            borderseg.m_dimensions[1] = WINDOW_APRON;
                        }

                        // Determine what to do with the border segment, if exists
                        if (borderseg.GetDimensions() > 0 && BORDER == WB_ZERO)
                        {
                            borderseg.m_bFill = true;
                            borderseg.m_fillValue = 0;
                        }
                        
                        datasegs.push_back(dataseg);
                        if (borderseg.GetDimensions() > 0)
                            datasegs.push_back(borderseg);

                        out_data_segments.push_back(datasegs);
                    }
                }
            };

            class ContainerFactory : public IContainerFactory
            {
            public:
                virtual std::shared_ptr<::maps::IInputContainer> 
                    CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const override
                {
                    return nullptr;
                }
            };


            std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter()); }
            std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }

        };

        //////////////////////////////////////////////////////////////////////////////////////////////
        // WINDOW 4D
        //////////////////////////////////////////////////////////////////////////////////////////////

        template <typename T, int WINDOW_APRON, BorderBehavior BORDER = WB_ZERO>
        class Window4DUnmodified
        {
        public:
            MAPS_STATIC_ASSERT(BORDER != WB_NOCHECKS, "No boundary checking is not supported in multi-GPU mode");

            Datum<T, 4> *datum;

            Window4DUnmodified(Datum<T, 4>& dat) : datum(&dat) {}

            class Segmenter : public ISegmenter
            {
            public:

                Datum<T, 4> *datum;
                Segmenter(Datum<T, 4> *dat) : datum(dat) {}

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
                        // Compute 4D W position of beginning and end
                        uint64_t grid_total = seg.total_grid_dims.x * seg.total_grid_dims.y * seg.total_grid_dims.z;
                        float significant_offset = ((float)seg.offset / (float)grid_total) * datum->GetDataDimension(3);
                        float significant_size = ((float)seg.blocks / (float)grid_total) * datum->GetDataDimension(3);
                        
                        unsigned int woff_begin = (unsigned int)floorf(significant_offset);
                        unsigned int woff_size  = (unsigned int)ceilf(significant_size);

                        // The list is split to memory-contiguous regions (for boundary conditions)
                        std::vector<DatumSegment> datasegs;

                        // Prepare the main data segment, the tile contains the two (top, bottom) aprons as boundary conditions
                        DatumSegment dataseg(4);
                        dataseg.m_borders = BORDER;
                        dataseg.m_offset[0] = 0;
                        dataseg.m_offset[1] = 0;
                        dataseg.m_offset[2] = 0;
                        dataseg.m_offset[3] = (int64_t)woff_begin * 1 - WINDOW_APRON;
                        dataseg.m_dimensions[0] = datum->GetDataDimension(0);
                        dataseg.m_dimensions[1] = datum->GetDataDimension(1);
                        dataseg.m_dimensions[2] = datum->GetDataDimension(2);
                        dataseg.m_dimensions[3] = std::max((unsigned int)1 + 2 * WINDOW_APRON,
                                                           woff_size + 2 * WINDOW_APRON);

                        // If there is only one segment, just copy it
                        if (segments.size() == 1)
                        {
                            dataseg.m_offset[3] = 0;
                            dataseg.m_dimensions[3] = datum->GetDataDimension(3);
                            
                            datasegs.push_back(dataseg);
                            out_data_segments.push_back(datasegs);
                            continue;
                        }

                        DatumSegment borderseg;

                        // If this is the topmost segment, add an additional border
                        if (dataseg.m_offset[3] < 0)
                        {
                            borderseg = dataseg;

                            dataseg.m_offset[3] = 0;
                            dataseg.m_dimensions[3] -= WINDOW_APRON;

                            borderseg.m_dimensions[3] = WINDOW_APRON;
                        }
                        // If this is the bottom segment, add an additional border
                        else if ((dataseg.m_offset[3] + dataseg.m_dimensions[3]) >= datum->GetDataDimension(3))
                        {
                            borderseg = dataseg;

                            dataseg.m_dimensions[3] -= WINDOW_APRON;

                            borderseg.m_offset[3] = datum->GetDataDimension(3);
                            borderseg.m_dimensions[3] = WINDOW_APRON;
                        }

                        // Determine what to do with the border segment, if exists
                        if (borderseg.GetDimensions() > 0 && BORDER == WB_ZERO)
                        {
                            borderseg.m_bFill = true;
                            borderseg.m_fillValue = 0;
                        }

                        datasegs.push_back(dataseg);
                        if (borderseg.GetDimensions() > 0)
                            datasegs.push_back(borderseg);

                        out_data_segments.push_back(datasegs);
                    }
                }
            };

            class ContainerFactory : public IContainerFactory
            {
            public:
                virtual std::shared_ptr<::maps::IInputContainer>
                    CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const override
                {
                    return nullptr;
                }
            };


            std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum)); }
            std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }

        };

        
        //////////////////////////////////////////////////////////////////////////////////////////////
        // WINDOW ND
        /////////////////////////////////////////////////////////////////////////////////////////////

        template <typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, int WINDOW_APRON, int IPX = 1, int IPY = 1, int IPZ = 1,
                  ::maps::BorderBehavior BORDER = ::maps::WB_ZERO>
        class Window
        {
        public:
            Datum<T, DIMS> *datum;

            Window(Datum<T, DIMS>& dat) : datum(&dat) {}

            class Segmenter : public ISegmenter
            {
            public:
                Datum<T, DIMS> *datum;
                Segmenter(Datum<T, DIMS> *inDatum) : datum(inDatum) {}

                virtual unsigned int ItemsPerThread(unsigned int dimension) const override
                {
                    switch (dimension)
                    {
                    default:
                        return 1;
                    case 0:
                        return IPX;
                    case 1:
                        return IPY;
                    case 2:
                        return IPZ;
                    }
                }

                inline unsigned int BlockSize(unsigned int dimension) const
                {
                    switch (dimension)
                    {
                    default:
                        return 1;
                    case 0:
                        return BLOCK_WIDTH;
                    case 1:
                        return BLOCK_HEIGHT;
                    case 2:
                        return BLOCK_DEPTH;
                    }
                }

                virtual void Segment(const std::vector<GridSegment>& segments,
                                     std::vector< std::vector<DatumSegment> >& out_data_segments) const override
                {
                    out_data_segments.reserve(segments.size());

                    // Slices the block according to its contiguous dimension (last one),
                    // if the principal dimension is the last one, we need the entire buffer.
                    for (const auto& seg : segments)
                    {
                        // The list is split to memory-contiguous regions (for boundary conditions)
                        std::vector<DatumSegment> datasegs;

                        // Compute ND position of beginning and end
                        uint64_t grid_total = seg.total_grid_dims.x * seg.total_grid_dims.y * seg.total_grid_dims.z;
                        float significant_offset = ((float)seg.offset / (float)grid_total) * datum->GetDimension(DIMS - 1);
                        float significant_size = ((float)seg.blocks / (float)grid_total) * datum->GetDimension(DIMS - 1);
                        //significant_offset *= BlockSize(DIMS - 1) * ItemsPerThread(DIMS - 1);
                        //significant_size *= BlockSize(DIMS - 1) * ItemsPerThread(DIMS - 1); <-- Already taken into consideration while computing grid dimensions

                        unsigned int off_begin = (unsigned int)floorf(significant_offset);
                        unsigned int off_size = (unsigned int)ceilf(significant_size);
                                                
                        // Prepare the main data segment:
                        DatumSegment dataseg(DIMS);
                        dataseg.m_borders = BORDER;

                        // Prepare appropriate block
                        for (int d = 0; d < DIMS - 1; ++d)
                        {
                            dataseg.m_offset[d] = 0;
                            dataseg.m_dimensions[d] = datum->GetDimension(d);
                        }                        

                        // If there is only one segment, just copy it
                        if (segments.size() == 1)
                        {
                            dataseg.m_offset[DIMS - 1] = 0;
                            dataseg.m_dimensions[DIMS - 1] = datum->GetDimension(DIMS - 1);

                            datasegs.push_back(dataseg);
                            out_data_segments.push_back(datasegs);
                            continue;
                        }

                        DatumSegment borderseg;

                        dataseg.m_offset[DIMS - 1] = (int64_t)off_begin - WINDOW_APRON;
                        dataseg.m_dimensions[DIMS - 1] = off_size + 2 * WINDOW_APRON;

                        // Avoid unnecessary overflows due to block size
                        for (int d = 0; d < DIMS - 1; ++d)
                        {
                            if ((dataseg.m_offset[d] + dataseg.m_dimensions[d]) >= (datum->GetDimension(d) + WINDOW_APRON))
                            {
                                dataseg.m_dimensions[d] = datum->GetDimension(d) + WINDOW_APRON - dataseg.m_offset[d];
                            }
                        }
                        
                        // Boundary conditions

                        // If this is the topmost segment, add an additional border
                        if (dataseg.m_offset[DIMS - 1] < 0)
                        {
                            borderseg = dataseg;

                            dataseg.m_offset[DIMS - 1] = 0;
                            dataseg.m_dimensions[DIMS - 1] -= WINDOW_APRON;

                            borderseg.m_dimensions[DIMS - 1] = WINDOW_APRON;
                        }
                        // If this is the bottom segment, add an additional border
                        else if ((dataseg.m_offset[DIMS - 1] + dataseg.m_dimensions[DIMS - 1]) >= datum->GetDimension(DIMS - 1))
                        {
                            borderseg = dataseg;

                            dataseg.m_dimensions[DIMS - 1] -= WINDOW_APRON;

                            borderseg.m_offset[DIMS - 1] = datum->GetDimension(DIMS - 1);
                            borderseg.m_dimensions[DIMS - 1] = WINDOW_APRON;
                        }

                        // Determine what to do with the border segment, if exists
                        if (borderseg.GetDimensions() > 0 && BORDER == WB_ZERO)
                        {
                            borderseg.m_bFill = true;
                            borderseg.m_fillValue = 0;
                        }

                        datasegs.push_back(dataseg);
                        if (borderseg.GetDimensions() > 0)
                            datasegs.push_back(borderseg);

                        out_data_segments.push_back(datasegs);
                    }
                }
            };

            class ContainerFactory : public IContainerFactory
            {
            public:
                virtual std::shared_ptr<::maps::IInputContainer>
                    CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const override
                {
#ifndef NDEBUG
                    if (!datum || datum->GetDimensions() != DIMS)
                    {
                        printf("Invalid datum dimensions for ND window (dimensions: %d, expected %d)\n", datum->GetDimensions(), DIMS);
                        return nullptr;
                    }
#endif
                    typedef ::maps::Window<T, DIMS, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, WINDOW_APRON, IPX, IPY, IPZ, BORDER, -1, ::maps::GR_DISTINCT> ContainerType;
                    ContainerType *cont = new ContainerType();
                    cont->m_ptr = (T *)buff.ptr;
                    cont->m_stride = (int)(buff.stride_bytes / datum->GetElementSize());
                    for (unsigned int d = 0; d < seg.m_dims; ++d)
                        cont->m_dimensions[d] = (int)seg.GetDimension(d);
                    cont->m_containsApron = !seg.Covers(datum);
                    cont->m_gridWidth = maps::RoundUp((int)seg.GetDimension(0), BLOCK_WIDTH * IPX);

                    // Offset block ID for irregular grids
                    cont->block_offset = grid_seg.offset % grid_seg.total_grid_dims.x;

                    return std::shared_ptr<::maps::IInputContainer>(cont);
                }
            };


            std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum)); }
            std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }

        };

        //////////////////////////////////////////////////////////////////////////////////////////////
        // BLOCK ND
        /////////////////////////////////////////////////////////////////////////////////////////////

        template <typename T, int DIMS, int PRINCIPAL_DIM, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, int IPX = 1, int IPY = 1, int IPZ = 1,
                  ::maps::BorderBehavior BORDER = ::maps::WB_ZERO>
        class Block
        {
        public:
            static_assert((PRINCIPAL_DIM >= 0 && PRINCIPAL_DIM < DIMS), "Invalid principal dimension");

            Datum<T, DIMS> *datum;

            Block(Datum<T, DIMS>& dat) : datum(&dat) {}

            class Segmenter : public ISegmenter
            {
            public:
                Datum<T, DIMS> *datum;
                Segmenter(Datum<T, DIMS> *inDatum) : datum(inDatum) {}

                virtual unsigned int ItemsPerThread(unsigned int dimension) const override
                {
                    switch (dimension)
                    {
                    default:
                        return 1;
                    case 0:
                        return IPX;
                    case 1:
                        return IPY;
                    case 2:
                        return IPZ;
                    }
                }

                virtual void Segment(const std::vector<GridSegment>& segments,
                                     std::vector< std::vector<DatumSegment> >& out_data_segments) const override
                {
                    out_data_segments.reserve(segments.size());

                    // Slices the block according to its contiguous dimension (last one),
                    // if the principal dimension is the last one, we need the entire buffer.
                    for (const auto& seg : segments)
                    {
                        // The list is split to memory-contiguous regions (for boundary conditions)
                        std::vector<DatumSegment> datasegs;

                        // Compute ND position of beginning and end
                        uint64_t grid_total = seg.total_grid_dims.x * seg.total_grid_dims.y * seg.total_grid_dims.z;
                        float significant_offset = ((float)seg.offset / (float)grid_total) * datum->GetDimension(DIMS - 1);
                        float significant_size = ((float)seg.blocks / (float)grid_total) * datum->GetDimension(DIMS - 1);
                        unsigned int off_begin = (unsigned int)floorf(significant_offset);
                        unsigned int off_size = (unsigned int)ceilf(significant_size);

                        // Prepare the main data segment:
                        DatumSegment dataseg(DIMS);
                        dataseg.m_borders = BORDER;

                        // Last dimension, use entire datum
                        if (PRINCIPAL_DIM == (DIMS - 1))
                        {
                            for (int d = 0; d < DIMS; ++d)
                            {
                                dataseg.m_offset[d] = 0;
                                dataseg.m_dimensions[d] = datum->GetDimension(d);
                            }
                        }
                        else
                        {
                            // Any other dimension, prepare appropriate block
                            for (int d = 0; d < DIMS - 1; ++d)
                            {
                                dataseg.m_offset[d] = 0;
                                dataseg.m_dimensions[d] = datum->GetDimension(d);
                            }

                            dataseg.m_offset[DIMS - 1] = (int64_t)off_begin;
                            dataseg.m_dimensions[DIMS - 1] = off_size;

                            // Avoid unnecessary overflows due to block size
                            if ((dataseg.m_offset[DIMS - 1] + dataseg.m_dimensions[DIMS - 1]) >= datum->GetDimension(DIMS - 1))
                            {
                                dataseg.m_dimensions[DIMS - 1] = datum->GetDimension(DIMS - 1) - dataseg.m_offset[DIMS - 1];
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
                    CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const override
                {
#ifndef NDEBUG
                    if (!datum || datum->GetDimensions() != DIMS)
                    {
                        printf("Invalid datum dimensions for ND block (dimensions: %d, expected %d)\n", datum->GetDimensions(), DIMS);
                        return nullptr;
                    }
#endif
                    typedef ::maps::Block<T, DIMS, PRINCIPAL_DIM, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, IPX, IPY, IPZ, 
                                          BORDER, -1, ::maps::GR_DISTINCT> ContainerType;
                    ContainerType *cont = new ContainerType();
                    cont->m_ptr = (T *)buff.ptr;
                    cont->m_stride = buff.stride_bytes / datum->GetElementSize();
                    for (int d = 0; d < DIMS; ++d)
                        cont->m_dimensions[d] = (int)seg.GetDimension(d);
                    cont->grid_dims = grid_seg.total_grid_dims;
                    
                    return std::shared_ptr<::maps::IInputContainer>(cont);

                }
            };


            std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum)); }
            std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }

        };

        // Template aliases for ease of use
        template <typename T, int BLOCK_WIDTH, int IPX = 1, BorderBehavior BORDER = ::maps::WB_ZERO, int BLOCK_HEIGHT = 1, 
                  int BLOCK_DEPTH = 1>
        using Block1D = Block<T, 1, 0, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, IPX, 1, 1, BORDER>;

        template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int IPX = 1, int IPY = 1, BorderBehavior BORDER = ::maps::WB_ZERO,
                  int BLOCK_DEPTH = 1>
        using Block2D = Block<T, 2, 0, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, IPX, IPY, 1, BORDER>;

        template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int IPX = 1, int IPY = 1, BorderBehavior BORDER = ::maps::WB_ZERO,
                  int BLOCK_DEPTH = 1>
        using Block2DTransposed = Block<T, 2, 1, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, IPX, IPY, 1, BORDER>;
        

        //////////////////////////////////////////////////////////////////////////////////////////////
        // BLOCK 1D
        //////////////////////////////////////////////////////////////////////////////////////////////
          
        template <typename T>
        class Block1DUnmodified
        {
        public:
            Vector<T> *datum;

            Block1DUnmodified(Vector<T>& mat) : datum(&mat) {}

            class Segmenter : public ISegmenter
            {
            public:
                Vector<T> *datum;
                Segmenter(Vector<T> *inDatum) : datum(inDatum) {}

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
                        // The list is split to memory-contiguous regions (for boundary conditions)
                        std::vector<DatumSegment> datasegs;

                        // Prepare the main data segment:
                        // Needs the entire block (1D vector)
                        DatumSegment dataseg(1);
                        dataseg.m_borders = WB_ZERO;
                        
                        dataseg.m_offset[0] = 0;
                        dataseg.m_dimensions[0] = datum->GetDataDimension(0);
                        
                        datasegs.push_back(dataseg);
                        out_data_segments.push_back(datasegs);
                    }
                }
            };

            class ContainerFactory : public IContainerFactory
            {
            public:
                virtual std::shared_ptr<::maps::IInputContainer> 
                    CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const override
                {
                    return nullptr;
                }
            };


            std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum)); }
            std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }

        };

        //////////////////////////////////////////////////////////////////////////////////////////////
        // BLOCK 2D
        //////////////////////////////////////////////////////////////////////////////////////////////

        template <bool TRANSPOSED, typename T>
        class Block2DUnmodified
        {
        public:
            Matrix<T> *datum;

            Block2DUnmodified(Matrix<T>& mat) : datum(&mat) {}

            class Segmenter : public ISegmenter
            {
            public:
                Matrix<T> *datum;
                Segmenter(Matrix<T> *inDatum) : datum(inDatum) {}

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
                        // Compute 2D Y position (assuming 2D kernel, due to Block2D input)
                        // of beginning and end
                        unsigned int yoff_begin = seg.offset / seg.total_grid_dims.x;
                        unsigned int yoff_end = (seg.offset + seg.blocks) / seg.total_grid_dims.x;
                    
                        // The list is split to memory-contiguous regions (for boundary conditions)
                        std::vector<DatumSegment> datasegs;

                        // Prepare the main data segment:
                        // Transposed gets the whole matrix
                        // Normal gets only the necessary lines
                        DatumSegment dataseg(2);
                        dataseg.m_borders = WB_ZERO;
                        if (TRANSPOSED)
                        {
                            dataseg.m_offset[0] = 0;
                            dataseg.m_offset[1] = 0;
                            dataseg.m_dimensions[0] = datum->GetDimension(0);
                            dataseg.m_dimensions[1] = datum->GetDimension(1);
                        }
                        else
                        {
                            dataseg.m_offset[0] = 0;
                            dataseg.m_offset[1] = (int64_t)yoff_begin;
                            dataseg.m_dimensions[0] = datum->GetDimension(0);
                            dataseg.m_dimensions[1] = std::max((unsigned int)1, (yoff_end - yoff_begin));
                        }
                        
                        // Avoid unnecessary overflows due to block size
                        for (int d = 0; d < 2; ++d)
                        {
                            if (dataseg.m_offset[d] + dataseg.m_dimensions[d] >= datum->GetDimension(d))
                            {
                                dataseg.m_dimensions[d] = datum->GetDimension(d) - dataseg.m_offset[d];
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
                    CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const override
                {
                    return nullptr;
                }
            };


            std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum)); }
            std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }

        };

        template <bool TRANSPOSED, typename T>
        class Block2D4DUnmodified
        {
        public:
            Datum<T, 4> *datum;

            Block2D4DUnmodified(Datum<T, 4>& dat) : datum(&dat) {}

            class Segmenter : public ISegmenter
            {
            public:
                Datum<T, 4> *datum;
                Segmenter(Datum<T, 4> *inDatum) : datum(inDatum) {}

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
                        int total_blocks = seg.total_grid_dims.x * seg.total_grid_dims.y * seg.total_grid_dims.z;
                        float relative_blocks = (seg.offset + seg.blocks) / (float)total_blocks;
                        float relative_offset = seg.offset / (float)total_blocks;

                        // Compute 4D W position (assuming 4D kernel, due to Layered-Window3D input)
                        // of beginning and end
                        unsigned int woff_begin = (unsigned int)(datum->GetDataDimension(3) * relative_offset);
                        unsigned int woff_end = (unsigned int)(datum->GetDataDimension(3) * relative_blocks);

                    
                        // The list is split to memory-contiguous regions (for boundary conditions)
                        std::vector<DatumSegment> datasegs;

                        // Prepare the main data segment:
                        // Transposed gets the whole matrix
                        // Normal gets only the necessary lines
                        DatumSegment dataseg(4);
                        dataseg.m_borders = WB_ZERO;
                        if (TRANSPOSED)
                        {
                            dataseg.m_offset[0] = 0;
                            dataseg.m_offset[1] = 0;
                            dataseg.m_offset[2] = 0;
                            dataseg.m_offset[3] = 0;
                            dataseg.m_dimensions[0] = datum->GetDataDimension(0);
                            dataseg.m_dimensions[1] = datum->GetDataDimension(1);
                            dataseg.m_dimensions[2] = datum->GetDataDimension(2);
                            dataseg.m_dimensions[3] = datum->GetDataDimension(3);
                        }
                        else
                        {
                            dataseg.m_offset[0] = 0;
                            dataseg.m_offset[1] = 0;
                            dataseg.m_offset[2] = 0;
                            dataseg.m_offset[3] = (int64_t)woff_begin;
                            dataseg.m_dimensions[0] = datum->GetDataDimension(0);
                            dataseg.m_dimensions[1] = datum->GetDataDimension(1);
                            dataseg.m_dimensions[2] = datum->GetDataDimension(2);
                            dataseg.m_dimensions[3] = std::max((unsigned int)1, (woff_end - woff_begin));
                        }
                        
                        // Avoid unnecessary overflows due to block size
                        for (int d = 0; d < 4; ++d)
                        {
                            if (dataseg.m_offset[d] + dataseg.m_dimensions[d] >= datum->GetDataDimension(d))
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
                    CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const override
                {
                    return nullptr;
                }
            };


            std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum)); }
            std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }

        };

        //////////////////////////////////////////////////////////////////////////////////////////////
        // (4D) LAYERED WINDOW 3D
        //////////////////////////////////////////////////////////////////////////////////////////////

        template <typename T, int WINDOW_APRON_X, int WINDOW_APRON_Y, int STRIDE_X = 1, int STRIDE_Y = 1>
        class Window3DLayeredUnmodified
        {
        public:
            Datum<T, 4> *datum;

            Window3DLayeredUnmodified(Datum<T, 4>& dat) : datum(&dat) {}

            class Segmenter : public ISegmenter
            {
            public:
                Datum<T, 4> *datum;
                Segmenter(Datum<T, 4> *inDatum) : datum(inDatum) {}

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
                        int total_blocks = seg.total_grid_dims.x * seg.total_grid_dims.y * seg.total_grid_dims.z;
                        float relative_blocks = (seg.offset + seg.blocks) / (float)total_blocks;
                        float relative_offset = seg.offset / (float)total_blocks;

                        // Compute 4D W position (assuming 4D kernel, due to Layered-Window3D input)
                        // of beginning and end
                        unsigned int woff_begin = (unsigned int)(datum->GetDataDimension(3) * relative_offset);
                        unsigned int woff_end = (unsigned int)(datum->GetDataDimension(3) * relative_blocks);

                        // The list is split to memory-contiguous regions (for boundary conditions)
                        std::vector<DatumSegment> datasegs;

                        // Prepare the main data segment:
                        DatumSegment dataseg(4);
                        dataseg.m_borders = WB_ZERO;
                        dataseg.m_offset[0] = 0;
                        dataseg.m_offset[1] = 0;
                        dataseg.m_offset[2] = 0;
                        dataseg.m_offset[3] = (int64_t)woff_begin;
                        dataseg.m_dimensions[0] = datum->GetDataDimension(0);
                        dataseg.m_dimensions[1] = datum->GetDataDimension(1);
                        dataseg.m_dimensions[2] = datum->GetDataDimension(2);
                        dataseg.m_dimensions[3] = std::max(1U, (woff_end - woff_begin));

                        // Avoid unnecessary overflows due to block size
                        for (int d = 0; d < 4; ++d)
                        {
                            if (dataseg.m_offset[d] + dataseg.m_dimensions[d] >= datum->GetDataDimension(d))
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
                    CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg, const GridSegment& grid_seg) const override
                {
                    return nullptr;
                }
            };


            std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum)); }
            std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory()); }

        };
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        // IRREGULAR INPUT
        //////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T, int DIMS>
        class IrregularInput
        {
        public:
            bool isColVector;
            IDatum *datum;

            IrregularInput(Datum<T, DIMS>& dat) : datum(&dat), isColVector(false) {}
            IrregularInput(ColVector<T>& dat) : datum(&dat), isColVector(true)
            {
                MAPS_STATIC_ASSERT(DIMS == 1, "Irregular input cannot be initialized with column vector if it's not 1D");
            }

            class Segmenter : public ISegmenter
            {
            public:
                bool isColVector;
                IDatum *datum;
                Segmenter(IDatum *inDatum, bool colvec) : datum(inDatum), isColVector(colvec) {}

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
                        // The list is split to memory-contiguous regions (for boundary conditions)
                        std::vector<DatumSegment> datasegs;

                        // Prepare the main data segment:
                        DatumSegment dataseg(DIMS);
                        dataseg.m_borders = WB_NOCHECKS;
                    
                        // Prepare entire buffer
                        for (int d = 0; d < DIMS; ++d)
                        {
                            dataseg.m_offset[d] = 0;
                            dataseg.m_dimensions[d] = datum->GetDataDimension(d);
                        }

                        datasegs.push_back(dataseg);
                        out_data_segments.push_back(datasegs);
                    }
                }
            };

            class ContainerFactory : public IContainerFactory
            {
            protected:
                bool isColVector;
            public:
                ContainerFactory(bool colvec) : isColVector(colvec) {}

                virtual std::shared_ptr<::maps::IInputContainer>
                    CreateContainer(IDatum *datum, Memory& buff, const DatumSegment& seg,
                                    const GridSegment& grid_seg) const override
                {
#ifndef NDEBUG
                    if (!datum || datum->GetDataDimensions() != DIMS)
                    {
                        printf("Invalid datum dimensions for irregular input (dimensions: %d, expected %d)\n",
                               datum->GetDataDimensions(), DIMS);
                        return nullptr;
                    }
#endif
                    typedef ::maps::IrregularInput<T, DIMS> ContainerType;
                    ContainerType *cont = new ContainerType();
                    cont->m_ptr = (T *)buff.ptr;
                    cont->m_stride = buff.stride_bytes / datum->GetElementSize();
                    if (buff.stride_bytes == 0)
                        cont->m_stride = seg.GetDimension(0);

                    for (int d = 0; d < DIMS; ++d)
                        cont->m_dims[d] = (int)seg.GetDimension(d);

                    return std::shared_ptr<::maps::IInputContainer>(cont);
                }
            };


            std::shared_ptr<ISegmenter> CreateSegmenter() const { return std::shared_ptr<ISegmenter>(new Segmenter(datum, isColVector)); }
            std::shared_ptr<IContainerFactory> CreateContainerFactory() const { return std::shared_ptr<IContainerFactory>(new ContainerFactory(isColVector)); }
        };

    } // namespace multi

} // namespace maps

#endif // __MAPS_MULTI_INPUT_CONTAINERS_H
