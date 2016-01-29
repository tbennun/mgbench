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

#ifndef __MAPS_MULTI_MEMORY_ANALYZER_H
#define __MAPS_MULTI_MEMORY_ANALYZER_H

#include <vector>
#include <map>
#include <set>
#include "common.h"

namespace maps
{
    namespace multi
    {
        /**
         * @brief Analyzes memory access patterns for allocation purposes.
         */
        class MemoryAnalyzer
        {
        protected:
            /// The segments to be allocated and used throughout the invocations.
            /// @note The segments are queried by device ID (for continuous allocation on the same device),
            ///        and queried by Datum pointer and device ID (for creating the input/output containers).
            /// @note Assuming one segment per GPU. In the future, this may change
            std::vector< std::map<IDatum *, DatumSegment> > m_segments;

            /// Used for keeping track of which segments this analyzer owns
            std::set<IDatum *> m_buffers;

            void ClampToDatum(IDatum *datum)
            {
                unsigned int numGPUs = (unsigned int)m_segments.size();

                for (unsigned int i = 0; i < numGPUs; ++i)
                {
                    if (m_segments[i].find(datum) == m_segments[i].end())
                        continue;

                    if (m_segments[i][datum].Covers(datum))
                    {
                        unsigned int dims = datum->GetDataDimensions();
                        for (unsigned int dim = 0; dim < dims; ++dim)
                        {
                            m_segments[i][datum].m_offset[dim] = 0;
                            m_segments[i][datum].m_dimensions[dim] = datum->GetDataDimension(dim);
                        }
                    }
                }
            }
        public:

            MemoryAnalyzer() { }
            void Reset(size_t numGPUs)
            {
                m_buffers.clear();
                m_segments.clear();
                m_segments.resize(numGPUs);
            }            

            bool HasDatum(IDatum *datum)
            {
                return (m_buffers.find(datum) != m_buffers.end());
            }

            /**
             * @brief Returns the allocated memory segment relating to the 
             *        given datum and device ID, if found.
             * @param[in] gpu The device ID to look for the datum in.
             * @param[in] datum The datum to look for.
             * @param[out] seg The allocated memory segment on the device, 
             *                 filled only if found.
             * @return True if datum is found in the given device, false 
             *         otherwise.
             */
            bool GetSegment(unsigned int gpu, IDatum *datum, DatumSegment& seg) const
            {
                if (gpu >= m_segments.size())
                    return false;
                if (!datum)
                    return false;

                const auto& seg_iter = m_segments[gpu].find(datum);
                if (seg_iter == m_segments[gpu].end())
                    return false;

                seg = seg_iter->second;

                return true;
            }

            void AnalyzeTask(const Task& task)
            {
                unsigned int numGPUs = (unsigned int)task.segmentation.size();

                // Update bounding boxes for allocation (input containers)
                for (const auto& input : task.inputs)
                {
                    std::vector< std::vector<DatumSegment> > segs;
                    input.segmenter->Segment(task.segmentation, segs);
                    for (unsigned int i = 0; i < numGPUs; ++i)
                    {
                        // For each contiguous-memory segment
                        for (const auto& seg : segs[i])
                        {
                            // If not found, initialize
                            if (m_segments[i].find(input.datum) == m_segments[i].end())
                                m_segments[i][input.datum] = seg;
                            else
                                m_segments[i][input.datum].BoundingBox(seg);
                        }
                    }
                }

                // Update bounding boxes for allocation (output containers)
                for (const auto& output : task.outputs)
                {
                    std::vector< std::vector<DatumSegment> > segs;
                    output.segmenter->Segment(task.segmentation, segs);
                    for (unsigned int i = 0; i < numGPUs; ++i)
                    {
                        // For each contiguous-memory segment
                        for (const auto& seg : segs[i])
                        {
                            // If not found, initialize
                            if (m_segments[i].find(output.datum) == m_segments[i].end())
                                m_segments[i][output.datum] = seg;
                            else
                                m_segments[i][output.datum].BoundingBox(seg);
                        }
                    }
                }

                // If any segment contains the entire datum (or more), clamp to datum size
                for (const auto& input : task.inputs)
                    ClampToDatum(input.datum);
                for (const auto& output : task.outputs)
                    ClampToDatum(output.datum);
            }

            
        };

    }  // namespace multi
}  // namespace maps

#endif  // __MAPS_MULTI_MEMORY_ANALYZER_H
