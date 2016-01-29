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

#ifndef __MAPS_MULTI_DATUM_H
#define __MAPS_MULTI_DATUM_H

#include <cstdio>
#include <vector>
#include <memory>
#include <algorithm>
#include <cuda_runtime.h> // for dim3
#include "../internal/common.h"

namespace maps
{
    namespace multi
    {

        struct Memory
        {
            void *ptr;
            size_t stride_bytes;

            Memory() : ptr(nullptr), stride_bytes(0) {}
        };

        struct IDatum
        {
            // Datum segment overrides
            virtual unsigned int GetDimensions() const = 0;
            virtual size_t GetDimension(unsigned int dim) const = 0;
            virtual unsigned int GetDataDimensions() const = 0;
            virtual size_t GetDataDimension(unsigned int dim) const = 0;

            virtual size_t GetElementSize() const = 0;            

            virtual void Bind(void *hostptr, size_t strideBytes) = 0;
            virtual const void *HostPtr() const = 0;
            virtual void *HostPtrMutable() = 0;
            virtual size_t GetHostStrideBytes() = 0;

            inline size_t GetTotalBytes() const
            {
                size_t result = 1;
                unsigned int dims = this->GetDataDimensions();

                for (unsigned int d = 0; d < dims; ++d)
                    result *= GetDataDimension(d);

                return result;
            }
        };

        template <typename T, int DIMS>
        struct Datum : public IDatum
        {
            size_t m_dimensions[DIMS];

            T *m_hostptr;
            size_t m_hostStride;

            template<typename... Dimension>
            Datum(Dimension... dims) : m_hostptr(nullptr), m_hostStride(0)
            {
                static_assert(sizeof...(dims) == DIMS, "Input dimensions must agree with data parameters");
                size_t dim_array[] = { (size_t)dims... };

                int i = 0;
                for (size_t dim : dim_array)
                {
                    m_dimensions[i] = dim;
                    ++i;
                }
            }

            Datum(unsigned int dims[DIMS]) : m_hostptr(nullptr), m_hostStride(0)
            {
                for (int i = 0; i < DIMS; ++i)
                    m_dimensions[i] = dims[i];
            }

            virtual size_t GetElementSize() const override { return sizeof(T); }
            virtual unsigned int GetDimensions() const override { return DIMS; }
            virtual unsigned int GetDataDimensions() const override { return DIMS; }
            virtual size_t GetDimension(unsigned int dim) const override 
            { 
                if (dim >= DIMS)
                    return 0;
                return m_dimensions[dim]; 
            }
            virtual size_t GetDataDimension(unsigned int dim) const override
            {
                return GetDimension(dim);
            }
            
            virtual void Bind(void *hostptr, size_t strideBytes = 0) override 
            { 
                m_hostptr = (T *)hostptr;

                // Automatic deduction
                if (strideBytes == 0)
                    m_hostStride = sizeof(T) * m_dimensions[0];
                else
                    m_hostStride = strideBytes;
            }
            virtual const void *HostPtr() const override { return m_hostptr; }
            virtual void *HostPtrMutable() override { return m_hostptr; }
            virtual size_t GetHostStrideBytes() override { return m_hostStride; }
        };
            
        template <typename T>
        struct ColVector : public IDatum
        {
            size_t m_dimensions[2];

            T *m_hostptr;
            size_t m_hostStride;

            ColVector(size_t rows) : m_hostptr(nullptr), m_hostStride(0)
            {
                m_dimensions[0] = 1;
                m_dimensions[1] = rows;
            }

            virtual size_t GetElementSize() const override { return sizeof(T); }
            virtual unsigned int GetDimensions() const override { return 2; }
            virtual unsigned int GetDataDimensions() const override { return 1; }
            virtual size_t GetDimension(unsigned int dim) const override 
            { 
                if (dim >= 2)
                    return 0;
                return m_dimensions[dim]; 
            }
            virtual size_t GetDataDimension(unsigned int dim) const override
            {
                if (dim >= 1)
                    return 0;
                return m_dimensions[1];
            }
            
            virtual void Bind(void *hostptr, size_t strideBytes = 0) override 
            { 
                m_hostptr = (T *)hostptr;

                // Automatic deduction
                if (strideBytes == 0)
                    m_hostStride = sizeof(T) * m_dimensions[1];
                else
                    m_hostStride = strideBytes;
            }
            virtual const void *HostPtr() const override { return m_hostptr; }
            virtual void *HostPtrMutable() override { return m_hostptr; }
            virtual size_t GetHostStrideBytes() override { return m_hostStride; }
        };


        /**
        * Computes index after boundary conditions.
        */
        static inline int64_t ComputeBoundary(int64_t ind, BorderBehavior border, size_t dimsize)
        {
            switch (border)
            {
            default:
            case WB_NOCHECKS:
            case WB_ZERO:
                return ind;
            case WB_COPY:
                return ::maps::Clamp(ind, (int64_t)0, (int64_t)dimsize - 1);
            case WB_WRAP:
                if (dimsize == 0)
                    return 0;
                while (ind < 0)
                    ind += (int64_t)dimsize;
                while (ind >= (int64_t)dimsize)
                    ind -= (int64_t)dimsize;
                return ind;
            }
        }

        struct DatumSegment
        {
            unsigned int m_dims;

            /// Offset inside the datum.
            /// @note Note that the offsets can be out of bounds. MAPS uses the boundary conditions to fill these indices.
            std::vector<int64_t> m_offset; 
            
            /// Dimensions of the segment
            std::vector<size_t> m_dimensions;

            /// Used when segment goes out of the bounds of the datum
            ::maps::BorderBehavior m_borders;

            /// Tells the scheduler that this segment is to be filled with fillValue, instead of copied to
            bool m_bFill;

            /// The value to fill this segment with
            int m_fillValue;

            /// Stride of the datum (only filled when calling unmodified routines)
            size_t m_stride_bytes;

            DatumSegment() : m_dims(0), m_borders(::maps::WB_ZERO), m_bFill(false), m_fillValue(0), m_stride_bytes(0) { }
            DatumSegment(unsigned int dims) : m_dims(dims), m_offset(dims, 0), m_dimensions(dims, 0),
                                              m_bFill(false), m_fillValue(0), m_stride_bytes(0) { }
            DatumSegment(IDatum *datum) : m_dims(datum->GetDataDimensions()), m_offset(datum->GetDataDimensions(), 0), 
                                          m_dimensions(datum->GetDataDimensions(), 0),
                                          m_bFill(false), m_fillValue(0), m_stride_bytes(0)
            {
                for (unsigned int d = 0; d < m_dims; ++d)
                    m_dimensions[d] = datum->GetDataDimension(d);
            }

            virtual unsigned int GetDimensions() const { return m_dims; }
            virtual ptrdiff_t GetOffset(unsigned int dim) const
            { 
                if (dim >= m_dims)
                    return 0;
                return m_offset[dim]; 
            }
            virtual size_t GetDimension(unsigned int dim) const
            {
                if (dim >= m_dims)
                    return 0;
                return m_dimensions[dim];
            }

            inline size_t GetTotalElements() const
            {
                size_t result = 1;
                unsigned int dims = this->GetDimensions();

                for (unsigned int d = 0; d < dims; ++d)
                    result *= this->GetDimension(d);

                return result;
            }

            /**
             * @brief Computes the pointer w.r.t to an offset
             */
            inline void *OffsetPtr(const std::vector<int64_t>& offset, void *ptr, 
                                   size_t element_size, size_t stride_bytes) const
            {
#ifndef NDEBUG
                if (GetDimensions() != offset.size())
                {
                    printf("ERROR: Dimensions must agree\n");
                    return nullptr;
                }
#endif
                size_t byteOffset = 0;
                size_t curSize = 1;
                for (unsigned int i = 0; i < m_dims; ++i)
                {
#ifndef NDEBUG
                    if (offset[i] < m_offset[i])
                    {
                        printf("ERROR: Trying to access an out of bounds index (dimension %d)\n", i + 1);
                        return nullptr;
                    }
#endif
                    byteOffset += ((i == 0) ? element_size : curSize) * (offset[i] - m_offset[i]);
                    curSize    *= ((i == 0) ? stride_bytes : m_dimensions[i]);
                }

                return (unsigned char *)ptr + byteOffset;
            }

            /**
             * @brief Computes the pointer w.r.t to an offset and boundary conditions
             */
            inline void *OffsetPtrBounds(void *ptr, IDatum *datum, size_t element_size, size_t stride_bytes) const
            {
#ifndef NDEBUG
                if (!datum)
                    return nullptr;
                if (GetDimensions() != datum->GetDataDimensions())
                {
                    printf("ERROR: Dimensions must agree\n");
                    return nullptr;
                }
#endif
                size_t byteOffset = 0;
                size_t curSize = 1;
                for (unsigned int i = 0; i < m_dims; ++i)
                {
                    int64_t off = ComputeBoundary(m_offset[i], m_borders, datum->GetDataDimension(i));
                    byteOffset += ((i == 0) ? element_size : curSize) * off;
                    curSize    *= ((i == 0) ? stride_bytes : m_dimensions[i]);
                }

                return (unsigned char *)ptr + byteOffset;
            }

            /**
             * @brief Computes a segment that contains both input segments.
             */
            bool BoundingBox(const DatumSegment& other)
            {
#ifndef NDEBUG
                if (GetDimensions() != other.GetDimensions())
                {
                    printf("ERROR: Dimensions must agree\n");
                    return false;
                }
#endif
                for (unsigned int dim = 0; dim < m_dims; ++dim)
                {
                    int64_t end_offset = m_offset[dim] + m_dimensions[dim] - 1;
                    int64_t other_end_offset = other.m_offset[dim] + other.m_dimensions[dim] - 1;
                    
                    m_offset[dim] = std::min(m_offset[dim], other.m_offset[dim]);
                    m_dimensions[dim] = std::max(end_offset, other_end_offset) - m_offset[dim] + 1;
                }

                return true;
            }

            bool Exclude(const DatumSegment& other)
            {
#ifndef NDEBUG
                if (GetDimensions() != other.GetDimensions())
                {
                    printf("ERROR: Dimensions must agree\n");
                    return false;
                }
#endif
                
                bool intersect = true;

                for (unsigned int dim = 0; dim < m_dims; ++dim)
                {
                    int64_t ad1 = m_offset[dim];
                    int64_t ad2 = m_offset[dim] + m_dimensions[dim] - 1;
                    int64_t bd1 = other.m_offset[dim];
                    int64_t bd2 = other.m_offset[dim] + other.m_dimensions[dim] - 1;

                    intersect &= (ad1 <= bd2) && (ad2 >= bd1);
                }
                if (!intersect)
                    return true; // Nothing to exclude

                unsigned int intersection_axes = 0;
                for (unsigned int dim = 0; dim < m_dims; ++dim)
                {
                    int64_t ad1 = m_offset[dim];
                    int64_t ad2 = m_offset[dim] + m_dimensions[dim] - 1;
                    int64_t bd1 = other.m_offset[dim];
                    int64_t bd2 = other.m_offset[dim] + other.m_dimensions[dim] - 1;
         
                    int64_t offset_begin = ad1, offset_end = ad2;

                    // Non-contiguous intersection
                    if (bd1 > ad1 && bd2 < ad2)
                        return false;

                    // No intersection
                    if (bd2 < ad1 || bd1 > ad2)
                        continue;
                    if (bd1 <= ad1 && bd2 >= ad2)
                        continue;

                    if (bd1 <= ad1 && (bd2 + 1) > ad1)
                        offset_begin = bd2 + 1;
                    if ((bd1 - 1) < ad2 && bd2 >= ad2)
                        offset_end = bd1 - 1;

                    if (offset_begin != ad1 || offset_end != ad2)
                        ++intersection_axes;


                    m_offset[dim] = offset_begin;
                    m_dimensions[dim] = offset_end - offset_begin + 1;
                }

                // Non-contiguous intersection
                if (intersection_axes > 1)
                    return false;

                return true;
            }

            /**
             * @brief Returns true if an entire datum is completely inside this segment.
             **/
            inline bool Covers(const IDatum *datum) const
            {
#ifndef NDEBUG
                if (datum == nullptr)
                {
                    printf("ERROR: null datum given\n");
                    return false;
                }
                if (GetDimensions() != datum->GetDataDimensions())
                {
                    printf("ERROR: Dimensions must agree\n");
                    return false;
                }
#endif

                bool result = true;
                for (unsigned int dim = 0; dim < m_dims; ++dim)
                    result &= (m_offset[dim] <= 0 && (m_offset[dim] + m_dimensions[dim]) >= datum->GetDataDimension(dim));

                return result;
            }

            /**
            * @brief Returns true if another segment is completely inside this segment.
            **/
            inline bool Covers(const DatumSegment& other) const
            {
#ifndef NDEBUG
                if (GetDimensions() != other.GetDimensions())
                {
                    printf("ERROR: Dimensions must agree\n");
                    return false;
                }
#endif
                bool result = true;
                for (unsigned int dim = 0; dim < m_dims; ++dim)
                {
                    int64_t ad1 = m_offset[dim];
                    int64_t ad2 = m_offset[dim] + m_dimensions[dim] - 1;
                    int64_t bd1 = other.m_offset[dim];
                    int64_t bd2 = other.m_offset[dim] + other.m_dimensions[dim] - 1;
                    
                    result &= (ad1 <= bd1 && ad2 >= bd2);
                }

                return result;
            }

            inline size_t ComputeSize() const
            {
                size_t result = 0;

                // Datum segment, static portion
                result += sizeof(uint32_t) + sizeof(::maps::BorderBehavior) + sizeof(uint8_t) + sizeof(int32_t) + sizeof(uint64_t);
                // Datum segment, dynamic portion
                result += m_dims * (sizeof(int64_t) + sizeof(uint64_t));

                return result;
            }

        };

        /**
        * Rectangular-rectangular intersection. For use when computing
        * necessary segment copies. Assumes memory-contiguous regions. (namely,
        * segments with both out-of-bound and in-bound values will produce
        * erroneous results).
        */
        static bool IntersectsWith(const DatumSegment& a, const DatumSegment& b, 
                                   BorderBehavior border, const IDatum *datum)
        {
#ifndef NDEBUG
            if (datum == nullptr)
            {
                printf("ERROR: null datum given\n");
                return false;
            }
            if (datum->GetDataDimensions() != a.m_dims || a.m_dims != b.m_dims)
            {
                printf("ERROR: Dimensions must agree\n");
                return false;
            }
#endif
            bool intersect = true;

            for (unsigned int dim = 0; dim < a.m_dims; ++dim)
            {
                int64_t ad1 = ComputeBoundary(a.m_offset[dim],                           border, datum->GetDataDimension(dim));
                int64_t ad2 = ComputeBoundary(a.m_offset[dim] + a.m_dimensions[dim] - 1, border, datum->GetDataDimension(dim));
                int64_t bd1 = ComputeBoundary(b.m_offset[dim],                           border, datum->GetDataDimension(dim));
                int64_t bd2 = ComputeBoundary(b.m_offset[dim] + b.m_dimensions[dim] - 1, border, datum->GetDataDimension(dim));

                intersect &= (ad1 <= bd2) && (ad2 >= bd1);
            }

            return intersect;
        }

        static bool Intersection(const DatumSegment& a, const DatumSegment& b, IDatum *datum,
                                 DatumSegment& intersection_offset_a, DatumSegment& intersection_offset_b)
        {
#ifndef NDEBUG
            if (datum == nullptr)
            {
                printf("ERROR: null datum given\n");
                return false;
            }
            if (datum->GetDataDimensions() != a.m_dims || a.m_dims != b.m_dims)
            {
                printf("ERROR: Dimensions must agree\n");
                return false;
            }                
#endif
            intersection_offset_a = a;
            intersection_offset_b = b;
            for (unsigned int dim = 0; dim < a.m_dims; ++dim)
            {
                int64_t ad1 = a.m_offset[dim];
                int64_t ad2 = a.m_offset[dim] + a.m_dimensions[dim] - 1;
                int64_t bd1 = b.m_offset[dim];
                int64_t bd2 = b.m_offset[dim] + b.m_dimensions[dim] - 1;
                int64_t adb1 = ComputeBoundary(a.m_offset[dim],                           a.m_borders, datum->GetDataDimension(dim));
                int64_t adb2 = ComputeBoundary(a.m_offset[dim] + a.m_dimensions[dim] - 1, a.m_borders, datum->GetDataDimension(dim));
                int64_t bdb1 = ComputeBoundary(b.m_offset[dim],                           b.m_borders, datum->GetDataDimension(dim));
                int64_t bdb2 = ComputeBoundary(b.m_offset[dim] + b.m_dimensions[dim] - 1, b.m_borders, datum->GetDataDimension(dim));

                // Compute intersection with boundaries, use original values to get offsets
                int64_t bound_offset = std::max(adb1, bdb1);
                int64_t bound_dim    = std::min(adb2, bdb2) - bound_offset + 1;

                intersection_offset_a.m_offset[dim] = bound_offset - adb1 + ad1;
                intersection_offset_b.m_offset[dim] = bound_offset - bdb1 + bd1;
                intersection_offset_a.m_dimensions[dim] = bound_dim;
                intersection_offset_b.m_dimensions[dim] = bound_dim;
            }

            return true;
        }

    }  // namespace multi
}  // namespace maps

#endif  // __MAPS_MULTI_DATUM_H
