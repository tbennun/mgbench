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

#ifndef __MAPS_WINDOW_ITERATOR_INL_
#define __MAPS_WINDOW_ITERATOR_INL_

// Don't include this file directly

namespace maps {

    /// @brief Internal Window ND iterator class
    template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
             int WINDOW_APRON, int IPX, int IPY, int IPZ, 
             BorderBehavior BORDERS, int TEXTURE_UID, GlobalReadScheme GRS, 
             bool USE_REGISTERS, int XSTRIDE, bool MULTI_GPU>
    class WindowIterator<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                         WINDOW_APRON, IPX, IPY, IPZ, BORDERS, TEXTURE_UID, GRS,
                         USE_REGISTERS, XSTRIDE, MULTI_GPU> 
        : public std::iterator<std::input_iterator_tag, T>
    {
        typedef Window<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT,
                       BLOCK_DEPTH, WINDOW_APRON, IPX, IPY,
                       IPZ, BORDERS, TEXTURE_UID, GRS,
                       MULTI_GPU> Parent;
    protected:

        int m_id;
        int m_pos;
        const Parent& m_parent;

        __device__  __forceinline__ void next()
        {
            ++m_id;
        }
    public:
        __device__ WindowIterator(
            unsigned int pos, const Parent& parent) : m_parent(parent)
        {
            m_id = 0;
            m_pos = pos;
        }

        __device__ WindowIterator(const WindowIterator& other) : m_parent(other.m_parent)
        {
            m_id = other.m_id;
            m_pos = other.m_pos;
        }
        
        __device__ __forceinline__ int index() const
        {
            return m_id;
        }

        __device__ __forceinline__ const T& operator*() const
        {
            if (USE_REGISTERS)
                return m_parent.m_regs[m_id + m_pos];
            else
                return m_parent.m_sdata[m_id + m_pos];
        }

        __device__  __forceinline__ WindowIterator& operator++() // Prefix
        {
            next();
            return *this;
        }

        __device__  __forceinline__ WindowIterator operator++(int) // Postfix
        {
            WindowIterator temp(*this);
            next();
            return temp;
        }

        __device__  __forceinline__ bool operator==(
            const WindowIterator& a) const
        {
            return (m_pos + m_id) == (a.m_pos + a.m_id);
        }
        __device__  __forceinline__ bool operator!=(
            const WindowIterator& a) const
        {
            return (m_pos + m_id) != (a.m_pos + a.m_id);
        }
    };

    /// @brief Internal Window ND iterator class
    template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
             int WINDOW_APRON, int IPX, int IPY, int IPZ, 
             BorderBehavior BORDERS, int TEXTURE_UID, GlobalReadScheme GRS, 
             bool USE_REGISTERS, int XSTRIDE, bool MULTI_GPU>
    class WindowIterator<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                         WINDOW_APRON, IPX, IPY, IPZ, BORDERS, TEXTURE_UID, GRS,
                         USE_REGISTERS, XSTRIDE, MULTI_GPU>
        : public std::iterator<std::input_iterator_tag, T>
    {
        typedef Window<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT,
                       BLOCK_DEPTH, WINDOW_APRON, IPX, IPY,
                       IPZ, BORDERS, TEXTURE_UID, GRS,
                       MULTI_GPU> Parent;
    protected:
        unsigned int m_pos;
        int m_id;
        const Parent& m_parent;
        int m_initialOffset;

        enum
        {
            WIND_WIDTH = (WINDOW_APRON * 2 + 1),
        };

        __device__  __forceinline__ void next()
        {
            m_id++;
            m_pos = m_initialOffset + (m_id % WIND_WIDTH) + 
                ((m_id / WIND_WIDTH) * XSTRIDE);
        }
    public:
        __device__ WindowIterator(
            unsigned int pos, const Parent& parent) : m_parent(parent)
        {
            m_pos = pos;
            m_id = 0;
            m_initialOffset = pos;
        }

        __device__ WindowIterator(const WindowIterator& other) : m_parent(other.m_parent)
        {
            m_pos = other.m_pos;
            m_id = other.m_id;
            m_initialOffset = other.m_initialOffset;
        }
        
        __device__ __forceinline__ int index() const
        {
            return m_id;
        }

        __device__ __forceinline__ const T& operator*() const
        {
            if (USE_REGISTERS)
                return m_parent.m_regs[m_pos];
            else
                return m_parent.m_sdata[m_pos];
        }

        __device__  __forceinline__ WindowIterator& operator++() // Prefix
        {
            next();
            return *this;
        }

        __device__  __forceinline__ WindowIterator operator++(int) // Postfix
        {
            WindowIterator temp(*this);
            next();
            return temp;
        }

        __device__  __forceinline__ bool operator==(
            const WindowIterator& a) const
        {
            return m_pos == a.m_pos;
        }
        __device__  __forceinline__ bool operator!=(
            const WindowIterator& a) const
        {
            return m_pos != a.m_pos;
        }
    };

}  // namespace maps

#endif  // __MAPS_WINDOW_ITERATOR_INL_
