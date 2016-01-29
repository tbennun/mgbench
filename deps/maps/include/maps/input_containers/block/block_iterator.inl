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

#ifndef __MAPS_BLOCK_ITERATOR_INL_
#define __MAPS_BLOCK_ITERATOR_INL_

// Don't include this file directly

namespace maps
{

    /// @brief Internal Block 1D iterator class
    template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
             int IPX, int IPY, int IPZ, int CHUNKX, int CHUNKY, int CHUNKZ,
             int XSTRIDE, BorderBehavior BORDERS, int TEXTURE_UID, 
             GlobalReadScheme GRS, bool MULTI_GPU>
    class BlockIterator<T, 1, 0, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, IPX, 
                        IPY, IPZ, CHUNKX, CHUNKY, CHUNKZ, XSTRIDE, BORDERS, 
                        TEXTURE_UID, GRS, MULTI_GPU>
      : public std::iterator<std::input_iterator_tag, T>
    {
    protected:
        typedef Block<T, 1, 0, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, IPX, 
                      IPY, IPZ, CHUNKX, CHUNKY, CHUNKZ, BORDERS, 
                      TEXTURE_UID, GRS, MULTI_GPU> Parent;

        const Parent& m_parent;
        int m_id;
        int m_pos;        

        __device__  __forceinline__ void next()
        {
            ++m_id;
        }

    public:
        __device__ BlockIterator(unsigned int pos, unsigned int offset, 
                                 const Parent& parent) : m_parent(parent),
            m_id(0), m_pos(pos + offset)
        {
        }

        __device__ BlockIterator(const BlockIterator& other) : m_parent(other.m_parent),
            m_id(other.m_id), m_pos(other.m_pos)
        {
        }
        
        __device__ __forceinline__ int index() const
        {
            return m_id;
        }

        __device__ __forceinline__ const T& operator*() const
        {
            return m_parent.m_sdata[m_pos + m_id];
        }

        __device__  __forceinline__ BlockIterator& operator++() // Prefix
        {
            next();
            return *this;
        }

        __device__  __forceinline__ BlockIterator operator++(int) // Postfix
        {
            BlockIterator temp(*this);
            next();
            return temp;
        }

        __device__  __forceinline__ bool operator==(
            const BlockIterator& a) const
        {
            return m_id == a.m_id;
        }
        __device__  __forceinline__ bool operator!=(
            const BlockIterator& a) const
        {
            return m_id != a.m_id;
        }
    };

    /// @brief Internal Block 2D iterator class
    template<typename T, int PRINCIPAL_DIM, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
             int BLOCK_DEPTH, int IPX, int IPY, int IPZ,
             int CHUNKX, int CHUNKY, int CHUNKZ, int XSTRIDE,
             BorderBehavior BORDERS, int TEXTURE_UID, GlobalReadScheme GRS, 
             bool MULTI_GPU>
    class BlockIterator<T, 2, PRINCIPAL_DIM, BLOCK_WIDTH, BLOCK_HEIGHT, 
                        BLOCK_DEPTH, IPX, IPY, IPZ, CHUNKX, CHUNKY, CHUNKZ, 
                        XSTRIDE, BORDERS, TEXTURE_UID, GRS, MULTI_GPU> 
        : public std::iterator<std::input_iterator_tag, T>
    {
    protected:
        typedef Block<T, 2, PRINCIPAL_DIM, BLOCK_WIDTH, BLOCK_HEIGHT, 
                      BLOCK_DEPTH, IPX, IPY, IPZ, CHUNKX, CHUNKY, CHUNKZ, 
                      BORDERS, TEXTURE_UID, GRS, MULTI_GPU> Parent;

        const Parent& m_parent;
        int m_id;
        int m_pos;

    public:

        __device__ BlockIterator(unsigned int pos, unsigned int offset, 
                                 const Parent& parent) : m_parent(parent),
            m_id(pos), m_pos(offset)
        {
        }

        __device__ __forceinline__ unsigned int index() const
        {
            return m_id;
        }

        __device__ __forceinline__ const T& operator*() const
        {            
            if (PRINCIPAL_DIM == 0)
                return m_parent.m_sdata[m_pos + m_id];
            else
                return m_parent.m_sdata[m_pos + XSTRIDE*m_id];
        }


        __device__ __forceinline__ BlockIterator& operator++() // Prefix
        {
            next();
            return *this;
        }

        __device__ __forceinline__ BlockIterator operator++(int) // Postfix
        {
            BlockIterator temp(*this);
            next();
            return temp;
        }

        __device__ __forceinline__ void next()
        {
            ++m_id;
        }

        __device__ __forceinline__ bool operator==(const BlockIterator &a)
        {
            return (m_id == a.m_id);
        }
        __device__ __forceinline__ bool operator!=(const BlockIterator &a)
        {
            return !(m_id == a.m_id);
        }
    };

}  // namespace maps

#endif  // __MAPS_BLOCK_ITERATOR_INL_
