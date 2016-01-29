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

#ifndef __MAPS_COMMON_H_
#define __MAPS_COMMON_H_

#include <cstdint>
#include <iterator>

#include <cuda_runtime.h>

namespace maps
{
    /**
     * @brief Determines the behavior when accessing data beyond the 
     *        dimensions of the input data.
     */
    enum BorderBehavior
    {
        WB_NOCHECKS,    ///< Assume input is allocated beyond the boundaries 
                        ///  and do not perform the checks.
        WB_ZERO,        ///< Return a constant value of T(0).
        WB_COPY,        ///< Copy the closest value at the border.
        WB_WRAP,        ///< Wrap the results around the input data.
    };
    
    /// @brief The ILP scheme to use.
    enum ILPScheme
    {
        ILP_CONTINUOUS,        ///< Continuous (blocked) indices 
                               ///  (good for small data sizes, e.g. uint8_t, 
                               ///  which can be read as 32-bit).

        ILP_SKIPBLOCK,         ///< Striped indices, skipping by block size 
                               ///  each time (for coalescing and general use).
    };

    enum GlobalReadScheme
    {
        GR_DIRECT = 0,          ///< Reads the pointer directly
        GR_DISTINCT,            ///< Reads the pointer using LDG/Noncoherent 
                                ///  cache
        GR_TEXTURE,             ///< Reads the pointer using textures
    };

    // The use of enum ensures compile-time evaluation (pre-C++11 "constexpr").
    enum
    {
        /// @brief If given as template parameter, determines shared 
        /// memory size at runtime
        DYNAMIC_SMEM = 0,
    };

    static __host__ __device__ inline int64_t Wrap(const int64_t& val, const int64_t& dim)
    {
        if (val >= dim)
            return (val - dim);
        if (val < 0)
            return (val + dim);
        return val;
    }
    
    template <typename T>
    static __host__ __device__ inline T Clamp(const T& value, const T& minValue, const T& maxValue)
    {
        if(value < minValue)
            return minValue;
        if(value > maxValue)
            return maxValue;
        return value;
    }

    // Compile-time version of integer power
    template<int BASE, unsigned int POWER>
    struct Power
    {
        enum
        {
            value = BASE * Power<BASE, POWER - 1>::value,
        };
    };

    template<int BASE>
    struct Power<BASE, 0>
    {
        enum
        {
            value = 1,
        };
    };
    ///////////////////////////////////////

    struct IContainer
    {
        // Each device-level container should implement these
        enum
        {
            ELEMENTS = 0,
            SYNC_AFTER_NEXTCHUNK = true,
            SYNC_AFTER_INIT = true,
        };
        void *m_ptr;

        __host__ __device__ IContainer() : m_ptr(nullptr) {}
    };

    // Generic structures
    struct IInputContainer : public IContainer
    {
    };
    struct IOutputContainer : public IContainer
    {
    };

    template<typename T>
    class TypedInputContainer : public IInputContainer
    {
    private:
        ptrdiff_t m_offset;

    public:
        TypedInputContainer(ptrdiff_t offset = 0) : m_offset(offset) { }

        __host__ __device__ __forceinline__ T* GetTypedPtr() const 
        { 
          return (T*)m_ptr; 
        }
        __host__ __device__ __forceinline__ ptrdiff_t GetOffset() const 
        { 
          return m_offset; 
        }
    };

    struct IContainerComposition : public IContainer
    {

    };

    struct IOutputContainerIterator
    {
        int m_pos;
    };


    ////////////////////////////////////////////////////////////////////////

    // Static assertions
#if (_MSC_VER >= 1600) || (__cplusplus >= 201103L)
    #define MAPS_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
    #ifdef __MAPS_CAT
    #error Using disallowed macro name __MAPS_CAT
    #endif
    #ifdef __MAPS_CAT_
    #error Using disallowed macro name __MAPS_CAT_
    #endif

    // Workaround for static assertions pre-C++11
    #define __MAPS_CAT_(a, b) a ## b
    #define __MAPS_CAT(a, b) __MAPS_CAT_(a, b)
    #define MAPS_STATIC_ASSERT(cond, msg) typedef int __MAPS_CAT(maps_static_assert, __LINE__)[(cond) ? 1 : -1]
#endif

}  // namespace maps

#endif  // __MAPS_COMMON_H_
