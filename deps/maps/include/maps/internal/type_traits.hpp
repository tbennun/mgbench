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

#ifndef __MAPS_TYPE_TRAITS_HPP_
#define __MAPS_TYPE_TRAITS_HPP_

namespace maps
{
    // Remove qualifiers from type
    template<typename T>
    struct RemoveConst
    {
        typedef T type;
    };

    template<typename T>
    struct RemoveConst<const T>
    {
        typedef T type;
    };

    template<typename T>
    struct RemoveVolatile
    {
        typedef T type;
    };

    template<typename T>
    struct RemoveVolatile<volatile T>
    {
        typedef T type;
    };

    template<typename T>
    struct RemoveQualifiers
    {
        typedef typename RemoveConst<typename RemoveVolatile<T>::type>::type 
          type;
    };


    //////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct _IsIntegral
    {
        enum { value = false };
    };

    #ifdef __MAPS_IS_INTEGRAL_TYPE
    #error Using disallowed macro name __MAPS_IS_INTEGRAL_TYPE
    #endif

    #define __MAPS_IS_INTEGRAL_TYPE(type, val)              \
    template<>                                              \
    struct _IsIntegral<type>                                \
    {                                                       \
        enum { value = val };                               \
    };

    __MAPS_IS_INTEGRAL_TYPE(bool, true);
    __MAPS_IS_INTEGRAL_TYPE(char, true);
    __MAPS_IS_INTEGRAL_TYPE(unsigned char, true);
    __MAPS_IS_INTEGRAL_TYPE(signed char, true);
    __MAPS_IS_INTEGRAL_TYPE(unsigned short, true);
    __MAPS_IS_INTEGRAL_TYPE(signed short, true);
    __MAPS_IS_INTEGRAL_TYPE(unsigned int, true);
    __MAPS_IS_INTEGRAL_TYPE(signed int, true);
    __MAPS_IS_INTEGRAL_TYPE(unsigned long, true);
    __MAPS_IS_INTEGRAL_TYPE(signed long, true);

    // Determines whether T is of integer type
    template <typename T>
    struct IsIntegral : public _IsIntegral<typename RemoveQualifiers<T>::type>
    {        
    };
    
    #undef __MAPS_IS_INTEGRAL_TYPE

    //////////////////////////////////////////////////////////////////////////

    // Converts from an integral amount of bytes to a type.
    template <int BYTES>
    struct BytesToType
    {
        typedef void type;
    };

    #ifdef __MAPS_BYTES_TO_TYPE
    #error Using disallowed macro name __MAPS_BYTES_TO_TYPE
    #endif

    #define __MAPS_BYTES_TO_TYPE(bytes, t)                  \
    template<>                                              \
    struct BytesToType<bytes>                               \
    {                                                       \
        typedef t type;                                     \
    }

    __MAPS_BYTES_TO_TYPE(16, float4);
    __MAPS_BYTES_TO_TYPE(8, uint64_t);
    __MAPS_BYTES_TO_TYPE(4, uint32_t);
    __MAPS_BYTES_TO_TYPE(2, uint16_t);
    __MAPS_BYTES_TO_TYPE(1, uint8_t);

    
    #undef __MAPS_BYTES_TO_TYPE

}  // namespace maps

//////////////////////////////////////////////////////////////////////////

// Operators for vector types

__host__ __device__ inline bool operator==(const float4& a, const float4& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

#endif  // __MAPS_TYPE_TRAITS_HPP_
