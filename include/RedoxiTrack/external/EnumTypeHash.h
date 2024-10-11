//
// Created by sfj on 2022/4/15.
//

#pragma once
#include "stdint.h"
#include "iostream"
namespace hash
{
    template <typename S> struct fnv_internal;
    template <typename S> struct fnv1;
    template <typename S> struct fnv1a;

    template <> struct fnv_internal<uint32_t>
    {
        constexpr static uint32_t default_offset_basis = 0x811C9DC5;
        constexpr static uint32_t prime				   = 0x01000193;
    };

    template <> struct fnv1<uint32_t> : public fnv_internal<uint32_t>
    {
        constexpr static inline uint32_t hash(char const*const aString, const uint32_t val = default_offset_basis)
        {
            return (aString[0] == '\0') ? val : hash( &aString[1], ( val * prime ) ^ uint32_t(aString[0]) );
        }

        constexpr static inline uint32_t hash(char const*const aString, const size_t aStrlen, const uint32_t val)
        {
            return (aStrlen == 0) ? val : hash( aString + 1, aStrlen - 1, ( val * prime ) ^ uint32_t(aString[0]) );
        }
    };

    template <> struct fnv1a<uint32_t> : public fnv_internal<uint32_t>
    {
        constexpr static inline uint32_t hash(char const*const aString, const uint32_t val = default_offset_basis)
        {
            return (aString[0] == '\0') ? val : hash( &aString[1], ( val ^ uint32_t(aString[0]) ) * prime);
        }

        constexpr static inline uint32_t hash(char const*const aString, const size_t aStrlen, const uint32_t val)
        {
            return (aStrlen == 0) ? val : hash( aString + 1, aStrlen - 1, ( val ^ uint32_t(aString[0]) ) * prime);
        }
    };
} // namespace hash

