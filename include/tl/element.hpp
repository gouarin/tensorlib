#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tl
{
    // The scalar types the kernel IR computes with. Complex, float16 and bfloat16 join
    // this set with the dtypes of the core.
    template <class T>
    concept scalar = std::same_as<T, bool> || std::same_as<T, std::int8_t> || std::same_as<T, std::int16_t> || std::same_as<T, std::int32_t>
                  || std::same_as<T, std::int64_t> || std::same_as<T, std::uint8_t> || std::same_as<T, std::uint16_t>
                  || std::same_as<T, std::uint32_t> || std::same_as<T, std::uint64_t> || std::same_as<T, float> || std::same_as<T, double>;

    // How an element type is stored: a scalar of the closed set, repeated over a component
    // shape of rank 0 to 4, in C order. Specialise it for your own trivially copyable type
    // whose layout is exactly `scalar[extents...]`:
    //
    //     struct dual { double value, derivative; };
    //     template <> struct tl::element_traits<dual>
    //     {
    //         using scalar = double;
    //         static constexpr std::array<std::size_t, 1> extents{2};
    //     };
    template <class T>
    struct element_traits;

    template <scalar T>
    struct element_traits<T>
    {
        using scalar = T;
        static constexpr std::array<std::size_t, 0> extents{};
    };

    namespace detail
    {
        template <std::size_t Rank>
        constexpr std::size_t product(const std::array<std::size_t, Rank>& extents)
        {
            std::size_t size = 1;
            for (const std::size_t extent : extents)
            {
                size *= extent;
            }
            return size;
        }
    }

    // A type tensorlib can store in a tensor and compute with.
    template <class T>
    concept element = requires {
        typename element_traits<T>::scalar;
        element_traits<T>::extents;
    } && scalar<typename element_traits<T>::scalar> && (element_traits<T>::extents.size() <= 4) && std::is_trivially_copyable_v<T> && sizeof(T) == sizeof(typename element_traits<T>::scalar) * detail::product(element_traits<T>::extents) && alignof(T) == alignof(typename element_traits<T>::scalar);
}
