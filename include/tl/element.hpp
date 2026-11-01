#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tl
{
    /**
     * @brief The scalar types tensorlib computes with.
     *
     * Satisfied when `T` is `bool`, a signed or unsigned integer of 8, 16, 32 or 64 bits,
     * `float` or `double`. Every element type is stored as one of these, alone or repeated
     * over a component shape (see tl::element).
     */
    template <class T>
    concept scalar = std::same_as<T, bool> || std::same_as<T, std::int8_t> || std::same_as<T, std::int16_t> || std::same_as<T, std::int32_t>
                  || std::same_as<T, std::int64_t> || std::same_as<T, std::uint8_t> || std::same_as<T, std::uint16_t>
                  || std::same_as<T, std::uint32_t> || std::same_as<T, std::uint64_t> || std::same_as<T, float> || std::same_as<T, double>;

    /**
     * @brief How an element type is stored: a scalar, repeated over a component shape.
     *
     * A specialisation declares two members:
     *
     * - `scalar`, one of tl::scalar;
     * - `extents`, a `std::array<std::size_t, R>` of any rank `R` giving the component shape;
     *   the components are stored in C order.
     *
     * tensorlib provides the specialisations for every tl::scalar (rank 0) and every
     * tl::fixed. Specialise it for your own type when it is trivially copyable and laid out
     * exactly like `scalar[extents...]`; tl::element then accepts it, and tensors store it.
     *
     * @code
     * struct dual
     * {
     *     double value;
     *     double derivative;
     * };
     *
     * template <>
     * struct tl::element_traits<dual>
     * {
     *     using scalar = double;
     *     static constexpr std::array<std::size_t, 1> extents{2};
     * };
     * @endcode
     *
     * Kernels see such a type as its components: a `dual` is two lanes of `double`, like
     * `tl::fixed<double, 2>`. Arithmetic on it goes through traced functions.
     *
     * @tparam T The element type described.
     */
    template <class T>
    struct element_traits;

    /**
     * @brief Storage of a scalar: itself, with an empty component shape.
     *
     * @tparam T One of tl::scalar.
     */
    template <scalar T>
    struct element_traits<T>
    {
        /// The scalar itself.
        using scalar = T;
        /// No components.
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

    /**
     * @brief A type tensorlib can store in a tensor and compute with.
     *
     * Satisfied when `T` has a tl::element_traits specialisation whose `scalar` is one of
     * tl::scalar, and `T` is trivially copyable with the size and alignment of
     * `scalar[extents...]`. The check catches traits that do not match the type, such as a
     * wrong number of components.
     */
    template <class T>
    concept element = requires {
        typename element_traits<T>::scalar;
        element_traits<T>::extents;
    } && scalar<typename element_traits<T>::scalar> && std::is_trivially_copyable_v<T> && sizeof(T) == sizeof(typename element_traits<T>::scalar) * detail::product(element_traits<T>::extents) && alignof(T) == alignof(typename element_traits<T>::scalar);
}
