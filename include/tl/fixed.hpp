#pragma once

#include <tl/detail/assert.hpp>
#include <tl/element.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tl
{
    // A small tensor whose shape is known at compile time, stored inline in C order. It is a
    // plain aggregate, evaluated eagerly and usable in constant expressions:
    //
    //     tl::fixed<double, 3> v{1., 2., 3.};
    //     tl::fixed<double, 2, 2> m{1., 2., 3., 4.};   // or tl::fixed_from({{1., 2.}, {3., 4.}})
    //     double x = m(1, 0);                          // 3.
    //     m(1) = v2;                                   // m(1) is a view of row 1
    //
    // `==` compares whole values; tl::equal, tl::less and the other Array API comparisons work
    // element by element.
    template <scalar T, std::size_t... N>
        requires(sizeof...(N) >= 1 && sizeof...(N) <= 4 && ((N > 0) && ...))
    struct fixed;

    // A reference to a block of a `fixed`, made by indexing it with fewer indices than its
    // rank. Copying a view gives another view of the same elements; assigning to it writes
    // them. T is const for a view of a const `fixed`.
    template <class T, std::size_t... N>
    class fixed_view;

    namespace detail
    {
        template <class F>
        struct fixed_traits
        {
            static constexpr bool is_fixed = false;
        };

        template <class T, std::size_t... N>
        struct fixed_traits<fixed<T, N...>>
        {
            static constexpr bool is_fixed = true;
            using value_type               = fixed<T, N...>;
        };

        template <class T, std::size_t... N>
        struct fixed_traits<fixed_view<T, N...>>
        {
            static constexpr bool is_fixed = true;
            using value_type               = fixed<std::remove_const_t<T>, N...>;
        };

        // A fixed or a view of one.
        template <class F>
        concept fixed_like = fixed_traits<std::remove_cvref_t<F>>::is_fixed;

        // The fixed value a fixed-like operand stands for.
        template <fixed_like F>
        using value_of = typename fixed_traits<std::remove_cvref_t<F>>::value_type;

        template <class A, class B>
        concept same_fixed = fixed_like<A> && fixed_like<B> && std::same_as<value_of<A>, value_of<B>>;

        // A scalar that adopts the element type of the fixed it is combined with, as Python
        // scalars do in the Array API.
        template <class S>
        concept weak_scalar = std::is_arithmetic_v<S>;

        template <class T>
        concept arithmetic_element = scalar<std::remove_const_t<T>> && !std::same_as<std::remove_const_t<T>, bool>;

        template <std::size_t K, class T, std::size_t... N, std::size_t... I>
        auto subview_type(std::index_sequence<I...>) -> fixed_view<T, std::array<std::size_t, sizeof...(N)>{N...}[K + I]...>;

        // The view left after fixing the first K indices of a block of shape N...
        template <std::size_t K, class T, std::size_t... N>
        using subview = decltype(subview_type<K, T, N...>(std::make_index_sequence<sizeof...(N) - K>{}));

        // Element or block at the leading indices `index...` of C-order storage of shape N...
        template <std::size_t... N, class T, std::convertible_to<std::size_t>... I>
        constexpr decltype(auto) at(T* data, I... index) noexcept
        {
            constexpr std::array<std::size_t, sizeof...(N)> extents{N...};
            const std::array<std::size_t, sizeof...(I)> indices{static_cast<std::size_t>(index)...};
            std::size_t offset = 0;
            for (std::size_t axis = 0; axis < indices.size(); ++axis)
            {
                detail::check_index(indices[axis], extents[axis]);
                offset = (offset * extents[axis]) + indices[axis];
            }
            std::size_t block = 1;
            for (std::size_t axis = indices.size(); axis < extents.size(); ++axis)
            {
                block *= extents[axis];
            }
            if constexpr (sizeof...(I) == sizeof...(N))
            {
                return data[offset];
            }
            else
            {
                return subview<sizeof...(I), T, N...>(data + (offset * block));
            }
        }

        template <fixed_like F>
        using element_of = typename element_traits<value_of<F>>::scalar;

        // Operands of element-wise arithmetic: two fixed of the same shape and element type,
        // or a fixed and a scalar, with a numeric element type.
        template <class A, class B>
        concept arithmetic_operands = (same_fixed<A, B> && arithmetic_element<element_of<A>>)
                                   || (fixed_like<A> && weak_scalar<B> && arithmetic_element<element_of<A>>)
                                   || (weak_scalar<A> && fixed_like<B> && arithmetic_element<element_of<B>>);

        // Operands of element-wise comparisons: as for arithmetic, with any element type
        // for equality and a numeric one for ordering.
        template <class A, class B>
        concept equality_operands = same_fixed<A, B> || (fixed_like<A> && weak_scalar<B>) || (weak_scalar<A> && fixed_like<B>);

        template <class A, class B>
        concept ordered_operands = equality_operands<A, B> && arithmetic_operands<A, B>;

        template <class A, class B>
        using result_of = value_of<std::conditional_t<fixed_like<A>, A, B>>;

        template <class F, class U>
        struct rebind;

        template <class T, std::size_t... N, class U>
        struct rebind<fixed<T, N...>, U>
        {
            using type = fixed<U, N...>;
        };

        // The fixed of booleans an element-wise comparison of A and B gives.
        template <class A, class B>
        using mask_of = typename rebind<result_of<A, B>, bool>::type;

        // Element i of an operand, seen as a value of type T.
        template <class T, class A>
        constexpr T component(const A& operand, std::size_t i) noexcept
        {
            if constexpr (fixed_like<A>)
            {
                return operand.data()[i];
            }
            else
            {
                return static_cast<T>(operand);
            }
        }

        // Applies `op` element by element into a new fixed R; a scalar operand is converted
        // to the element type of the fixed one.
        template <class R, class A, class B, class Op>
        constexpr R elementwise(Op op, const A& a, const B& b) noexcept
        {
            using T = typename element_traits<result_of<A, B>>::scalar;
            using U = typename element_traits<R>::scalar;
            R result{};
            for (std::size_t i = 0; i < product(R::shape()); ++i)
            {
                result.elements[i] = static_cast<U>(op(component<T>(a, i), component<T>(b, i)));
            }
            return result;
        }

        template <class A, class Op, class B>
        constexpr void update(A& target, Op op, const B& operand) noexcept
        {
            using T         = element_of<A>;
            auto* const out = target.data();
            for (std::size_t i = 0; i < product(value_of<A>::shape()); ++i)
            {
                out[i] = static_cast<T>(op(out[i], component<T>(operand, i)));
            }
        }
    }

    template <scalar T, std::size_t... N>
        requires(sizeof...(N) >= 1 && sizeof...(N) <= 4 && ((N > 0) && ...))
    struct fixed
    {
        std::array<T, (N * ...)> elements;

        static constexpr std::array<std::size_t, sizeof...(N)> shape() noexcept
        {
            return {N...};
        }

        constexpr T* data() noexcept
        {
            return elements.data();
        }

        constexpr const T* data() const noexcept
        {
            return elements.data();
        }

        // As many indices as the rank give an element; fewer give a view of a block.
        template <std::convertible_to<std::size_t>... I>
            requires(sizeof...(I) >= 1 && sizeof...(I) <= sizeof...(N))
        constexpr decltype(auto) operator()(I... index) & noexcept
        {
            return detail::at<N...>(data(), index...);
        }

        template <std::convertible_to<std::size_t>... I>
            requires(sizeof...(I) >= 1 && sizeof...(I) <= sizeof...(N))
        constexpr decltype(auto) operator()(I... index) const& noexcept
        {
            return detail::at<N...>(data(), index...);
        }

        template <std::convertible_to<std::size_t>... I>
            requires(sizeof...(I) == sizeof...(N))
        constexpr T operator()(I... index) && noexcept
        {
            return detail::at<N...>(data(), index...);
        }

        // A view of a temporary would dangle.
        template <std::convertible_to<std::size_t>... I>
            requires(sizeof...(I) >= 1 && sizeof...(I) < sizeof...(N))
        void operator()(I... index) && = delete;

        friend constexpr bool operator==(const fixed&, const fixed&) = default;
    };

    template <class T, std::size_t... N>
    class fixed_view
    {
      public:

        using value_type = fixed<std::remove_const_t<T>, N...>;

        constexpr explicit fixed_view(T* data) noexcept
            : m_data(data)
        {
        }

        constexpr fixed_view(const fixed_view&) noexcept = default;
        constexpr fixed_view(fixed_view&&) noexcept      = default;
        constexpr ~fixed_view()                          = default;

        // Assigning writes the elements, like a tl::view.
        // Self-assignment copies each element onto itself, which is harmless.
        constexpr fixed_view& operator=(const fixed_view& other) noexcept // NOLINT(bugprone-unhandled-self-assignment)
            requires(!std::is_const_v<T>)
        {
            return assign(other);
        }

        constexpr fixed_view& operator=(fixed_view&& other) noexcept
            requires(!std::is_const_v<T>)
        {
            return assign(other);
        }

        template <class F>
            requires(!std::is_const_v<T> && detail::same_fixed<F, fixed_view>)
        constexpr fixed_view& operator=(const F& other) noexcept
        {
            return assign(other);
        }

        template <detail::weak_scalar S>
            requires(!std::is_const_v<T>)
        constexpr fixed_view& operator=(S value) noexcept
        {
            for (std::size_t i = 0; i < detail::product(shape()); ++i)
            {
                m_data[i] = static_cast<T>(value);
            }
            return *this;
        }

        static constexpr std::array<std::size_t, sizeof...(N)> shape() noexcept
        {
            return {N...};
        }

        constexpr T* data() const noexcept
        {
            return m_data;
        }

        template <std::convertible_to<std::size_t>... I>
            requires(sizeof...(I) >= 1 && sizeof...(I) <= sizeof...(N))
        constexpr decltype(auto) operator()(I... index) const noexcept
        {
            return detail::at<N...>(m_data, index...);
        }

        // The values the view refers to, copied.
        constexpr operator value_type() const noexcept // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
        {
            value_type copy{};
            for (std::size_t i = 0; i < detail::product(shape()); ++i)
            {
                copy.elements[i] = m_data[i];
            }
            return copy;
        }

      private:

        template <class F>
        constexpr fixed_view& assign(const F& other) noexcept
        {
            for (std::size_t i = 0; i < detail::product(shape()); ++i)
            {
                m_data[i] = other.data()[i];
            }
            return *this;
        }

        T* m_data;
    };

    template <scalar T, std::size_t N0, std::size_t N1>
    constexpr fixed<T, N0, N1> fixed_from(const T (&values)[N0][N1]) noexcept // NOLINT(*-avoid-c-arrays): nested braces deduce only C
                                                                              // arrays
    {
        fixed<T, N0, N1> result{};
        for (std::size_t i = 0; i < N0; ++i)
        {
            for (std::size_t j = 0; j < N1; ++j)
            {
                result(i, j) = values[i][j];
            }
        }
        return result;
    }

    // Builds a fixed of rank 2 to 4 from nested braces: tl::fixed_from({{1., 2.}, {3., 4.}}).
    template <scalar T, std::size_t N0, std::size_t N1, std::size_t N2>
    constexpr fixed<T, N0, N1, N2> fixed_from(const T (&values)[N0][N1][N2]) noexcept // NOLINT(*-avoid-c-arrays)
    {
        fixed<T, N0, N1, N2> result{};
        for (std::size_t i = 0; i < N0; ++i)
        {
            result(i) = fixed_from(values[i]);
        }
        return result;
    }

    template <scalar T, std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3>
    constexpr fixed<T, N0, N1, N2, N3> fixed_from(const T (&values)[N0][N1][N2][N3]) noexcept // NOLINT(*-avoid-c-arrays)
    {
        fixed<T, N0, N1, N2, N3> result{};
        for (std::size_t i = 0; i < N0; ++i)
        {
            result(i) = fixed_from(values[i]);
        }
        return result;
    }

    template <scalar T, std::size_t... N>
    struct element_traits<fixed<T, N...>>
    {
        using scalar = T;
        static constexpr std::array<std::size_t, sizeof...(N)> extents{N...};
    };

    // Equality of whole values, between fixed and views of the same shape and element type.
    template <class A, class B>
        requires detail::same_fixed<A, B>
    constexpr bool operator==(const A& a, const B& b) noexcept
    {
        for (std::size_t i = 0; i < detail::product(A::shape()); ++i)
        {
            if (a.data()[i] != b.data()[i])
            {
                return false;
            }
        }
        return true;
    }

    // Element-wise comparisons, as in the Array API: a fixed of booleans of the same shape.
    template <class A, class B>
        requires detail::equality_operands<A, B>
    constexpr detail::mask_of<A, B> equal(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::mask_of<A, B>>(
            [](auto x, auto y)
            {
                return x == y;
            },
            a,
            b);
    }

    template <class A, class B>
        requires detail::equality_operands<A, B>
    constexpr detail::mask_of<A, B> not_equal(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::mask_of<A, B>>(
            [](auto x, auto y)
            {
                return x != y;
            },
            a,
            b);
    }

    template <class A, class B>
        requires detail::ordered_operands<A, B>
    constexpr detail::mask_of<A, B> less(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::mask_of<A, B>>(
            [](auto x, auto y)
            {
                return x < y;
            },
            a,
            b);
    }

    template <class A, class B>
        requires detail::ordered_operands<A, B>
    constexpr detail::mask_of<A, B> less_equal(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::mask_of<A, B>>(
            [](auto x, auto y)
            {
                return x <= y;
            },
            a,
            b);
    }

    template <class A, class B>
        requires detail::ordered_operands<A, B>
    constexpr detail::mask_of<A, B> greater(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::mask_of<A, B>>(
            [](auto x, auto y)
            {
                return x > y;
            },
            a,
            b);
    }

    template <class A, class B>
        requires detail::ordered_operands<A, B>
    constexpr detail::mask_of<A, B> greater_equal(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::mask_of<A, B>>(
            [](auto x, auto y)
            {
                return x >= y;
            },
            a,
            b);
    }

    // Element-wise arithmetic between two fixed of the same shape and element type, or a
    // fixed and a scalar, which is converted to the element type and broadcast.
    template <class A, class B>
        requires detail::arithmetic_operands<A, B>
    constexpr auto operator+(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::result_of<A, B>>(
            [](auto x, auto y)
            {
                return x + y;
            },
            a,
            b);
    }

    template <class A, class B>
        requires detail::arithmetic_operands<A, B>
    constexpr auto operator-(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::result_of<A, B>>(
            [](auto x, auto y)
            {
                return x - y;
            },
            a,
            b);
    }

    template <class A, class B>
        requires detail::arithmetic_operands<A, B>
    constexpr auto operator*(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::result_of<A, B>>(
            [](auto x, auto y)
            {
                return x * y;
            },
            a,
            b);
    }

    template <class A, class B>
        requires detail::arithmetic_operands<A, B>
    constexpr auto operator/(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::result_of<A, B>>(
            [](auto x, auto y)
            {
                return x / y;
            },
            a,
            b);
    }

    template <class A>
        requires detail::arithmetic_operands<A, A>
    constexpr detail::value_of<A> operator-(const A& a) noexcept
    {
        return detail::elementwise<detail::value_of<A>>(
            [](auto x, auto)
            {
                return -x;
            },
            a,
            a);
    }

    // Compound assignment writes into a fixed, or through a view.
    template <class A, class B>
        requires detail::fixed_like<A> && detail::arithmetic_operands<A, B>
    constexpr A&& operator+=(A&& target, const B& operand) noexcept
    {
        detail::update(
            target,
            [](auto x, auto y)
            {
                return x + y;
            },
            operand);
        return std::forward<A>(target);
    }

    template <class A, class B>
        requires detail::fixed_like<A> && detail::arithmetic_operands<A, B>
    constexpr A&& operator-=(A&& target, const B& operand) noexcept
    {
        detail::update(
            target,
            [](auto x, auto y)
            {
                return x - y;
            },
            operand);
        return std::forward<A>(target);
    }

    template <class A, class B>
        requires detail::fixed_like<A> && detail::arithmetic_operands<A, B>
    constexpr A&& operator*=(A&& target, const B& operand) noexcept
    {
        detail::update(
            target,
            [](auto x, auto y)
            {
                return x * y;
            },
            operand);
        return std::forward<A>(target);
    }

    template <class A, class B>
        requires detail::fixed_like<A> && detail::arithmetic_operands<A, B>
    constexpr A&& operator/=(A&& target, const B& operand) noexcept
    {
        detail::update(
            target,
            [](auto x, auto y)
            {
                return x / y;
            },
            operand);
        return std::forward<A>(target);
    }
}
