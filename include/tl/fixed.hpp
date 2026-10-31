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
    template <scalar T, std::size_t... N>
        requires(sizeof...(N) >= 1 && sizeof...(N) <= 4 && ((N > 0) && ...))
    struct fixed;

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

    /**
     * @brief A small tensor whose shape is known at compile time, stored inline.
     *
     * A `fixed` holds its elements in a flat array, in C order (the last index varies
     * fastest). It is a plain aggregate: it is trivially copyable, it has exactly the layout
     * of `T[N0][N1]...`, and every operation on it is evaluated eagerly and can be used in
     * constant expressions. It is used for local computations and as the element type of
     * larger tensors, such as a field of 3-vectors.
     *
     * Values are written as a flat list in C order, or with nested braces through
     * tl::fixed_from(). `{}` gives zeros.
     *
     * @code
     * tl::fixed<double, 3> v{1., 2., 3.};
     * tl::fixed<double, 2, 2> m{1., 2., 3., 4.};              // m(1, 0) == 3.
     * tl::fixed<double, 2, 2> same = tl::fixed_from({{1., 2.}, {3., 4.}});
     * @endcode
     *
     * `==` and `!=` compare whole values. tl::equal(), tl::less() and the other Array API
     * comparisons compare element by element.
     *
     * @tparam T Element type, one of tl::scalar.
     * @tparam N Extent of each axis, 1 to 4 axes, each extent at least 1.
     */
    template <scalar T, std::size_t... N>
        requires(sizeof...(N) >= 1 && sizeof...(N) <= 4 && ((N > 0) && ...))
    struct fixed
    {
        /// The elements in C order. Prefer `operator()` to reach them by index.
        std::array<T, (N * ...)> elements;

        /// @brief The extent of each axis. @return `{N...}`.
        static constexpr std::array<std::size_t, sizeof...(N)> shape() noexcept
        {
            return {N...};
        }

        /// @brief The elements in C order. @return Pointer to the first one.
        constexpr T* data() noexcept
        {
            return elements.data();
        }

        /// @brief The elements in C order. @return Pointer to the first one.
        constexpr const T* data() const noexcept
        {
            return elements.data();
        }

        /**
         * @brief Element or block at the given indices.
         *
         * With as many indices as the rank, returns a reference to the element. With fewer,
         * returns a tl::fixed_view of the block those leading indices select: `m(1)` is row 1
         * of a matrix, and writing to it writes into `m`.
         *
         * With `TL_ENABLE_ASSERTIONS` defined, an index past its extent aborts with a message.
         *
         * @param index One index per leading axis, from 1 to the rank.
         * @return `T&`, or a `tl::fixed_view<T, ...>` of the remaining axes.
         */
        template <std::convertible_to<std::size_t>... I>
            requires(sizeof...(I) >= 1 && sizeof...(I) <= sizeof...(N))
        constexpr decltype(auto) operator()(I... index) & noexcept
        {
            return detail::at<N...>(data(), index...);
        }

        /**
         * @brief Element or read-only block at the given indices.
         *
         * @param index One index per leading axis, from 1 to the rank.
         * @return `const T&`, or a `tl::fixed_view<const T, ...>` of the remaining axes.
         */
        template <std::convertible_to<std::size_t>... I>
            requires(sizeof...(I) >= 1 && sizeof...(I) <= sizeof...(N))
        constexpr decltype(auto) operator()(I... index) const& noexcept
        {
            return detail::at<N...>(data(), index...);
        }

        /**
         * @brief Element of a temporary, by value.
         *
         * @param index One index per axis.
         * @return A copy of the element.
         */
        template <std::convertible_to<std::size_t>... I>
            requires(sizeof...(I) == sizeof...(N))
        constexpr T operator()(I... index) && noexcept
        {
            return detail::at<N...>(data(), index...);
        }

        /// A block of a temporary cannot be viewed: the view would dangle.
        template <std::convertible_to<std::size_t>... I>
            requires(sizeof...(I) >= 1 && sizeof...(I) < sizeof...(N))
        void operator()(I... index) && = delete;

        /// True when every element of `a` equals the element of `b` at the same position.
        friend constexpr bool operator==(const fixed& a, const fixed& b) = default;
    };

    /**
     * @brief A reference to a block of a tl::fixed.
     *
     * Indexing a `fixed` with fewer indices than its rank gives a view of the block those
     * indices select. The view refers to the elements of the `fixed`; it does not own them,
     * so it must not outlive it.
     *
     * - Copying a view gives another view of the same elements.
     * - Assigning to a view writes the elements: from a `fixed` or a view of the same shape
     *   and element type, or from a scalar, which fills the block.
     * - Converting a view to tl::fixed copies the values.
     *
     * @code
     * tl::fixed<double, 2, 2> m{};
     * m(1) = tl::fixed<double, 2>{3., 4.};     // writes row 1 of m
     * m(0) = 7.;                               // fills row 0
     * tl::fixed<double, 2> row = m(1);         // a copy: later writes to m do not reach it
     * @endcode
     *
     * @tparam T Element type, `const`-qualified for a view of a const `fixed`, which is
     *           read-only.
     * @tparam N Extent of each axis of the block.
     */
    template <class T, std::size_t... N>
    class fixed_view
    {
      public:

        /// The tl::fixed a view converts to.
        using value_type = fixed<std::remove_const_t<T>, N...>;

        /**
         * @brief View of the block whose first element is `data`.
         *
         * Indexing a tl::fixed creates views; this constructor serves that purpose.
         *
         * @param data First element of a block of shape `N...`, stored in C order.
         */
        constexpr explicit fixed_view(T* data) noexcept
            : m_data(data)
        {
        }

        /// Another view of the same elements.
        constexpr fixed_view(const fixed_view&) noexcept = default;
        /// Another view of the same elements.
        constexpr fixed_view(fixed_view&&) noexcept = default;
        /// Leaves the elements untouched.
        constexpr ~fixed_view() = default;

        /**
         * @brief Writes the elements of `other` into the block.
         *
         * @param other View of the same shape and element type.
         * @return This view.
         */
        // Self-assignment copies each element onto itself, which is harmless.
        constexpr fixed_view& operator=(const fixed_view& other) noexcept // NOLINT(bugprone-unhandled-self-assignment)
            requires(!std::is_const_v<T>)
        {
            return assign(other);
        }

        /**
         * @brief Writes the elements of `other` into the block.
         *
         * @param other View of the same shape and element type.
         * @return This view.
         */
        constexpr fixed_view& operator=(fixed_view&& other) noexcept
            requires(!std::is_const_v<T>)
        {
            return assign(other);
        }

        /**
         * @brief Writes the elements of a tl::fixed or of another view into the block.
         *
         * @param other `fixed` or view of the same shape and element type.
         * @return This view.
         */
        template <class F>
            requires(!std::is_const_v<T> && detail::same_fixed<F, fixed_view>)
        constexpr fixed_view& operator=(const F& other) noexcept
        {
            return assign(other);
        }

        /**
         * @brief Fills the block with one value.
         *
         * @param value Arithmetic value, converted to the element type.
         * @return This view.
         */
        template <class S>
            requires(std::is_arithmetic_v<S> && !std::is_const_v<T>)
        constexpr fixed_view& operator=(S value) noexcept
        {
            for (std::size_t i = 0; i < detail::product(shape()); ++i)
            {
                m_data[i] = static_cast<T>(value);
            }
            return *this;
        }

        /// @brief The extent of each axis of the block. @return `{N...}`.
        static constexpr std::array<std::size_t, sizeof...(N)> shape() noexcept
        {
            return {N...};
        }

        /// @brief The elements of the block in C order. @return Pointer to the first one.
        constexpr T* data() const noexcept
        {
            return m_data;
        }

        /**
         * @brief Element or sub-block at the given indices, as for tl::fixed.
         *
         * @param index One index per leading axis of the block.
         * @return `T&`, or a view of the remaining axes.
         */
        template <std::convertible_to<std::size_t>... I>
            requires(sizeof...(I) >= 1 && sizeof...(I) <= sizeof...(N))
        constexpr decltype(auto) operator()(I... index) const noexcept
        {
            return detail::at<N...>(m_data, index...);
        }

        /// @brief Copies the values the view refers to. @return A tl::fixed holding them.
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

    /**
     * @brief A tl::fixed of rank 2 written with nested braces, one pair per row.
     *
     * @code
     * constexpr auto m = tl::fixed_from({{1., 2.}, {3., 4.}});   // tl::fixed<double, 2, 2>
     * @endcode
     *
     * The flat form `tl::fixed<double, 2, 2>{1., 2., 3., 4.}` gives the same value.
     *
     * @param values Nested braces whose shape gives the extents of the result.
     * @return The tl::fixed holding `values`.
     */
    // Nested braces deduce only C arrays, hence the C-array parameters.
    template <scalar T, std::size_t N0, std::size_t N1>
    constexpr fixed<T, N0, N1> fixed_from(const T (&values)[N0][N1]) noexcept // NOLINT(*-avoid-c-arrays)
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

    /**
     * @brief A tl::fixed of rank 3 written with nested braces.
     *
     * @param values Nested braces whose shape gives the extents of the result.
     * @return The tl::fixed holding `values`.
     */
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

    /**
     * @brief A tl::fixed of rank 4 written with nested braces.
     *
     * @param values Nested braces whose shape gives the extents of the result.
     * @return The tl::fixed holding `values`.
     */
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

    /**
     * @brief Storage of a tl::fixed as an element type: its scalar, repeated over its shape.
     *
     * @tparam T Element type of the `fixed`.
     * @tparam N Extents of the `fixed`.
     */
    template <scalar T, std::size_t... N>
    struct element_traits<fixed<T, N...>>
    {
        /// The element type of the `fixed`.
        using scalar = T;
        /// The shape of the `fixed`.
        static constexpr std::array<std::size_t, sizeof...(N)> extents{N...};
    };

    /**
     * @brief True when two values of the same shape and element type are equal everywhere.
     *
     * Compares a tl::fixed with a tl::fixed_view, or two views. Two tl::fixed compare with
     * their own `operator==`.
     *
     * @param a, b `fixed` or views of the same shape and element type.
     */
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

    /**
     * @brief `a == b` element by element, as in the Array API.
     *
     * @code
     * tl::equal(tl::fixed<double, 3>{1., 2., 3.}, 2.)
     * @endcode
     *
     * @param a, b Two `fixed` or views of the same shape and element type, or one of them and
     *             a scalar, which is converted to the element type.
     * @return A `tl::fixed<bool, N...>` of the same shape.
     */
    template <class A, class B>
        requires detail::equality_operands<A, B>
    constexpr auto equal(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::mask_of<A, B>>(
            [](auto x, auto y)
            {
                return x == y;
            },
            a,
            b);
    }

    /**
     * @brief `a != b` element by element, as in the Array API.
     *
     * @code
     * tl::not_equal(tl::fixed<double, 3>{1., 2., 3.}, 2.)
     * @endcode
     *
     * @param a, b Two `fixed` or views of the same shape and element type, or one of them and
     *             a scalar, which is converted to the element type.
     * @return A `tl::fixed<bool, N...>` of the same shape.
     */
    template <class A, class B>
        requires detail::equality_operands<A, B>
    constexpr auto not_equal(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::mask_of<A, B>>(
            [](auto x, auto y)
            {
                return x != y;
            },
            a,
            b);
    }

    /**
     * @brief `a < b` element by element, as in the Array API.
     *
     * @code
     * tl::less(tl::fixed<double, 3>{1., 2., 3.}, 2.)
     * @endcode
     *
     * @param a, b Two `fixed` or views of the same shape and element type, or one of them and
     *             a scalar, which is converted to the element type. The element type is not `bool`.
     * @return A `tl::fixed<bool, N...>` of the same shape.
     */
    template <class A, class B>
        requires detail::ordered_operands<A, B>
    constexpr auto less(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::mask_of<A, B>>(
            [](auto x, auto y)
            {
                return x < y;
            },
            a,
            b);
    }

    /**
     * @brief `a <= b` element by element, as in the Array API.
     *
     * @code
     * tl::less_equal(tl::fixed<double, 3>{1., 2., 3.}, 2.)
     * @endcode
     *
     * @param a, b Two `fixed` or views of the same shape and element type, or one of them and
     *             a scalar, which is converted to the element type. The element type is not `bool`.
     * @return A `tl::fixed<bool, N...>` of the same shape.
     */
    template <class A, class B>
        requires detail::ordered_operands<A, B>
    constexpr auto less_equal(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::mask_of<A, B>>(
            [](auto x, auto y)
            {
                return x <= y;
            },
            a,
            b);
    }

    /**
     * @brief `a > b` element by element, as in the Array API.
     *
     * @code
     * tl::greater(tl::fixed<double, 3>{1., 2., 3.}, 2.)
     * @endcode
     *
     * @param a, b Two `fixed` or views of the same shape and element type, or one of them and
     *             a scalar, which is converted to the element type. The element type is not `bool`.
     * @return A `tl::fixed<bool, N...>` of the same shape.
     */
    template <class A, class B>
        requires detail::ordered_operands<A, B>
    constexpr auto greater(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::mask_of<A, B>>(
            [](auto x, auto y)
            {
                return x > y;
            },
            a,
            b);
    }

    /**
     * @brief `a >= b` element by element, as in the Array API.
     *
     * @code
     * tl::greater_equal(tl::fixed<double, 3>{1., 2., 3.}, 2.)
     * @endcode
     *
     * @param a, b Two `fixed` or views of the same shape and element type, or one of them and
     *             a scalar, which is converted to the element type. The element type is not `bool`.
     * @return A `tl::fixed<bool, N...>` of the same shape.
     */
    template <class A, class B>
        requires detail::ordered_operands<A, B>
    constexpr auto greater_equal(const A& a, const B& b) noexcept
    {
        return detail::elementwise<detail::mask_of<A, B>>(
            [](auto x, auto y)
            {
                return x >= y;
            },
            a,
            b);
    }

    /**
     * @brief Element-wise sum `a + b`.
     *
     * @param a, b Two `fixed` or views of the same shape and a numeric element type, or one of
     *             them and a scalar, which is converted to the element type and broadcast.
     * @return A tl::fixed of the same shape and element type.
     */
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

    /**
     * @brief Element-wise difference `a - b`.
     *
     * @param a, b Two `fixed` or views of the same shape and a numeric element type, or one of
     *             them and a scalar, which is converted to the element type and broadcast.
     * @return A tl::fixed of the same shape and element type.
     */
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

    /**
     * @brief Element-wise product `a * b`.
     *
     * @param a, b Two `fixed` or views of the same shape and a numeric element type, or one of
     *             them and a scalar, which is converted to the element type and broadcast.
     * @return A tl::fixed of the same shape and element type.
     */
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

    /**
     * @brief Element-wise quotient `a / b`.
     *
     * @param a, b Two `fixed` or views of the same shape and a numeric element type, or one of
     *             them and a scalar, which is converted to the element type and broadcast.
     * @return A tl::fixed of the same shape and element type.
     */
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

    /**
     * @brief Element-wise negation.
     *
     * @param a `fixed` or view with a numeric element type.
     * @return A tl::fixed of the same shape and element type.
     */
    template <class A>
        requires detail::arithmetic_operands<A, A>
    constexpr auto operator-(const A& a) noexcept
    {
        return detail::elementwise<detail::value_of<A>>(
            [](auto x, auto)
            {
                return -x;
            },
            a,
            a);
    }

    /**
     * @brief Adds `operand` to `target`, element by element, in place.
     *
     * Writes into a tl::fixed, or through a tl::fixed_view into the `fixed` it refers to.
     *
     * @param target `fixed` or view with a numeric element type.
     * @param operand `fixed` or view of the same shape and element type, or a scalar, which
     *                is converted to the element type and broadcast.
     * @return `target`.
     */
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

    /**
     * @brief Subtracts `operand` from `target`, element by element, in place.
     *
     * Writes into a tl::fixed, or through a tl::fixed_view into the `fixed` it refers to.
     *
     * @param target `fixed` or view with a numeric element type.
     * @param operand `fixed` or view of the same shape and element type, or a scalar, which
     *                is converted to the element type and broadcast.
     * @return `target`.
     */
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

    /**
     * @brief Multiplies `target` by `operand`, element by element, in place.
     *
     * Writes into a tl::fixed, or through a tl::fixed_view into the `fixed` it refers to.
     *
     * @param target `fixed` or view with a numeric element type.
     * @param operand `fixed` or view of the same shape and element type, or a scalar, which
     *                is converted to the element type and broadcast.
     * @return `target`.
     */
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

    /**
     * @brief Divides `target` by `operand`, element by element, in place.
     *
     * Writes into a tl::fixed, or through a tl::fixed_view into the `fixed` it refers to.
     *
     * @param target `fixed` or view with a numeric element type.
     * @param operand `fixed` or view of the same shape and element type, or a scalar, which
     *                is converted to the element type and broadcast.
     * @return `target`.
     */
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
