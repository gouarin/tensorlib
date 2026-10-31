// PROTOTYPE - throwaway. Minimal C++20 expression templates with 2-D broadcasting.
// The "lighter templates" baseline: shallow node types, concepts, <cmath> only,
// evaluated on assignment like xtensor, so it fuses only within one assignment.
#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>

namespace et
{
    template <class E>
    concept expression = requires(const E& e, std::size_t i, std::size_t j) {
        { e.at(i, j) } -> std::convertible_to<double>;
        { e.rows() } -> std::convertible_to<std::size_t>;
    };

    class tensor
    {
    public:
        tensor(std::size_t r, std::size_t c)
            : m_r(r), m_c(c), m_p(static_cast<double*>(std::aligned_alloc(64, ((r * c * 8 + 63) / 64) * 64)))
        {
            for (std::size_t i = 0; i < r * c; ++i) m_p[i] = 0;
        }
        tensor(const tensor&) = delete;
        tensor(tensor&& o) noexcept : m_r(o.m_r), m_c(o.m_c), m_p(o.m_p) { o.m_p = nullptr; }
        ~tensor() { std::free(m_p); }
        template <expression E>
        tensor(const E& e) : tensor(e.rows(), e.cols()) { assign(e); }
        template <expression E>
        tensor& operator=(const E& e) { assign(e); return *this; }

        double at(std::size_t i, std::size_t j) const { return m_p[(m_r == 1 ? 0 : i) * m_c + (m_c == 1 ? 0 : j)]; }
        std::size_t rows() const { return m_r; }
        std::size_t cols() const { return m_c; }
        double* data() { return m_p; }

    private:
        template <class E>
        void assign(const E& e)
        {
            const std::size_t R = m_r, C = m_c;
            double* p = m_p;
#pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < R; ++i)
                for (std::size_t j = 0; j < C; ++j) p[i * C + j] = e.at(i, j);
        }
        std::size_t m_r, m_c;
        double* m_p;
    };

    // Leaves are held by reference, inner nodes by value.
    template <class E>
    using hold = std::conditional_t<std::is_same_v<E, tensor>, const tensor&, E>;

    template <class F, class A, class B>
    struct binary
    {
        hold<A> a;
        hold<B> b;
        double at(std::size_t i, std::size_t j) const { return F{}(a.at(i, j), b.at(i, j)); }
        std::size_t rows() const { return a.rows() > b.rows() ? a.rows() : b.rows(); }
        std::size_t cols() const { return a.cols() > b.cols() ? a.cols() : b.cols(); }
    };
    template <class F, class A>
    struct unary
    {
        hold<A> a;
        double s;
        double at(std::size_t i, std::size_t j) const { return F{}(a.at(i, j), s); }
        std::size_t rows() const { return a.rows(); }
        std::size_t cols() const { return a.cols(); }
    };

    struct add { double operator()(double x, double y) const { return x + y; } };
    struct sub { double operator()(double x, double y) const { return x - y; } };
    struct mul { double operator()(double x, double y) const { return x * y; } };
    struct dv { double operator()(double x, double y) const { return x / y; } };
    struct add_s { double operator()(double x, double s) const { return s + x; } };
    struct mul_s { double operator()(double x, double s) const { return s * x; } };
    struct rsub_s { double operator()(double x, double s) const { return s - x; } };
    struct neg { double operator()(double x, double) const { return -x; } };
    struct f_sqrt { double operator()(double x, double) const { return std::sqrt(x); } };
    struct f_exp { double operator()(double x, double) const { return std::exp(x); } };
    struct f_sin { double operator()(double x, double) const { return std::sin(x); } };
    struct f_cos { double operator()(double x, double) const { return std::cos(x); } };
    struct f_tanh { double operator()(double x, double) const { return std::tanh(x); } };

    template <expression A, expression B> auto operator+(const A& a, const B& b) { return binary<add, A, B>{a, b}; }
    template <expression A, expression B> auto operator-(const A& a, const B& b) { return binary<sub, A, B>{a, b}; }
    template <expression A, expression B> auto operator*(const A& a, const B& b) { return binary<mul, A, B>{a, b}; }
    template <expression A, expression B> auto operator/(const A& a, const B& b) { return binary<dv, A, B>{a, b}; }
    template <expression A> auto operator+(double s, const A& a) { return unary<add_s, A>{a, s}; }
    template <expression A> auto operator*(double s, const A& a) { return unary<mul_s, A>{a, s}; }
    template <expression A> auto operator-(double s, const A& a) { return unary<rsub_s, A>{a, s}; }
    template <expression A> auto operator-(const A& a) { return unary<neg, A>{a, 0}; }
    template <expression A> auto sqrt(const A& a) { return unary<f_sqrt, A>{a, 0}; }
    template <expression A> auto exp(const A& a) { return unary<f_exp, A>{a, 0}; }
    template <expression A> auto sin(const A& a) { return unary<f_sin, A>{a, 0}; }
    template <expression A> auto cos(const A& a) { return unary<f_cos, A>{a, 0}; }
    template <expression A> auto tanh(const A& a) { return unary<f_tanh, A>{a, 0}; }
} // namespace et
