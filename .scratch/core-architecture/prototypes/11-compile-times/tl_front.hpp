// PROTOTYPE - throwaway. Light front-end of a graph-recording tensor library.
// What a user TU includes: no Kokkos, no heavy std headers, no expression templates.
// Every operator records a node through a non-template function compiled once in tl_lib.cpp.
#pragma once

#include <cstddef>

namespace tl
{
    namespace detail
    {
        enum class op : unsigned char
        {
            add, sub, mul, div, add_s, mul_s, rsub_s, neg, sqrt, exp, sin, cos, tanh
        };
        struct impl; // tensor state, defined in the library
        int leaf(impl*);
        int binary(op, int, int);
        int scalar(op, int, double);
        int unary(op, int);
        impl* make(std::size_t rows, std::size_t cols);
        void retain(impl*);
        void release(impl*);
        impl* bind(impl* old, int node); // returns the state that now owns the value of node
        double read(impl*, std::size_t i, std::size_t j);
        double* data(impl*);
    } // namespace detail

    struct expr
    {
        int id;
    };

    class tensor
    {
    public:
        tensor(std::size_t rows, std::size_t cols) : m(detail::make(rows, cols)) {}
        tensor(expr e) : m(detail::bind(nullptr, e.id)) {}
        tensor(const tensor& o) : m(o.m) { detail::retain(m); }
        tensor& operator=(const tensor& o)
        {
            detail::retain(o.m);
            detail::release(m);
            m = o.m;
            return *this;
        }
        tensor& operator=(expr e)
        {
            m = detail::bind(m, e.id);
            return *this;
        }
        ~tensor() { detail::release(m); }
        operator expr() const { return {detail::leaf(m)}; }
        double operator()(std::size_t i, std::size_t j) const { return detail::read(m, i, j); }
        double* data() { return detail::data(m); }

    private:
        detail::impl* m;
    };

    void eval_all();

    // Operands are anything convertible to expr: tensor or expr. One overload set, no type growth.
    inline expr operator+(expr a, expr b) { return {detail::binary(detail::op::add, a.id, b.id)}; }
    inline expr operator-(expr a, expr b) { return {detail::binary(detail::op::sub, a.id, b.id)}; }
    inline expr operator*(expr a, expr b) { return {detail::binary(detail::op::mul, a.id, b.id)}; }
    inline expr operator/(expr a, expr b) { return {detail::binary(detail::op::div, a.id, b.id)}; }
    inline expr operator+(double s, expr a) { return {detail::scalar(detail::op::add_s, a.id, s)}; }
    inline expr operator*(double s, expr a) { return {detail::scalar(detail::op::mul_s, a.id, s)}; }
    inline expr operator-(double s, expr a) { return {detail::scalar(detail::op::rsub_s, a.id, s)}; }
    inline expr operator-(expr a) { return {detail::unary(detail::op::neg, a.id)}; }
    inline expr sqrt(expr a) { return {detail::unary(detail::op::sqrt, a.id)}; }
    inline expr exp(expr a) { return {detail::unary(detail::op::exp, a.id)}; }
    inline expr sin(expr a) { return {detail::unary(detail::op::sin, a.id)}; }
    inline expr cos(expr a) { return {detail::unary(detail::op::cos, a.id)}; }
    inline expr tanh(expr a) { return {detail::unary(detail::op::tanh, a.id)}; }
} // namespace tl
