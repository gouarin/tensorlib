#pragma once

namespace tl
{
    struct expr
    {
    };

    template <class T>
    concept operand = true;

    namespace detail
    {
        expr add(expr, expr);

        template <operand A>
        expr helper(A a)
        {
            expr e;
            (void)a;
            return e; // internal: not a facade
        }
    }

    template <operand A>
    expr as_expr(const A&)
    {
        return {};
    }
}
