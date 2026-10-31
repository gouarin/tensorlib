#pragma once
#include <tl/detail/expr.hpp>

namespace tl
{
    template <operand A, operand B>
    expr add(const A& a, const B& b)
    {
        return detail::add(as_expr(a), as_expr(b));
    }

    template <operand A>
    expr twice(const A& a)
    {
        const expr e = as_expr(a);
        return detail::add(e, e);
    }
}
