// PROTOTYPE - throwaway. Library side of tl_front.hpp, compiled once.
// Pending window + block interpreter with 2-D broadcasting, OpenMP over rows.
// It stands in for the real backend; only its compile-time isolation matters here.
#include "tl_front.hpp"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace tl::detail
{
    struct buffer
    {
        explicit buffer(std::size_t n) : p(static_cast<double*>(std::aligned_alloc(64, ((n * 8 + 63) / 64) * 64))) {}
        ~buffer() { std::free(p); }
        double* p;
    };

    struct impl
    {
        int rc = 1;
        std::size_t rows = 0, cols = 0;
        int pending = -1;
        std::shared_ptr<buffer> buf, recycle;
    };

    enum class kind : unsigned char { input, compute };

    struct node
    {
        kind k;
        op o;
        int a = -1, b = -1;
        double s = 0;
        std::size_t rows, cols;
        std::shared_ptr<buffer> in; // inputs: captured value
    };

    struct window
    {
        std::vector<node> nodes;
        std::vector<impl*> outputs; // the window holds one reference to each
        std::unordered_map<buffer*, int> inputs;
    };
    thread_local window win;

    impl* make(std::size_t r, std::size_t c)
    {
        auto* m = new impl;
        m->rows = r;
        m->cols = c;
        m->buf = std::make_shared<buffer>(r * c);
        for (std::size_t i = 0; i < r * c; ++i) m->buf->p[i] = 0;
        return m;
    }
    void retain(impl* m) { if (m) ++m->rc; }
    void release(impl* m) { if (m && --m->rc == 0) delete m; }

    int push(node n)
    {
        win.nodes.push_back(std::move(n));
        return static_cast<int>(win.nodes.size()) - 1;
    }

    int leaf(impl* m)
    {
        if (m->pending >= 0) return m->pending;
        auto [it, fresh] = win.inputs.try_emplace(m->buf.get(), -1);
        if (fresh) it->second = push({kind::input, op::add, -1, -1, 0, m->rows, m->cols, m->buf});
        return it->second;
    }

    std::size_t bdim(std::size_t x, std::size_t y)
    {
        if (x == y || y == 1) return x;
        if (x == 1) return y;
        throw std::runtime_error("shapes do not broadcast");
    }

    int binary(op o, int a, int b)
    {
        const auto& na = win.nodes[a];
        const auto& nb = win.nodes[b];
        return push({kind::compute, o, a, b, 0, bdim(na.rows, nb.rows), bdim(na.cols, nb.cols), nullptr});
    }
    int scalar(op o, int a, double s)
    {
        return push({kind::compute, o, a, -1, s, win.nodes[a].rows, win.nodes[a].cols, nullptr});
    }
    int unary(op o, int a) { return push({kind::compute, o, a, -1, 0, win.nodes[a].rows, win.nodes[a].cols, nullptr}); }

    impl* bind(impl* old, int id)
    {
        auto* m = new impl;
        m->rows = win.nodes[id].rows;
        m->cols = win.nodes[id].cols;
        m->pending = id;
        if (old && old->rc == 1 && old->buf && old->buf.use_count() == 1 && old->rows == m->rows && old->cols == m->cols)
        {
            m->recycle = std::move(old->buf); // copy-on-write uniqueness: rewrite in place
        }
        release(old);
        ++m->rc; // the window's reference
        win.outputs.push_back(m);
        return m;
    }

    constexpr int BLOCK = 512, MAX_SLOTS = 32;

    struct instr
    {
        op o;
        bool load; // fill a slot from a broadcast input
        int dst, a, b;
        double s;
        const double* src; // load: input base
        std::size_t rs, cs; // load: row and column strides (0 when broadcast)
    };

    void evaluate_group(const std::vector<impl*>& live, std::size_t R, std::size_t C)
    {
        const int nn = static_cast<int>(win.nodes.size());
        std::vector<char> reach(nn, 0);
        std::vector<double*> out_ptr(nn, nullptr);
        for (auto* m : live)
        {
            if (!m->recycle) m->recycle = std::make_shared<buffer>(R * C);
            m->buf = std::move(m->recycle);
            if (!out_ptr[m->pending]) out_ptr[m->pending] = m->buf->p;
            reach[m->pending] = 1;
        }
        for (int i = nn - 1; i >= 0; --i)
        {
            if (!reach[i]) continue;
            if (win.nodes[i].a >= 0) reach[win.nodes[i].a] = 1;
            if (win.nodes[i].b >= 0) reach[win.nodes[i].b] = 1;
        }
        std::vector<int> last(nn, -1);
        for (int i = 0; i < nn; ++i)
        {
            if (!reach[i]) continue;
            if (win.nodes[i].a >= 0) last[win.nodes[i].a] = i;
            if (win.nodes[i].b >= 0) last[win.nodes[i].b] = i;
        }
        // Operand encoding: >= 0 scratch slot, < 0 full-shape pointer -id-1 (row-major R x C).
        std::vector<instr> code;
        std::vector<double*> ptrs;
        std::vector<int> loc(nn, 0), freel;
        int nslots = 0;
        auto slot = [&] {
            if (!freel.empty()) { int s = freel.back(); freel.pop_back(); return s; }
            if (nslots == MAX_SLOTS) throw std::runtime_error("too many slots");
            return nslots++;
        };
        for (int i = 0; i < nn; ++i)
        {
            if (!reach[i]) continue;
            const node& n = win.nodes[i];
            if (n.k == kind::input)
            {
                if (n.rows == R && n.cols == C)
                {
                    ptrs.push_back(n.in->p);
                    loc[i] = -static_cast<int>(ptrs.size());
                }
                else
                {
                    loc[i] = slot();
                    code.push_back({op::add, true, loc[i], 0, 0, 0, n.in->p, n.rows == 1 ? 0 : n.cols, n.cols == 1 ? 0 : 1});
                }
                continue;
            }
            int dst;
            if (out_ptr[i])
            {
                ptrs.push_back(out_ptr[i]);
                dst = -static_cast<int>(ptrs.size());
            }
            else
            {
                dst = slot();
            }
            code.push_back({n.o, false, dst, loc[n.a], n.b >= 0 ? loc[n.b] : 0, n.s, nullptr, 0, 0});
            loc[i] = dst;
            for (int src : {n.a, n.b})
                if (src >= 0 && last[src] == i && loc[src] >= 0) freel.push_back(loc[src]);
        }

        const std::size_t nblk = (C + BLOCK - 1) / BLOCK;
#pragma omp parallel for schedule(static)
        for (std::size_t t = 0; t < R * nblk; ++t)
        {
            alignas(64) double scratch[MAX_SLOTS][BLOCK];
            const std::size_t r = t / nblk, c0 = (t % nblk) * BLOCK;
            const int len = static_cast<int>(c0 + BLOCK <= C ? BLOCK : C - c0);
            auto at = [&](int id) -> double* { return id >= 0 ? scratch[id] : ptrs[-id - 1] + r * C + c0; };
            for (const instr& in : code)
            {
                double* __restrict d = at(in.dst);
                if (in.load)
                {
                    const double* s = in.src + r * in.rs;
                    if (in.cs == 0) { const double v = s[0]; for (int i = 0; i < len; ++i) d[i] = v; }
                    else for (int i = 0; i < len; ++i) d[i] = s[c0 + i];
                    continue;
                }
                const double* __restrict x = at(in.a);
                const double* __restrict y = at(in.b);
                const double s = in.s;
                switch (in.o)
                {
                case op::add: for (int i = 0; i < len; ++i) d[i] = x[i] + y[i]; break;
                case op::sub: for (int i = 0; i < len; ++i) d[i] = x[i] - y[i]; break;
                case op::mul: for (int i = 0; i < len; ++i) d[i] = x[i] * y[i]; break;
                case op::div: for (int i = 0; i < len; ++i) d[i] = x[i] / y[i]; break;
                case op::add_s: for (int i = 0; i < len; ++i) d[i] = s + x[i]; break;
                case op::mul_s: for (int i = 0; i < len; ++i) d[i] = s * x[i]; break;
                case op::rsub_s: for (int i = 0; i < len; ++i) d[i] = s - x[i]; break;
                case op::neg: for (int i = 0; i < len; ++i) d[i] = -x[i]; break;
                case op::sqrt: for (int i = 0; i < len; ++i) d[i] = std::sqrt(x[i]); break;
                case op::exp: for (int i = 0; i < len; ++i) d[i] = std::exp(x[i]); break;
                case op::sin: for (int i = 0; i < len; ++i) d[i] = std::sin(x[i]); break;
                case op::cos: for (int i = 0; i < len; ++i) d[i] = std::cos(x[i]); break;
                case op::tanh: for (int i = 0; i < len; ++i) d[i] = std::tanh(x[i]); break;
                }
            }
        }
    }

    void evaluate()
    {
        std::vector<impl*> live;
        for (auto* m : win.outputs)
        {
            if (m->rc > 1 && m->pending >= 0) live.push_back(m);
        }
        // One kernel per output shape.
        std::vector<char> done(live.size(), 0);
        for (std::size_t i = 0; i < live.size(); ++i)
        {
            if (done[i]) continue;
            std::vector<impl*> group;
            for (std::size_t j = i; j < live.size(); ++j)
            {
                if (!done[j] && live[j]->rows == live[i]->rows && live[j]->cols == live[i]->cols)
                {
                    group.push_back(live[j]);
                    done[j] = 1;
                }
            }
            evaluate_group(group, live[i]->rows, live[i]->cols);
        }
        for (auto* m : live) m->pending = -1;
        for (auto* m : win.outputs) release(m);
        win.outputs.clear();
        win.nodes.clear();
        win.inputs.clear();
    }

    double read(impl* m, std::size_t i, std::size_t j)
    {
        if (m->pending >= 0) evaluate();
        return m->buf->p[i * m->cols + j];
    }
    double* data(impl* m)
    {
        if (m->pending >= 0) evaluate();
        return m->buf->p;
    }
} // namespace tl::detail

void tl::eval_all() { detail::evaluate(); }
