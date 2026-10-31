// PROTOTYPE - throwaway code for the wayfinder ticket "Prototype of multi-output fusion".
// Not production code. It answers one question: does a per-thread pending window,
// evaluated as a whole on a trigger, turn `c = a + b; d = a + 2*b` into one loop,
// and what does it cost compared to two loops and to xtensor?
//
// The fused kernel is a block interpreter (numexpr style): each block of BLOCK
// elements runs every instruction of the program over a cache-resident scratch
// buffer. A JIT would emit the straight-line loop instead; the "fused by hand"
// baseline stands for that upper bound.

#pragma once

#include <Kokkos_Core.hpp>

#include <chrono>
#include <memory>
#include <unordered_map>
#include <vector>

namespace tl
{
    using view_t = Kokkos::View<double*>;

    // ---------------------------------------------------------------- graph

    enum class op : unsigned char { input, add, sub, mul, add_s, mul_s };

    struct buffer
    {
        view_t v;
    };

    struct node
    {
        op kind;
        int a = -1, b = -1;
        double s = 0;
        std::shared_ptr<buffer> buf; // inputs only: captured value (copy-on-write)
    };

    struct state // what a tl::tensor handle points to
    {
        std::size_t n = 0;
        int pending = -1;            // node id in the current window, -1 once evaluated
        std::shared_ptr<buffer> buf; // set by evaluation
        std::shared_ptr<buffer> recycle; // previous buffer of a uniquely held handle: reusable in place
    };

    inline bool reuse_buffers = true; // toggled by the benchmark

    struct window
    {
        std::vector<node> nodes;
        std::vector<std::weak_ptr<state>> outputs;
        std::unordered_map<buffer*, int> input_ids;
        std::size_t n = 0;
    };

    inline thread_local window win;

    struct stats
    {
        long kernels = 0;
        long evaluations = 0;
        int last_program_size = 0;
        int last_scratch_slots = 0;
    };
    inline stats st;

    struct expr
    {
        int id;
        std::size_t n;
    };

    class tensor;
    int operand(const tensor& t);

    inline int push(node nd)
    {
        win.nodes.push_back(std::move(nd));
        return static_cast<int>(win.nodes.size()) - 1;
    }

    inline expr operator+(expr x, expr y) { return {push({op::add, x.id, y.id}), x.n}; }
    inline expr operator-(expr x, expr y) { return {push({op::sub, x.id, y.id}), x.n}; }
    inline expr operator*(expr x, expr y) { return {push({op::mul, x.id, y.id}), x.n}; }
    inline expr operator+(expr x, double s) { return {push({op::add_s, x.id, -1, s}), x.n}; }
    inline expr operator*(double s, expr x) { return {push({op::mul_s, x.id, -1, s}), x.n}; }

    void evaluate();

    class tensor
    {
    public:
        explicit tensor(std::size_t n)
            : m_st(std::make_shared<state>())
        {
            m_st->n = n;
            m_st->buf = std::make_shared<buffer>(buffer{view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "t"), n)});
        }

        tensor(expr e) { bind(e); }
        tensor& operator=(expr e)
        {
            bind(e);
            return *this;
        }

        operator expr() const { return {operand(*this), m_st->n}; }

        double operator()(std::size_t i) const // host read: a trigger
        {
            if (m_st->pending >= 0)
            {
                evaluate();
            }
            Kokkos::fence();
            return m_st->buf->v(i);
        }

        view_t raw() const { return m_st->buf->v; }
        const std::shared_ptr<state>& get_state() const { return m_st; }

    private:
        void bind(expr e) // value semantics: a fresh state, old buffers live on if captured
        {
            std::shared_ptr<buffer> old;
            // Copy-on-write uniqueness check: nobody else sees the old value, so its storage can be rewritten.
            if (reuse_buffers && m_st && m_st.use_count() == 1 && m_st->buf && m_st->buf.use_count() == 1
                && m_st->n == e.n)
            {
                old = std::move(m_st->buf);
            }
            m_st = std::make_shared<state>();
            m_st->recycle = std::move(old);
            m_st->n = e.n;
            m_st->pending = e.id;
            win.outputs.push_back(m_st);
            win.n = e.n;
        }

        std::shared_ptr<state> m_st;
    };

    inline int operand(const tensor& t)
    {
        const auto& s = t.get_state();
        if (s->pending >= 0)
        {
            return s->pending;
        }
        auto [it, inserted] = win.input_ids.try_emplace(s->buf.get(), -1);
        if (inserted)
        {
            node nd{op::input};
            nd.buf = s->buf;
            it->second = push(std::move(nd));
        }
        return it->second;
    }

    inline expr operator+(const tensor& x, const tensor& y) { return expr(x) + expr(y); }
    inline expr operator+(const tensor& x, expr y) { return expr(x) + y; }
    inline expr operator+(expr x, const tensor& y) { return x + expr(y); }
    inline expr operator*(double s, const tensor& x) { return s * expr(x); }
    inline expr operator*(const tensor& x, const tensor& y) { return expr(x) * expr(y); }

    // ------------------------------------------------------ compile and run

    constexpr int BLOCK = 512;
    constexpr int MAX_SLOTS = 16;
    constexpr int MAX_PTRS = 32;

    // Operand encoding: id >= 0 is a scratch slot, id < 0 is pointer table entry -id-1.
    struct instr
    {
        op kind;
        int dst, a, b;
        double s;
    };

    struct program
    {
        instr code[64];
        int size = 0;
        double* ptrs[MAX_PTRS];
        int nptrs = 0;
        int nslots = 0;
    };

    inline void run(const program& p, std::size_t n)
    {
        const std::size_t nblocks = (n + BLOCK - 1) / BLOCK;
        ++st.kernels;
        Kokkos::parallel_for(
            "tl::fused", Kokkos::RangePolicy<>(0, nblocks), KOKKOS_LAMBDA(std::size_t blk) {
                alignas(64) double scratch[MAX_SLOTS][BLOCK];
                const std::size_t i0 = blk * BLOCK;
                const int len = static_cast<int>(i0 + BLOCK <= n ? BLOCK : n - i0);
                auto at = [&](int id) -> double* { return id >= 0 ? scratch[id] : p.ptrs[-id - 1] + i0; };
                for (int k = 0; k < p.size; ++k)
                {
                    const instr& in = p.code[k];
                    double* __restrict d = at(in.dst);
                    const double* __restrict x = at(in.a);
                    const double s = in.s;
                    switch (in.kind)
                    {
                    case op::add: {
                        const double* __restrict y = at(in.b);
                        for (int i = 0; i < len; ++i) d[i] = x[i] + y[i];
                        break;
                    }
                    case op::sub: {
                        const double* __restrict y = at(in.b);
                        for (int i = 0; i < len; ++i) d[i] = x[i] - y[i];
                        break;
                    }
                    case op::mul: {
                        const double* __restrict y = at(in.b);
                        for (int i = 0; i < len; ++i) d[i] = x[i] * y[i];
                        break;
                    }
                    case op::add_s:
                        for (int i = 0; i < len; ++i) d[i] = x[i] + s;
                        break;
                    case op::mul_s:
                        for (int i = 0; i < len; ++i) d[i] = s * x[i];
                        break;
                    default:
                        break;
                    }
                }
            });
    }

    inline void evaluate()
    {
        ++st.evaluations;
        const int nn = static_cast<int>(win.nodes.size());

        // 1. Live outputs: handles still alive and still pending.
        std::vector<std::shared_ptr<state>> live;
        for (auto& w : win.outputs)
        {
            if (auto s = w.lock(); s && s->pending >= 0)
            {
                live.push_back(std::move(s));
            }
        }

        // 2. Keep only nodes reachable from live outputs (dead temporaries vanish).
        std::vector<char> reach(nn, 0);
        std::vector<std::shared_ptr<buffer>> out_buf(nn);
        for (auto& s : live)
        {
            reach[s->pending] = 1;
            if (!out_buf[s->pending] && s->recycle)
            {
                out_buf[s->pending] = std::move(s->recycle);
            }
            else if (!out_buf[s->pending])
            {
                out_buf[s->pending] = std::make_shared<buffer>(
                    buffer{view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "out"), s->n)});
            }
        }
        for (int i = nn - 1; i >= 0; --i)
        {
            if (reach[i])
            {
                if (win.nodes[i].a >= 0) reach[win.nodes[i].a] = 1;
                if (win.nodes[i].b >= 0) reach[win.nodes[i].b] = 1;
            }
        }

        // 3. Last use of every node, for scratch slot reuse.
        std::vector<int> last_use(nn, -1);
        for (int i = 0; i < nn; ++i)
        {
            if (!reach[i]) continue;
            if (win.nodes[i].a >= 0) last_use[win.nodes[i].a] = i;
            if (win.nodes[i].b >= 0) last_use[win.nodes[i].b] = i;
        }

        // 4. Linear program in recording order (already topological).
        program p;
        std::vector<int> loc(nn, 0);
        std::vector<int> free_slots;
        for (int i = 0; i < nn; ++i)
        {
            if (!reach[i]) continue;
            const node& nd = win.nodes[i];
            if (nd.kind == op::input)
            {
                p.ptrs[p.nptrs] = nd.buf->v.data();
                loc[i] = -(++p.nptrs);
                continue;
            }
            int dst;
            if (out_buf[i])
            {
                p.ptrs[p.nptrs] = out_buf[i]->v.data();
                dst = -(++p.nptrs);
            }
            else if (!free_slots.empty())
            {
                dst = free_slots.back();
                free_slots.pop_back();
            }
            else
            {
                dst = p.nslots++;
            }
            loc[i] = dst;
            p.code[p.size++] = {nd.kind, dst, loc[nd.a], nd.b >= 0 ? loc[nd.b] : 0, nd.s};
            for (int src : {nd.a, nd.b})
            {
                if (src >= 0 && last_use[src] == i && loc[src] >= 0) free_slots.push_back(loc[src]);
            }
        }
        st.last_program_size = p.size;
        st.last_scratch_slots = p.nslots;

        // 5. Launch once (asynchronous), then publish the buffers.
        run(p, win.n);
        for (auto& s : live)
        {
            s->buf = out_buf[s->pending];
            s->pending = -1;
        }
        win.nodes.clear();
        win.outputs.clear();
        win.input_ids.clear();
    }
} // namespace tl
