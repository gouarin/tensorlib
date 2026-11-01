
bench_demo:     file format elf64-x86-64

0000000000001000 <void (anonymous namespace)::kernel::expression_fixed<tl::fixed<double, 32ul> >(tl::fixed<double, 32ul> const*, tl::fixed<double, 32ul>*)>:
    1000:	mov    (%rdi),%rax
    1004:	call   1100 <tl::fixed<double, 32ul> tl::detail::elementwise<tl::fixed<double, 32ul> >(tl::fixed<double, 32ul> const&)>
    1009:	ret

0000000000001010 <void (anonymous namespace)::kernel::expression_loop<tl::fixed<double, 32ul> >(tl::fixed<double, 32ul> const*, tl::fixed<double, 32ul>*)>:
    1010:	mov    (%rdi),%rax
    1014:	ret
