#include <tl/tl.hpp>

int main()
{
    return tl::library_version() == tl::header_version ? 0 : 1;
}
