// Check that the headers and the linked library come from the same installation.
#include <tl/tl.hpp>

#include <cstdio>

int main()
{
    const tl::version linked = tl::library_version();
    if (linked != tl::header_version)
    {
        std::printf("tensorlib headers %d.%d.%d, library %d.%d.%d\n",
                    tl::header_version.major,
                    tl::header_version.minor,
                    tl::header_version.patch,
                    linked.major,
                    linked.minor,
                    linked.patch);
        return 1;
    }
    return 0;
}
