# Strict warnings for tensorlib's own targets. They are PRIVATE, so they are never
# imposed on the targets a user links against tensorlib.
function(tl_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-
            $<$<BOOL:${TL_WARNINGS_AS_ERRORS}>:/WX>)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
            $<$<BOOL:${TL_WARNINGS_AS_ERRORS}>:-Werror>)
    endif()
endfunction()
