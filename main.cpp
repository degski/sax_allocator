
// MIT License
//
// Copyright (c) 2020 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR Allocator PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "include/plf_colony_allocator.hpp"
#include "include/win_virtual_allocator.hpp"
#include "include/mmap_virtual_allocator.hpp"

#include <atomic>
#include <sax/iostream.hpp>
#include <set>

/*
    -fsanitize = address

    C:\Program Files\LLVM\lib\clang\10.0.0\lib\windows\clang_rt.asan_cxx-x86_64.lib
    C:\Program Files\LLVM\lib\clang\10.0.0\lib\windows\clang_rt.asan-preinit-x86_64.lib
    C:\Program Files\LLVM\lib\clang\10.0.0\lib\windows\clang_rt.asan-x86_64.lib

    C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\tbb\lib\intel64_win\vc_mt\tbb.lib
*/

#include <sax/prng_sfc.hpp>
#include <sax/uniform_int_distribution.hpp>

#include <plf/plf_nanotimer.h>

#if defined( NDEBUG )
#    define RANDOM 1
#else
#    define RANDOM 0
#endif

namespace ThreadID {
// Creates a new ID.
[[nodiscard]] inline int get ( bool ) noexcept {
    static std::atomic<int> global_id = 0;
    return global_id++;
}
// Returns ID of this thread.
[[nodiscard]] inline int get ( ) noexcept {
    static thread_local int thread_local_id = get ( false );
    return thread_local_id;
}
} // namespace ThreadID

namespace Rng {
// Chris Doty-Humphrey's Small Fast Chaotic Prng.
[[nodiscard]] inline sax::Rng & generator ( ) noexcept {
    if constexpr ( RANDOM ) {
        static thread_local sax::Rng generator ( sax::os_seed ( ), sax::os_seed ( ), sax::os_seed ( ), sax::os_seed ( ) );
        return generator;
    }
    else {
        static thread_local sax::Rng generator ( sax::fixed_seed ( ) + ThreadID::get ( ) );
        return generator;
    }
}
} // namespace Rng

#undef RANDOM

sax::Rng & rng = Rng::generator ( );

template<typename T, typename C = std::less<T>>
using mi_plf_set = std::set<T, C, mi_colony_node_allocator<T>>;

template<typename T, typename C = std::less<T>>
using stl_plf_set = std::set<T, C, stl_colony_node_allocator<T>>;

int main768 ( ) {

    constexpr std::size_t N = 1'000;

    {
        std::vector<std::size_t, win_allocator<std::size_t>> vctr;

        std::size_t result = 0;

        std::uint64_t duration;
        plf::nanotimer timer;
        timer.start ( );

        for ( std::size_t i = 0; i < N; ++i )
            result = vctr.emplace_back ( i );

        duration = static_cast<std::uint64_t> ( timer.get_elapsed_us ( ) );
        std::cout << std::dec << duration << " us " << result << nl;
    }

    /*
    {
        std::set<std::size_t> set;
        std::size_t result = 0;

        std::uint64_t duration;
        plf::nanotimer timer;
        timer.start ( );

        for ( std::size_t i = 0; i < N; ++i )
            result = *set.emplace ( i ).first;

        duration = static_cast<std::uint64_t> ( timer.get_elapsed_us ( ) );
        std::cout << std::dec << duration << " us " << result << nl;
    }

    {
        stl_plf_set<std::size_t> set;
        std::size_t result = 0;

        std::uint64_t duration;
        plf::nanotimer timer;
        timer.start ( );

        for ( std::size_t i = 0; i < N; ++i )
            result = *set.emplace ( i ).first;

        duration = static_cast<std::uint64_t> ( timer.get_elapsed_us ( ) );
        std::cout << std::dec << duration << " us " << result << nl;
    }

    {
        mi_plf_set<std::size_t> set;
        std::size_t result = 0;

        std::uint64_t duration;
        plf::nanotimer timer;
        timer.start ( );

        for ( std::size_t i = 0; i < N; ++i )
            result = *set.emplace ( i ).first;

        duration = static_cast<std::uint64_t> ( timer.get_elapsed_us ( ) );
        std::cout << std::dec << duration << " us " << result << nl;
    }
    */
    return EXIT_SUCCESS;
}

template<typename Type>
struct Foo {
    Type m_member;
};

template<template<typename Type> class TemplateType>
struct Bar {
    TemplateType<int> m_ints;
};

template<template<template<typename> class> class TemplateTemplateType>
struct Baz {
    TemplateTemplateType<Foo> m_foos;
};

using Example = Baz<Bar>;

int main980780 ( ) {

    Example e;

    e.m_foos.m_ints.m_member = 42;
    std::cout << e.m_foos.m_ints.m_member << nl;

    return EXIT_SUCCESS;
}
