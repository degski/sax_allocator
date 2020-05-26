
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <memory>
#include <new>
#include <utility>

#include <plf/plf_colony.h>

template<typename T, template<typename> typename Allocator = std::allocator>
class colony_allocator {

    template<typename U>
    struct uninitialized_value_type {
        using value_type = U;
        alignas ( value_type ) char _[ sizeof ( value_type ) ];
        uninitialized_value_type ( ) noexcept                                  = default;
        uninitialized_value_type ( uninitialized_value_type const & ) noexcept = default;
        uninitialized_value_type ( uninitialized_value_type && ) noexcept      = default;
        [[nodiscard]] uninitialized_value_type & operator= ( uninitialized_value_type const & ) noexcept = default;
        [[nodiscard]] uninitialized_value_type & operator= ( uninitialized_value_type && ) noexcept = default;
    };

    using uninitialized_type = uninitialized_value_type<T>;

    template<typename U>
    using allocator_type = Allocator<U>;
    using allocator      = allocator_type<T>;

    using colony_container = plf::colony<uninitialized_type, typename allocator::template rebind<uninitialized_type>::other>;

    static colony_container nodes;

    public:
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type &;
    using const_reference = value_type const &;
    using pointer         = value_type *;
    using const_pointer   = value_type const *;

    template<class U>
    struct rebind {
        using other = colony_allocator<U>;
    };

    colony_allocator ( ) noexcept                          = default;
    colony_allocator ( const colony_allocator & ) noexcept = default;
    template<class U>
    colony_allocator ( const colony_allocator<U> & ) noexcept {}

    colony_allocator select_on_container_copy_construction ( ) const { return *this; }

    void deallocate ( T * ptr_, size_type ) { nodes.erase ( nodes.get_iterator_from_pointer ( ( uninitialized_type * ) ptr_ ) ); }

#if ( __cplusplus >= 201703L ) // C++17
    [[nodiscard]] T * allocate ( size_type count ) {
        return static_cast<T *> ( ( void * ) &*( colony_allocator::nodes.emplace ( ) ) );
    }
    [[nodiscard]] T * allocate ( size_type count, void const * ) { return allocate ( count ); }
#else
    [[nodiscard]] pointer allocate ( size_type, void const * = 0 ) {
        return static_cast<pointer> ( ( void * ) &*( colony_allocator::nodes.emplace ( ) ) );
    }
#endif

#if ( ( __cplusplus >= 201103L ) || ( _MSC_VER > 1900 ) ) // C++11
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap            = std::true_type;
    using is_always_equal                        = std::true_type;

    template<class U, class... Args>
    void construct ( U * p, Args &&... args ) {
        ::new ( p ) U ( std::forward<Args> ( args )... );
    }
    template<class U>
    void destroy ( U * p ) noexcept {
        p->~U ( );
    }
#else
    void construct ( pointer p, value_type const & val ) { ::new ( p ) value_type ( val ); }
    void destroy ( pointer p ) { p->~value_type ( ); }
#endif

    size_type max_size ( ) const noexcept { return ( PTRDIFF_MAX / sizeof ( value_type ) ); }
    pointer address ( reference x ) const { return &x; }
    const_pointer address ( const_reference x ) const { return &x; }
};

template<typename T, template<typename> typename Allocator>
typename colony_allocator<T, Allocator>::colony_container colony_allocator<T, Allocator>::nodes;

template<class T1, class T2>
bool operator== ( const colony_allocator<T1> &, const colony_allocator<T2> & ) noexcept {
    return true;
}
template<class T1, class T2>
bool operator!= ( const colony_allocator<T1> &, const colony_allocator<T2> & ) noexcept {
    return false;
}

#define USE_MIMALLOC_LTO 1
#include <mimalloc.h>

template<typename T>
using mi_colony_allocator = colony_allocator<T, mi_stl_allocator>;
template<typename T>
using stl_colony_allocator = colony_allocator<T, std::allocator>;
