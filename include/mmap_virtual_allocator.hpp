
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

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <memory>
#include <new>
#include <utility>

#include <plf/plf_colony.h>

template<typename T, template<typename> typename Allocator = std::allocator>
class mmap_allocator {

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
        using other = mmap_allocator<U>;
    };

    mmap_allocator ( ) noexcept                        = default;
    mmap_allocator ( const mmap_allocator & ) noexcept = default;
    template<class U>
    mmap_allocator ( const mmap_allocator<U> & ) noexcept {}

    mmap_allocator select_on_container_copy_construction ( ) const { return *this; }

    void deallocate ( T * ptr_, size_type ) { nodes.erase ( nodes.get_iterator_from_pointer ( ( uninitialized_type * ) ptr_ ) ); }

#if ( __cplusplus >= 201703L ) // C++17
    [[nodiscard]] T * allocate ( size_type size_ ) {
        return static_cast<T *> ( ( void * ) &*( mmap_allocator::nodes.emplace ( ) ) );
    }
    [[nodiscard]] T * allocate ( size_type size_, void const * ) { return allocate ( size_ ); }
#else
    [[nodiscard]] pointer allocate ( size_type, void const * = 0 ) {
        return static_cast<pointer> ( ( void * ) &*( mmap_allocator::nodes.emplace ( ) ) );
    }
#endif

#if ( ( __cplusplus >= 201103L ) || ( _MSC_VER > 1900 ) ) // C++11
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap            = std::true_type;
    using is_always_equal                        = std::true_type;

    template<class U, class... Args>
    void construct ( U * p_, Args &&... args ) {
        ::new ( p_ ) U ( std::forward<Args> ( args )... );
    }
    template<class U>
    void destroy ( U * p_ ) noexcept {
        p_->~U ( );
    }
#else
    void construct ( pointer p_, value_type const & val ) { ::new ( p_ ) value_type ( val ); }
    void destroy ( pointer p_ ) { p_->~value_type ( ); }
#endif

    size_type max_size ( ) const noexcept { return ( PTRDIFF_MAX / sizeof ( value_type ) ); }
    pointer address ( reference x_ ) const { return &x_; }
    const_pointer address ( const_reference x_ ) const { return &x_; }
};

template<typename T, template<typename> typename Allocator>
typename mmap_allocator<T, Allocator>::colony_container mmap_allocator<T, Allocator>::nodes;

template<class T1, class T2>
bool operator== ( const mmap_allocator<T1> &, const mmap_allocator<T2> & ) noexcept {
    return true;
}
template<class T1, class T2>
bool operator!= ( const mmap_allocator<T1> &, const mmap_allocator<T2> & ) noexcept {
    return false;
}
