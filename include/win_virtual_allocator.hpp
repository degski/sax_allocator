
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

#if defined( _MSC_VER )

#    ifndef NOMINMAX
#        define NOMINMAX
#    endif

#    ifndef _AMD64_
#        define _AMD64_
#    endif

#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN_DEFINED
#        define WIN32_LEAN_AND_MEAN
#    endif

#    include <windef.h>
#    include <WinBase.h>
#    include <immintrin.h>

#    ifdef WIN32_LEAN_AND_MEAN_DEFINED
#        undef WIN32_LEAN_AND_MEAN_DEFINED
#        undef WIN32_LEAN_AND_MEAN
#    endif

#else

#    include <sys/mman.h>

#endif

#define VM_VECTOR_USE_HEDLEY 1

#if VM_VECTOR_USE_HEDLEY
#    include <hedley.h>
#else
#    define HEDLEY_LIKELY( expr ) ( !!( expr ) )
#    define HEDLEY_UNLIKELY( expr ) ( !!( expr ) )
#    define HEDLEY_PREDICT( expr, res, perc ) ( !!( expr ) )
#    define HEDLEY_NEVER_INLINE
#    define HEDLEY_ALWAYS_INLINE
#    define HEDLEY_PURE
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <memory>
#include <new>
#include <utility>

#include <plf/plf_colony.h>

template<typename T, std::size_t Capacity = 65'536>
class win_allocator {

    struct win_virtual {

        win_virtual ( ) noexcept : allocate_fp{ &win_virtual::allocate_initial } { };
        ~win_virtual ( ) {
            if ( base_pointer )
                free ( base_pointer, committed );
        }

        void * allocate ( std::size_t size_ ) { return *allocate_fp ( std::forward<std::size_t> ( size_ ) ); }

        void * base_pointer = nullptr;
        void * ( *allocate_fp ) ( std::size_t );
        std::size_t reserved = 0, committed = 0;

        private:
        [[nodiscard]] void * allocate_initial ( std::size_t size_ ) {
            base_pointer = VirtualAlloc ( nullptr, Capacity, MEM_RESERVE, PAGE_READWRITE ); // reserve
            if ( HEDLEY_UNLIKELY ( not VirtualAlloc ( reinterpret_cast<char *> ( base_pointer ) + committed, size_, MEM_COMMIT,
                                                      PAGE_READWRITE ) ) )
                throw std::bad_alloc ( );
            committed += size_;
            allocate_fp = &win_virtual::allocate_regular;
            return std::forward<void *> ( base_pointer );
        }

        [[nodiscard]] void * allocate_regular ( std::size_t size_ ) {
            if ( HEDLEY_UNLIKELY ( not VirtualAlloc ( reinterpret_cast<char *> ( base_pointer ) + committed, size_, MEM_COMMIT,
                                                      PAGE_READWRITE ) ) )
                throw std::bad_alloc ( );
            committed += size_;
            return std::forward<void *> ( base_pointer );
        }

        void free ( void * const pointer_, std::size_t size_ ) noexcept {
            VirtualFree ( pointer_, 0, MEM_RELEASE );
            base_pointer = nullptr, reserved = 0, committed = 0;
        }
    };

    win_virtual data;

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
        using other = win_allocator<U>;
    };

    win_allocator ( ) noexcept                       = default;
    win_allocator ( const win_allocator & ) noexcept = default;
    template<class U>
    win_allocator ( const win_allocator<U> & ) noexcept {}

    win_allocator select_on_container_copy_construction ( ) const { return *this; }

    void deallocate ( T *, size_type ) noexcept { return; }

#if ( __cplusplus >= 201703L ) // C++17
    [[nodiscard]] T * allocate ( size_type count ) { return static_cast<T *> ( return data.allocate ( size_ * sizeof ( T ) ) ); }
    [[nodiscard]] T * allocate ( size_type count, void const * ) { return allocate ( count ); }
#else
    [[nodiscard]] pointer allocate ( size_type, void const * = 0 ) {
        return static_cast<pointer> ( return data.allocate ( size_ * sizeof ( T ) ) );
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

    private:
    // Returns rounded-up multiple of n.
    [[nodiscard]] constexpr std::size_t round_multiple ( std::size_t n_, std::size_t multiple_ ) noexcept {
        return ( ( n_ + multiple_ - 1 ) / multiple_ ) * multiple_;
    }
};

template<typename T, template<typename> typename Allocator>
typename win_allocator<T, Allocator>::colony_container win_allocator<T, Allocator>::nodes;

template<class T1, class T2>
bool operator== ( const win_allocator<T1> &, const win_allocator<T2> & ) noexcept {
    return true;
}
template<class T1, class T2>
bool operator!= ( const win_allocator<T1> &, const win_allocator<T2> & ) noexcept {
    return false;
}
