
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

#include <functional>
#include <memory>
#include <new>
#include <utility>
#include <variant>

#include <sax/iostream.hpp>

template<typename T, std::size_t SegmentSize = 65'536,
         std::size_t Capacity = 1'024 * SegmentSize> // An allocator that reserves a region of 64MB and allocates up to
                                                     // a maximum of 1'024 segments of 64KB.
struct alignas ( 32 ) win_allocator {

    [[nodiscard]] HEDLEY_ALWAYS_INLINE static constexpr std::size_t round_multiple ( std::size_t n_ ) noexcept;
    [[nodiscard]] HEDLEY_ALWAYS_INLINE static constexpr std::size_t round_multiple ( std::size_t n_,
                                                                                     std::size_t multiple_ ) noexcept;
    [[nodiscard]] HEDLEY_ALWAYS_INLINE static void * round_multiple ( void * pointer_, std::size_t multiple_ ) noexcept;

    // public:
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type &;
    using const_reference = value_type const &;
    using pointer         = value_type *;
    using const_pointer   = value_type const *;

    // private:
    static constexpr std::size_t windows_minimum_segement_size = 65'536,
                                 segment_size                  = round_multiple ( SegmentSize, windows_minimum_segement_size ),
                                 capacity_value                = round_multiple ( Capacity, segment_size );

    [[nodiscard]] void * initiate_implementation ( std::size_t size_ );
    [[nodiscard]] void * reserved_implementation ( std::size_t size_ );
    [[nodiscard]] void * allocate_implementation ( std::size_t size_ );

    void free_implementation ( std::size_t );

    void *begin_pointer = nullptr, *end_pointer = nullptr;
    std::size_t committed                                     = 0;
    void * ( win_allocator::*implementation ) ( std::size_t ) = &win_allocator::initiate_implementation;

    [[nodiscard]] auto copy_implementation ( win_allocator const & o_ ) noexcept {
        return o_.implementation == &o_.win_allocator::reserved_implementation
                   ? &win_allocator::reserved_implementation
                   : o_.implementation == &o_.win_allocator::initiate_implementation ? &win_allocator::initiate_implementation
                                                                                     : &win_allocator::allocate_implementation;
    }

    // public:
    win_allocator ( ) noexcept = default;

    win_allocator ( win_allocator && )      = delete;
    win_allocator ( win_allocator const & ) = delete;

    win_allocator & operator= ( win_allocator const & ) = delete;
    win_allocator & operator= ( win_allocator && ) = delete;

    template<class U>
    win_allocator ( win_allocator<U> const & ) noexcept {}

    ~win_allocator ( ) { free_implementation ( 0 ); }

    template<typename U>
    using allocator_type = win_allocator<U>;
    using allocator      = allocator_type<T>;

    template<class U>
    struct rebind {
        using other = win_allocator<U>;
    };

#if ( __cplusplus >= 201703L ) // C++17
    [[nodiscard]] T * allocate ( size_type size_ ) {
        return static_cast<T *> ( std::invoke ( implementation, this, size_ * sizeof ( T ) ) );
    }
    [[nodiscard]] T * allocate ( size_type size_, void const * ) { return allocate ( size_ ); }
#else
    [[nodiscard]] T * allocate ( size_type size_, void const * = 0 ) {
        return static_cast<T *> ( std::invoke ( implementation, this, size_ * sizeof ( T ) ) );
    }
#endif

    void deallocate ( T *, size_type ) noexcept { return; }

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

template<typename T, std::size_t SegmentSize, std::size_t Capacity>
[[nodiscard]] constexpr std::size_t win_allocator<T, SegmentSize, Capacity>::round_multiple ( std::size_t n_ ) noexcept {
    n_ += segment_size - 1;
    n_ /= segment_size;
    n_ *= segment_size;
    return n_;
}

template<typename T, std::size_t SegmentSize, std::size_t Capacity>
[[nodiscard]] constexpr std::size_t win_allocator<T, SegmentSize, Capacity>::round_multiple ( std::size_t n_,
                                                                                              std::size_t multiple_ ) noexcept {
    n_ += multiple_ - 1;
    n_ /= multiple_;
    n_ *= multiple_;
    return n_;
}

template<typename T, std::size_t SegmentSize, std::size_t Capacity>
[[nodiscard]] void * win_allocator<T, SegmentSize, Capacity>::round_multiple ( void * pointer_, std::size_t multiple_ ) noexcept {
    std::size_t p;
    std::memcpy ( &p, &pointer_, sizeof ( std::size_t ) );
    p = round_multiple ( p, multiple_ );
    std::memcpy ( &pointer_, &p, sizeof ( std::size_t ) );
    return pointer_;
}

template<typename T, std::size_t SegmentSize, std::size_t Capacity>
[[nodiscard]] void * win_allocator<T, SegmentSize, Capacity>::initiate_implementation ( std::size_t size_ ) {
    std::cout << "initiate" << ' ' << size_ << ' ' << this << nl;
    implementation = &win_allocator::reserved_implementation;
    // assert ( ( ( 2 * sizeof ( void * ) ) == ( size_ ) ) );
    return &begin_pointer;
}

template<typename T, std::size_t SegmentSize, std::size_t Capacity>
[[nodiscard]] void * win_allocator<T, SegmentSize, Capacity>::reserved_implementation ( std::size_t size_ ) {
    std::cout << "reserved" << ' ' << size_ << ' ' << this << nl;
    // begin_pointer = end_pointer = VirtualAlloc ( nullptr, size_, MEM_RESERVE, PAGE_READWRITE );
    implementation = &win_allocator::allocate_implementation;
    // return begin_pointer;
    // return VirtualAlloc ( nullptr, size_, MEM_RESERVE, PAGE_READWRITE );
    return nullptr;
}

template<typename T, std::size_t SegmentSize, std::size_t Capacity>
[[nodiscard]] void * win_allocator<T, SegmentSize, Capacity>::allocate_implementation ( std::size_t size_ ) {
    std::cout << "allocate" << ' ' << size_ << nl;
    return nullptr;

    if ( HEDLEY_PREDICT ( ( end_pointer = reinterpret_cast<char *> ( begin_pointer ) + size_ ) >
                              reinterpret_cast<char *> ( begin_pointer ) + committed,
                          true, static_cast<double> ( sizeof ( T ) ) / static_cast<double> ( segment_size ) ) ) {
        if ( HEDLEY_UNLIKELY ( not VirtualAlloc ( reinterpret_cast<char *> ( begin_pointer ) + committed, segment_size, MEM_COMMIT,
                                                  PAGE_READWRITE ) ) )
            throw std::bad_alloc ( );
        committed += segment_size;
    }
    return begin_pointer;
}

template<typename T, std::size_t SegmentSize, std::size_t Capacity>
void win_allocator<T, SegmentSize, Capacity>::free_implementation ( std::size_t ) {
    if ( begin_pointer ) {
        VirtualFree ( begin_pointer, 0, MEM_RELEASE );
        begin_pointer = nullptr, end_pointer = nullptr, committed = 0;
    }
}

template<class T1, class T2>
bool operator== ( const win_allocator<T1> &, const win_allocator<T2> & ) noexcept {
    return true;
}
template<class T1, class T2>
bool operator!= ( const win_allocator<T1> &, const win_allocator<T2> & ) noexcept {
    return false;
}
