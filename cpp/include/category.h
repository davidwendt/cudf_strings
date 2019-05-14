#pragma once

#include <thrust/host_vector.h>
#include "dstring.h"

namespace cudf
{

// 
template<typename T> // do not declare with const type
class category_impl
{
public:
    inline void create_keys(const T* items, const int* indexes, size_t ucount, bool includes_null, thrust::host_vector<T>& keys);
    inline void create_keys(const T* items, size_t ucount, bool includes_null, thrust::host_vector<T>& keys);
};

template<> class category_impl<dstring>
{
    char* the_keys_memory;
public:
    category_impl() : the_keys_memory(0) {}
    ~category_impl() { delete the_keys_memory; }
    inline void create_keys(const dstring* items, const int* indexes, size_t ucount, bool includes_null, thrust::host_vector<dstring>& keys);
    inline void create_keys(const dstring* items, size_t ucount, bool includes_null, thrust::host_vector<dstring>& keys);
};

template<typename T, class Impl=category_impl<T> >  // T cannot be const
class category
{
    thrust::host_vector<T> _keys; // should move this into impl; create individual objects on-demand
    thrust::host_vector<int> _values;
    thrust::host_vector<char> _bitmask;
    Impl impl;

    category() {};
    category( const category& ) {};

    inline void init_keys(const T* items, size_t count, bool includes_null=false );
    inline void init_keys(const T* items, const int* indexes, size_t count, bool includes_null=false );
public:

    category( const T* items, size_t count ); // needs bitmask
    ~category() {}

    inline category<T>* copy();

    size_t size()       { return _values.size(); }
    size_t keys_size()  { return _keys.size(); }

    const T* keys()       { return _keys.data(); } // on-demand makes this not possible
    const int* values()   { return _values.data(); }
    const char* bitmask() { return _bitmask.data(); }

    const T get_key_for(int idx) { return _keys[idx]; } //

    inline int get_value_for(T key);
    inline int* get_indexes_for(T key);

    inline category<T>* add_keys( const T* items, size_t count );
    inline category<T>* remove_keys( const T* items, size_t count );
    inline category<T>* remove_unused_keys();
    inline category<T>* set_keys( const T* items, size_t count );
    inline category<T>* merge( category<T>& cat );

    inline category<T>* gather(const int* indexes, size_t count );

    inline void to_type( T* results ); // must be able to hold size() entries
    inline void gather_type( const int* indexes, size_t count, T* results );
};

}

#include "category.inl"