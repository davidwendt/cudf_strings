#pragma once

typedef unsigned char BYTE;

namespace cudf
{

template<typename T> class category_impl;

template<typename T>
class category
{
    category_impl<T>* pImpl;

    category();
    category( const category& );

public:

    category( const T* items, size_t count, BYTE* nulls=nullptr );
    ~category();

    category<T>* copy();

    size_t size();
    size_t keys_size();

    const T* keys();
    const int* values();
    const BYTE* nulls_bitmask();
    bool has_nulls();

    void print(const char* prefix="", const char* delimiter=" ");

    const T get_key_for(int idx);
    bool is_value_null(int idx);

    int get_index_for(T key);
    size_t get_indexes_for(T key, int* result);
    size_t get_indexes_for_null_key(int* result);

    category<T>* add_keys( const T* items, size_t count, const BYTE* nulls=nullptr );
    category<T>* remove_keys( const T* items, size_t count, const BYTE* nulls=nullptr );
    category<T>* remove_unused_keys();
    category<T>* set_keys( const T* items, size_t count, const BYTE* nulls=nullptr );
    category<T>* merge( category<T>& cat );

    category<T>* gather(const int* indexes, size_t count );
    category<T>* gather_and_remap(const int* indexes, size_t count );
    category<T>* gather_values(const int* indexes, size_t count );

    void to_type( T* results, BYTE* nulls=nullptr ); // must be able to hold size() entries
    void gather_type( const int* indexes, size_t count, T* results, BYTE* nulls=nullptr );
};

}

