/*
*/
#pragma once

#include <cuda_runtime.h>

// utf8 characters are 1-4 bytes
typedef unsigned int Char;

//
//
namespace cudf
{

class string_view
{
    const char* _data;
    unsigned int _bytes;

    // utilities
    __device__ unsigned int char_offset(unsigned int bytepos) const;

public:

    __device__ string_view();
    __device__ string_view(const char* data, unsigned int bytes);
    __device__ string_view(const char* data);
    __device__ string_view(const string_view&);
    __device__ string_view(string_view&&);
    __device__ ~string_view();

    __device__ string_view& operator=(const string_view&);
    __device__ string_view& operator=(string_view&&);

    //
    __device__ unsigned int size() const;        // same as length()
    __device__ unsigned int length() const;      // number of bytes
    __device__ unsigned int chars_count() const; // number of characters
    __device__ const char* data() const;

    // returns true if string has no characters
    __device__ bool empty() const;
    // experimental
    __device__ bool is_null() const;

    // iterator is read-only
    class iterator
    {
        const char* p;
        unsigned int cpos, offset;
    public:
        __device__ iterator(const string_view& str,unsigned int initPos);
        __device__ iterator(const iterator& mit);
        __device__ iterator& operator++();
        __device__ iterator operator++(int);
        __device__ bool operator==(const iterator& rhs) const;
        __device__ bool operator!=(const iterator& rhs) const;
        __device__ Char operator*() const;
        __device__ unsigned int position() const;
        __device__ unsigned int byte_offset() const;
    };
    // iterator methods
    __device__ iterator begin() const;
    __device__ iterator end() const;

    // return character (UTF-8) at given position
    __device__ Char at(unsigned int pos) const;
    // this is read-only right now since modifying an individual character may change the memory requirements
    __device__ Char operator[](unsigned int pos) const;
    // return the byte offset for a character position
    __device__ unsigned int byte_offset_for(unsigned int pos) const;

    // return 0 if arg string matches
    // return <0 or >0 depending first different character
    __device__ int compare(const string_view& str) const;
    __device__ int compare(const char* data, unsigned int bytes) const;

    __device__ bool operator==(const string_view& rhs) const;
    __device__ bool operator!=(const string_view& rhs) const;
    __device__ bool operator<(const string_view& rhs) const;
    __device__ bool operator>(const string_view& rhs) const;
    __device__ bool operator<=(const string_view& rhs) const;
    __device__ bool operator>=(const string_view& rhs) const;

    // return character position if arg string is contained in this string
    // return -1 if string is not found
    // (pos,pos+count) is the range of this string that is scanned
    __device__ int find( const string_view& str, unsigned int pos=0, int count=-1 ) const;
    __device__ int find( const char* str, unsigned int bytes, unsigned int pos=0, int count=-1 ) const;
    __device__ int find( Char chr, unsigned int pos=0, int count=-1 ) const;
    // same as find() but searches from the end of this string
    __device__ int rfind( const string_view& str, unsigned int pos=0, int count=-1 ) const;
    __device__ int rfind( const char* str, unsigned int bytes, unsigned int pos=0, int count=-1 ) const;
    __device__ int rfind( Char chr, unsigned int pos=0, int count=-1 ) const;

    //
    __device__ string_view substr( unsigned int start, unsigned int length ) const;

    // tokenizes string around the given delimiter string upto count
    // call with strs=0, will return the number of string tokens
    __device__ unsigned int split( const char* delim, int count, string_view* strs ) const;
    __device__ unsigned int rsplit( const char* delim, int count, string_view* strs ) const;

    // some utilities for handling individual UTF-8 characters
    __host__ __device__ static unsigned int bytes_in_char( Char chr );
    __host__ __device__ static unsigned int char_to_Char( const char* str, Char& chr );
    __host__ __device__ static unsigned int Char_to_char( Char chr, char* str );
    __host__ __device__ static unsigned int chars_in_string( const char* str, unsigned int bytes );
};

}

#include "./string_view.inl"
