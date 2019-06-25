#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

// declaration of strings methods using column interface

namespace cudf {
// forward references
struct column;
struct column_view;

namespace custr {

enum sort_type {
        none=0,    ///< no sorting
        length=1,  ///< sort by string length
        name=2     ///< sort by characters code-points
    };


// create strings column from char-ptr/size pairs pointing to device memory
cudf::column* create_column( const std::pair<const char*,size_t>* strs );

// extract pairs from strings column
void create_index( const cudf::column& strings_column, std::pair<const char*,size_t>* strs );

// copy column strings to host pointers
void to_host( const cudf::column& strings_column, char** list );

// make a copy of a strings column
cudf::column* copy( const cudf::column& strings_column );

// new strings column from subset of given strings column
cudf::column* sublist( const cudf::column& strings_column, uint32_t start, uint32_t end, uint32_t step );

// new strings column by gathering strings from existing column
cudf::column* gather( const cudf::column& strings_column, const int32_t* indexes, uint32_t count );

// return sorted version of the given strings column
cudf::column* sort( const cudf::column& strings_column, sort_type stype, bool ascending=true, bool nullfirst=true );

// return sorted indexes only
void order( const cudf::column& strings_column, sort_type stype, bool ascending, uint32_t* indexes, bool nullfirst=true );

// number of characters for each string
void character_count( const cudf::column& strings_column, int32_t* lengths );

// number of bytes for each string
void byte_count( const cudf::column& strings_column, int32_t* lengths );

// return true/false for specified attributes
void is_alnum( const cudf::column& strings_column, int8_t* results );
void is_alpha( const cudf::column& strings_column, int8_t* results );
void is_digit( const cudf::column& strings_column, int8_t* results );
void is_empty( const cudf::column& strings_column, int8_t* results );

// columns with equal numbers of strings are concatenated to form a new column
cudf::column* concatentate( const cudf::column& strings_column, cudf::column& strings_others, const char* separator, const char* narep=nullptr);
cudf::column* concatentate( const cudf::column& strings_column, std::vector<cudf::column_view>& strings_others, const char* separator, const char* narep=nullptr);

// strings column is split vertically into new columns
void split( const cudf::column& strings_column, const char* delimiter, int32_t maxsplit, std::vector<cudf::column>& results);

// concatentate strings column with itself -- returns new column
cudf::column* repeat( const cudf::column& strings_column, uint32_t count);

// pad each string to specified width -- returns new column
cudf::column* left_justify( const cudf::column& strings_column, uint32_t width, const char* fillchar=nullptr );

// return new instance with substring of the provided strings column
cudf::column* slice( const cudf::column& strings_column, int32_t start=0, int32_t stop=-1, int32_t step=1 );

// extracts substrings using regex pattern -- results is one or more new strings columns
void extract( const cudf::column& strings_column, const char* pattern, std::vector<cudf::column>& results );

// replace each occurrence of the specified str with the target repl -- returns new column
cudf::column* replace( const cudf::column& strings_column, const char* str, const char* repl, int32_t maxrepl=-1 );

// replace null-string entries with the specified string(s) -- returns new column
cudf::column* fillna( const cudf::column& strings_column, const char* str );
cudf::column* fillna( const cudf::column& strings_column, const cudf::column& strings_column2 );

// insert string into each string at the specified position -- returns new column
cudf::column* insert( const cudf::column& strings_column, const char* repl, int32_t position );

// remove specified character(s) from beginning and end of each string -- returns new column
cudf::column* strip( const cudf::column& strings_column, const char* to_strip );

// change case of each string -- returns new column
cudf::column* to_lower( const cudf::column& strings_column );
cudf::column* swapcase( const cudf::column& strings_column );
cudf::column* capitalize( const cudf::column& strings_column );

// search for the specified string in each string using the specified range of character positions
uint32_t find( const cudf::column& strings_column, const char* str, int32_t start, int32_t end, int32_t* results );

// convert strings to numbers
void to_integers( const cudf::column& strings_column, int32_t* results );
void to_floats( const cudf::column& strings_column, float* results );
// perform has on each string
void hash( const cudf::column& strings_column, uint32_t* results );

// create strings column from numbers column
cudf::column* from_integers( const int32_t* values, uint32_t count, const int8_t* bitmask );
cudf::column* from_integers( cudf::column& integers_column );


}
}
