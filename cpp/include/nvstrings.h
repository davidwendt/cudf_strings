#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

// declaration of strings methods using new column interface

namespace cudf {

// forward references
struct column;
struct column_view;

namespace string {

enum sort_type {
        none=0,    ///< no sorting
        length=1,  ///< sort by string length
        name=2     ///< sort by characters code-points
    };

// create strings column from char-ptr/size pairs pointing to device memory
std::unique_ptr<cudf::column> create_column( const std::pair<const char*,size_t>* strs );

// extract pairs from strings column
void create_index( const cudf::column_view strings_column, std::pair<const char*,size_t>* strs );

// copy column strings to host pointers
void to_host( const cudf::column_view strings_column, char** list );
// print strings to stdout
void print( const cudf::column_view strings_column, uint32_t start=0, int32_t end=-1,
            int32_t max_width=-1, const char* delimiter = "\n" );

// make a copy of a strings column
std::unique_ptr<cudf::column> copy( const cudf::column_view strings_column );

// new strings column from subset of given strings column
std::unique_ptr<cudf::column> sublist( const cudf::column_view strings_column,
                                       uint32_t start, uint32_t end, uint32_t step );

// new strings column by gathering strings from existing column
std::unique_ptr<cudf::column> gather( const cudf::column_view strings_column,
                                      const int32_t* indexes, uint32_t count );

// return sorted version of the given strings column
std::unique_ptr<cudf::column> sort( const cudf::column_view strings_column,
                                    sort_type stype, bool ascending=true, bool nullfirst=true );

// return sorted indexes only -- returns int32 column
std::unique_ptr<cudf::column> order( const cudf::column_view strings_column,
                                     sort_type stype, bool ascending, bool nullfirst=true );

// number of characters for each string -- returns int32 column
std::unique_ptr<cudf::column> character_count( const cudf::column_view strings_column );

// number of bytes for each string -- returns int32 column
std::unique_ptr<cudf::column> byte_count( const cudf::column_view strings_column );

// return true/false for specified attributes -- returns int8 column
std::unique_ptr<cudf::column> is_alnum( const cudf::column_view strings_column );
std::unique_ptr<cudf::column> is_alpha( const cudf::column_view strings_column );
std::unique_ptr<cudf::column> is_digit( const cudf::column_view strings_column );
std::unique_ptr<cudf::column> is_empty( const cudf::column_view strings_column );

// columns with equal numbers of strings are concatenated to form a new column
std::unique_ptr<cudf::column> concatentate( const cudf::column_view strings_column,
                                            const cudf::column_view strings_others,
                                            const char* separator, const char* narep=nullptr);
std::unique_ptr<cudf::column> concatentate( const cudf::column_view strings_column,
                                            std::vector<cudf::column_view>& strings_others,
                                            const char* separator, const char* narep=nullptr);

// strings column is split vertically into new columns
void split( const cudf::column_view strings_column,
            const char* delimiter, int32_t maxsplit,
            std::vector<cudf::column>& results);

// concatentate strings column with itself -- returns new column
std::unique_ptr<cudf::column> repeat( const cudf::column_view strings_column, uint32_t count );

enum pad_side {
        left,   ///< Add padding to the left.
        right,  ///< Add padding to the right.
        both    ///< Add padding equally to the right and left.
    };
// pad each string to specified width -- returns new column
std::unique_ptr<cudf::column> pad( const cudf::column_view strings_column,
                                   uint32_t width, pad_side side,  const char* fillchar=nullptr );

// return new instance with substring of the provided strings column
std::unique_ptr<cudf::column> slice( const cudf::column_view strings_column,
                                     int32_t start=0, int32_t stop=-1, int32_t step=1 );

// extracts substrings using regex pattern -- results is one or more new strings columns
void extract( const cudf::column_view strings_column, const char* pattern,
              std::vector<cudf::column>& results );

// replace each occurrence of the specified str with the target repl -- returns new column
std::unique_ptr<cudf::column> replace( const cudf::column_view strings_column,
                                       const char* str, const char* repl, int32_t maxrepl=-1 );

// replace null-string entries with the specified string(s) -- returns new column
std::unique_ptr<cudf::column> fillna( const cudf::column_view strings_column, const char* str );
std::unique_ptr<cudf::column> fillna( const cudf::column_view strings_column, const cudf::column_view strings_column2 );

// insert string into each string at the specified position -- returns new column
std::unique_ptr<cudf::column> insert( const cudf::column_view strings_column,
                                      const char* repl, int32_t position );

// remove specified character(s) from beginning and end of each string -- returns new column
std::unique_ptr<cudf::column> strip( const cudf::column_view strings_column, const char* to_strip );

// change case of each string -- returns new column
std::unique_ptr<cudf::column> to_lower( const cudf::column_view strings_column );
std::unique_ptr<cudf::column> swapcase( const cudf::column_view strings_column );
std::unique_ptr<cudf::column> capitalize( const cudf::column_view strings_column );

// search for the specified string in each string -- returns int32 column
std::unique_ptr<cudf::column> find( const cudf::column_view strings_column,
                                    const char* str, int32_t start, int32_t end );

// search for the pattern in each string -- returns new columns
void find_all( const cudf::column_view strings_column,
               const char* pattern, std::vector<cudf::column>& results );

// convert strings to numbers -- returns number column
std::unique_ptr<cudf::column> to_integers( const cudf::column_view strings_column );
std::unique_ptr<cudf::column> to_floats( const cudf::column_view strings_column );
// compute hash on each string -- returns uint32 column
std::unique_ptr<cudf::column> hash( const cudf::column& strings_column );

// create strings column from numbers column
std::unique_ptr<cudf::column> from_integers( const cudf::column_view integers_column );
std::unique_ptr<cudf::column> from_floats( const cudf::column_view floats_column );

}
}
