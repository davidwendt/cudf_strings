#pragma once
#include <cstddef>

namespace cudf
{

// this will become device-only class
//
class dstring // view-only
{
    size_t _bytes;
    const char* _data; // not owned
public:
    dstring() : _data(nullptr), _bytes(0) {} // necessary
    dstring( const char* str, size_t bytes ) : _bytes(bytes), _data(str) {}
    dstring( const dstring& str ) { _bytes = str._bytes; _data = str._data; }
    ~dstring() {}

    const char* data() const { return _data; }
    size_t size() const { return _bytes; }

    inline dstring& operator=(const dstring& str);

    inline bool operator==(const dstring& rhs) const;
    inline bool operator!=(const dstring& rhs) const;
    inline bool operator<(const dstring& rhs) const;
};

}

#include "dstring.inl"