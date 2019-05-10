
#include <cstring>

namespace cudf
{

// this will become device-only class

inline dstring& dstring::operator=(const dstring& str)
{
    _bytes = str._bytes;
    _data = str._data;
    return *this;
}

inline bool dstring::operator==(const dstring& rhs) const
{
    if( rhs.size() == size() )
        return strncmp(_data,rhs._data,size())==0;
    return false;
}
inline bool dstring::operator!=(const dstring& rhs) const
{
    if( rhs.size() == size() )
        return strncmp(_data,rhs._data,size())!=0;
    return true;
}

inline bool dstring::operator<(const dstring& rhs) const
{
    if( rhs.size() == size() )
        return strncmp(_data,rhs._data,size())<0;
    size_t minlen = ( size() < rhs.size() ? size() : rhs.size() );
    if( strncmp(_data,rhs._data,minlen) < 0 )
        return true;
    return minlen == size();
}

}
