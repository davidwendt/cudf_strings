/*
*/

#include <cstdlib>

namespace cudf
{

typedef unsigned char BYTE;


// returns the number of bytes used to represent that char
__host__ __device__ inline static unsigned int bytes_in_char_byte(BYTE byte)
{
    unsigned int count = 1;
    // no if-statements means no divergence
    count += (int)((byte & 0xF0) == 0xF0);
    count += (int)((byte & 0xE0) == 0xE0);
    count += (int)((byte & 0xC0) == 0xC0);
    count -= (int)((byte & 0xC0) == 0x80);
    return count;
}


// utility for methods allowing for multiple delimiters (e.g. strip)
__device__ inline static bool has_one_of( const char* tgts, Char chr )
{
    Char tchr = 0;
    unsigned int cw = string_view::char_to_Char(tgts,tchr);
    while( tchr )
    {
        if( tchr==chr )
            return true;
        tgts += cw;
        cw = string_view::char_to_Char(tgts,tchr);
    }
    return false;
}

__device__ inline static unsigned int string_length( const char* str )
{
    if( !str )
        return 0;
    unsigned int bytes = 0;
    while(*str++)
        ++bytes;
    return bytes;
}


__device__ inline string_view::string_view() : _data(nullptr), _bytes(0)
{
}

__device__ inline string_view::string_view(const char* data, unsigned int bytes)
    : _data(data), _bytes(bytes)
{
}

__device__ inline string_view::string_view(const char* data)
    : _data(data)
{
    _bytes = string_length(data);
}

__device__ inline string_view::string_view(const string_view& src)
{
    _data = src._data;
    _bytes = src._bytes;
}

__device__ inline string_view::string_view(string_view&& src)
{
    _bytes = src._bytes;
    _data = src._data;
    src._bytes = 0;
    src._data = nullptr;
}

__device__ inline string_view::~string_view()
{
}

__device__ inline string_view& string_view::operator=(const string_view& src)
{
    //printf("cp=(%p):%d/(%p):%d\n", _data,m_flags,src._data,src.m_flags);
    if( &src == this )
        return *this;
    _bytes = src._bytes;
    _data = src._data;
    return *this;
}

__device__ inline string_view& string_view::operator=(string_view&& src)
{
    //printf("mv=(%p):%d/(%p):%d\n", _data,m_flags,src._data,src.m_flags);
    if( &src == this )
        return *this;
    _bytes = src._bytes;
    _data = src._data;
    src._data = nullptr;
    src._bytes = 0;
    return *this;
}

//
__device__ inline unsigned int string_view::size() const
{
    return _bytes;
}

__device__ inline unsigned int string_view::length() const
{
    return _bytes;
}

__device__ inline unsigned int string_view::chars_count() const
{
    return chars_in_string(_data,_bytes);
}

__device__ inline const char* string_view::data() const
{
    return _data;
}

__device__ inline bool string_view::empty() const
{
    return _bytes == 0;
}

__device__ inline bool string_view::is_null() const
{
    return _data == nullptr;
}

// the custom iterator knows about UTF8 encoding
__device__ inline string_view::iterator::iterator(const string_view& str, unsigned int initPos)
    : p(0), cpos(0), offset(0)
{
    p = str.data();
    cpos = initPos;
    offset = str.byte_offset_for(cpos);
}

__device__ inline string_view::iterator::iterator(const string_view::iterator& mit)
    : p(mit.p), cpos(mit.cpos), offset(mit.offset)
{}

__device__ inline string_view::iterator& string_view::iterator::operator++()
{
    offset += bytes_in_char_byte((BYTE)p[offset]);
    ++cpos;
    return *this;
}

// what is the int parm for?
__device__ inline string_view::iterator string_view::iterator::operator++(int)
{
    iterator tmp(*this);
    operator++();
    return tmp;
}

__device__ inline bool string_view::iterator::operator==(const string_view::iterator& rhs) const
{
    return (p == rhs.p) && (cpos == rhs.cpos);
}

__device__ inline bool string_view::iterator::operator!=(const string_view::iterator& rhs) const
{
    return (p != rhs.p) || (cpos != rhs.cpos);
}

// unsigned int can hold 1-4 bytes for the UTF8 char
__device__ inline Char string_view::iterator::operator*() const
{
    Char chr = 0;
    char_to_Char(p + offset, chr);
    return chr;
}

__device__ inline unsigned int string_view::iterator::position() const
{
    return cpos;
}

__device__ inline unsigned int string_view::iterator::byte_offset() const
{
    return offset;
}

__device__ inline string_view::iterator string_view::begin() const
{
    return iterator(*this, 0);
}

__device__ inline string_view::iterator string_view::end() const
{
    return iterator(*this, chars_count());
}

__device__ inline Char string_view::at(unsigned int pos) const
{
    unsigned int offset = byte_offset_for(pos);
    if(offset >= _bytes)
        return 0;
    Char chr = 0;
    char_to_Char(data() + offset, chr);
    return chr;
}

__device__ inline Char string_view::operator[](unsigned int pos) const
{
    return at(pos);
}

__device__ inline unsigned int string_view::byte_offset_for(unsigned int pos) const
{
    unsigned int offset = 0;
    const char* sptr = _data;
    const char* eptr = sptr + _bytes;
    while( (pos > 0) && (sptr < eptr) )
    {
        unsigned int charbytes = bytes_in_char_byte((BYTE)*sptr++);
        if( charbytes )
            --pos;
        offset += charbytes;
    }
    return offset;
}

// 0	They compare equal
// <0	Either the value of the first character of this string that does not match is lower in the arg string,
//      or all compared characters match but the arg string is shorter.
// >0	Either the value of the first character of this string that does not match is greater in the arg string,
//      or all compared characters match but the arg string is longer.
__device__ inline int string_view::compare(const string_view& in) const
{
    return compare(in.data(), in.size());
}

__device__ inline int string_view::compare(const char* data, unsigned int bytes) const
{
    const unsigned char* ptr1 = reinterpret_cast<const unsigned char*>(this->data());
    if(!ptr1)
        return -1;
    const unsigned char* ptr2 = reinterpret_cast<const unsigned char*>(data);
    if(!ptr2)
        return 1;
    unsigned int len1 = size();
    unsigned int len2 = bytes;
    unsigned int idx;
    for(idx = 0; (idx < len1) && (idx < len2); ++idx)
    {
        if(*ptr1 != *ptr2)
            return (int)*ptr1 - (int)*ptr2;
        ptr1++;
        ptr2++;
    }
    if(idx < len1)
        return 1;
    if(idx < len2)
        return -1;
    return 0;
}

__device__ inline bool string_view::operator==(const string_view& rhs) const
{
    return compare(rhs) == 0;
}

__device__ inline bool string_view::operator!=(const string_view& rhs) const
{
    return compare(rhs) != 0;
}

__device__ inline bool string_view::operator<(const string_view& rhs) const
{
    return compare(rhs) < 0;
}

__device__ inline bool string_view::operator>(const string_view& rhs) const
{
    return compare(rhs) > 0;
}

__device__ inline bool string_view::operator<=(const string_view& rhs) const
{
    int rc = compare(rhs);
    return (rc == 0) || (rc < 0);
}

__device__ inline bool string_view::operator>=(const string_view& rhs) const
{
    int rc = compare(rhs);
    return (rc == 0) || (rc > 0);
}

__device__ inline int string_view::find(const string_view& str, unsigned int pos, int count) const
{
    return find(str.data(), str.size(), pos, count);
}

__device__ inline int string_view::find(const char* str, unsigned int bytes, unsigned int pos, int count) const
{
    char* sptr = (char*)data();
    if(!str || !bytes)
        return -1;
    unsigned int nchars = chars_count();
    if(count < 0)
        count = nchars;
    int end = (int)pos + count;
    if(end < 0 || end > nchars)
       end = nchars;
    int spos = (int)byte_offset_for(pos);
    int epos = (int)byte_offset_for((unsigned int)end);

    int len2 = (int)bytes;
    int len1 = (epos - spos) - (int)len2 + 1;

    char* ptr1 = sptr + spos;
    char* ptr2 = (char*)str;
    for(int idx=0; idx < len1; ++idx)
    {
        bool match = true;
        for( int jdx=0; match && (jdx < len2); ++jdx )
            match = (ptr1[jdx] == ptr2[jdx]);
        if( match )
            return (int)char_offset(idx+spos);
        ptr1++;
    }
    return -1;
}

// maybe get rid of this one
__device__ inline int string_view::find(Char chr, unsigned int pos, int count) const
{
    unsigned int sz = size();
    unsigned int nchars = chars_count();
    if(count < 0)
        count = nchars;
    int end = (int)pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    if(pos > end || chr == 0 || sz == 0)
        return -1;
    int spos = (int)byte_offset_for(pos);
    int epos = (int)byte_offset_for((unsigned int)end);
    //
    int chsz = (int)bytes_in_char(chr);
    char* sptr = (char*)data();
    char* ptr = sptr + spos;
    int len = (epos - spos) - chsz;
    for(int idx = 0; idx <= len; ++idx)
    {
        Char ch = 0;
        char_to_Char(ptr++, ch);
        if(chr == ch)
            return (int)chars_in_string(sptr, idx + spos);
    }
    return -1;
}

__device__ inline int string_view::rfind(const string_view& str, unsigned int pos, int count) const
{
    return rfind(str.data(), str.size(), pos, count);
}

__device__ inline int string_view::rfind(const char* str, unsigned int bytes, unsigned int pos, int count) const
{
    char* sptr = (char*)data();
    if(!str || !bytes)
        return -1;
    unsigned int sz = size();
    unsigned int nchars = chars_count();
    int end = (int)pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    int spos = (int)byte_offset_for(pos);
    int epos = (int)byte_offset_for(end);

    int len2 = (int)bytes;
    int len1 = (epos - spos) - len2 + 1;

    char* ptr1 = sptr + epos - len2;
    char* ptr2 = (char*)str;
    for(int idx=0; idx < len1; ++idx)
    {
        bool match = true;
        for(int jdx=0; match && (jdx < len2); ++jdx)
            match = (ptr1[jdx] == ptr2[jdx]);
        if(match)
            return (int)char_offset(epos - len2 - idx);
        ptr1--; // go backwards
    }
    return -1;
}

__device__ inline int string_view::rfind(Char chr, unsigned int pos, int count) const
{
    unsigned int sz = size();
    unsigned int nchars = chars_count();
    if(count < 0)
        count = nchars;
    int end = (int)pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    if(pos > end || chr == 0 || sz == 0)
        return -1;
    int spos = (int)byte_offset_for(pos);
    int epos = (int)byte_offset_for(end);

    int chsz = (int)bytes_in_char(chr);
    char* sptr = (char*)data();
    char* ptr = sptr + epos - 1;
    int len = (epos - spos) - chsz;
    for(int idx = 0; idx < len; ++idx)
    {
        Char ch = 0;
        char_to_Char(ptr--, ch);
        if(chr == ch)
            return (int)chars_in_string(sptr, epos - idx - 1);
    }
    return -1;
}


// parameters are character position values
__device__ inline string_view string_view::substr(unsigned int pos, unsigned int length) const
{
    unsigned int spos = byte_offset_for(pos);
    unsigned int epos = byte_offset_for(pos + length);
    if( epos > size() )
        epos = size();
    if(spos >= epos)
        return string_view("",0);
    length = epos - spos; // converts length to bytes
    return string_view(data()+spos,length);
}

__device__ inline unsigned int string_view::split(const char* delim, int count, string_view* strs) const
{
    const char* sptr = data();
    unsigned int sz = size();
    if(sz == 0)
    {
        if(strs && count)
            strs[0] = *this;
        return 1;
    }

    unsigned int bytes = string_length(delim);
    unsigned int delimCount = 0;
    int pos = find(delim, bytes);
    while(pos >= 0)
    {
        ++delimCount;
        pos = find(delim, bytes, (unsigned int)pos + bytes);
    }

    unsigned int strsCount = delimCount + 1;
    unsigned int rtn = strsCount;
    if((count > 0) && (rtn > count))
        rtn = count;
    if(!strs)
        return rtn;
    //
    if(strsCount < count)
        count = strsCount;
    //
    unsigned int dchars = (bytes ? chars_in_string(delim,bytes) : 1);
    unsigned int nchars = chars_count();
    unsigned int spos = 0, sidx = 0;
    int epos = find(delim, bytes);
    while(epos >= 0)
    {
        if(sidx >= (count - 1)) // add this to the while clause
            break;
        int len = (unsigned int)epos - spos;
        strs[sidx++] = substr(spos, len);
        spos = epos + dchars;
        epos = find(delim, bytes, spos);
    }
    if((spos <= nchars) && (sidx < count))
        strs[sidx] = substr(spos, nchars - spos);
    //
    return rtn;
}


__device__ inline unsigned int string_view::rsplit(const char* delim, int count, string_view* strs) const
{
    const char* sptr = data();
    unsigned int sz = size();
    if(sz == 0)
    {
        if(strs && count)
            strs[0] = *this;
        return 1;
    }

    unsigned int bytes = string_length(delim);
    unsigned int delimCount = 0;
    int pos = find(delim, bytes);
    while(pos >= 0)
    {
        ++delimCount;
        pos = find(delim, bytes, (unsigned int)pos + bytes);
    }

    unsigned int strsCount = delimCount + 1;
    unsigned int rtn = strsCount;
    if((count > 0) && (rtn > count))
        rtn = count;
    if(!strs)
        return rtn;
    //
    if(strsCount < count)
        count = strsCount;
    //
    unsigned int dchars = (bytes ? chars_in_string(delim,bytes) : 1);
    int epos = (int)chars_count(); // end pos is not inclusive
    int sidx = count - 1;          // index for strs array
    int spos = rfind(delim, bytes);
    while(spos >= 0)
    {
        if(sidx <= 0)
            break;
        //int spos = pos + (int)bytes;
        int len = epos - spos - dchars;
        strs[sidx--] = substr((unsigned int)spos+dchars, (unsigned int)len);
        epos = spos;
        spos = rfind(delim, bytes, 0, (unsigned int)epos);
    }
    if(epos >= 0)
        strs[0] = substr(0, epos);
    //
    return rtn;
}


__host__ __device__ inline unsigned int string_view::bytes_in_char(Char chr)
{
    unsigned int count = 1;
    // no if-statements means no divergence
    count += (int)((chr & (unsigned)0x0000FF00) > 0);
    count += (int)((chr & (unsigned)0x00FF0000) > 0);
    count += (int)((chr & (unsigned)0xFF000000) > 0);
    return count;
}

__host__ __device__ inline unsigned int string_view::char_to_Char(const char* pSrc, Char &chr)
{
    unsigned int chwidth = bytes_in_char_byte((BYTE)*pSrc);
    chr = (Char)(*pSrc++) & 0xFF;
    if(chwidth > 1)
    {
        chr = chr << 8;
        chr |= ((Char)(*pSrc++) & 0xFF); // << 8;
        if(chwidth > 2)
        {
            chr = chr << 8;
            chr |= ((Char)(*pSrc++) & 0xFF); // << 16;
            if(chwidth > 3)
            {
                chr = chr << 8;
                chr |= ((Char)(*pSrc++) & 0xFF); // << 24;
            }
        }
    }
    return chwidth;
}

__host__ __device__ inline unsigned int string_view::Char_to_char(Char chr, char* dst)
{
    unsigned int chwidth = bytes_in_char(chr);
    for(unsigned int idx = 0; idx < chwidth; ++idx)
    {
        dst[chwidth - idx - 1] = (char)chr & 0xFF;
        chr = chr >> 8;
    }
    return chwidth;
}

// counts the number of character in the first bytes of the given char array
__host__ __device__ inline unsigned int string_view::chars_in_string(const char* str, unsigned int bytes)
{
    if( (str==0) || (bytes==0) )
        return 0;
    //
    unsigned int nchars = 0;
    for(unsigned int idx = 0; idx < bytes; ++idx)
        nchars += (unsigned int)(((BYTE)str[idx] & 0xC0) != 0x80);
    return nchars;
}

__device__ inline unsigned int string_view::char_offset(unsigned int bytepos) const
{
    return chars_in_string(data(), bytepos);
}

}