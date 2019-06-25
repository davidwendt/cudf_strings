
#include "category.inl"

size_t count_nulls( const BYTE* nulls, size_t count )
{
    if( !nulls || !count )
        return 0;
    size_t result = thrust::count_if( thrust::device, thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(count),
            [nulls] __device__ (size_t idx) { return ((nulls[idx/8] & (1 << (idx % 8)))==0); });
    return result;
}

base_category_type::~base_category_type()
{}

namespace custr
{
template<> const char* category<char>::get_type_name() { return "int8"; };
template class category<char>;
}
