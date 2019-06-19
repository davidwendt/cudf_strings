
#include "category.inl"

namespace custr
{
// pre-define these types
template<> const char* category<int>::get_type_name() { return "int32"; };
template class category<int>;
template<> const char* category<long>::get_type_name() { return "int64"; };
template class category<long>;
}
