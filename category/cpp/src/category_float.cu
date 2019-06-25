
#include "category.inl"

namespace custr
{
template<> const char* category<float>::get_type_name() { return "float32"; };
template class category<float>;
template<> const char* category<double>::get_type_name() { return "float64"; };
template class category<double>;
}

