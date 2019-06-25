

#include <cstdint>

namespace cudf {

using size_type = int32_t;

enum DType { INT8, INT16, INT32, INT64, FLOAT32, FLOAT64, INVALID };

struct column_view
{
    void* _data{nullptr};
    DType _type{INVALID};
    cudf::size_type _length{0};
    uint8_t* _mask;
    cudf::size_type _null_count{0};
    column_view* _other{nullptr};
};

struct column
{
    void* _data{nullptr};
    DType _type{INVALID};
    cudf::size_type _length{0};
    uint8_t* _mask;
    cudf::size_type _null_count{0};
    column* _other{nullptr};
};

}