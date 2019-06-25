/**---------------------------------------------------------------------------*
 * @brief
 * Assume that `DType` is the same as `gdf_dtype`, and that `device_buffer` is
 * effectively a `rmm::device_vector`.
 * Assume `bitmask` is a class that wraps a `device_buffer` of bitmask elements.
 *
 *  Note: This design purposefully ignores the issue of `allocators` and
controlling how the device memory for the column is allocated. This will be
added as the allocator design in fleshed out. For the timebeing, assume that an
allocator handle can be passed into each of the constructors that controls how
the device memory for the data and bitmask is allocated.
 *
*---------------------------------------------------------------------------**/
#include <cstdint>
#include <cuda_runtime.h>
#include "bitmask.hpp"

namespace cudf {

using size_type = int32_t;

enum DType { INT8, INT16, INT32, INT64, FLOAT32, FLOAT64, INVALID };

struct column_view {
  column_view(void* data, DType type, size_type length,
              bitmask_view const& mask, size_type null_count)
      : _data{data},
        _type{type},
        _length{length},
        _mask{mask},
        _null_count{null_count} {}

  template <typename T>
  __host__ __device__ T* typed_data() noexcept {
    return static_cast<T*>(data);
  }

  template <typename T>
  __host__ __device__ T const* typed_data() const noexcept {
    return static_cast<T const*>(data);
  }

  __host__ __device__ void const* data() noexcept { return _data; }
  __host__ __device__ void* data() const noexcept { return _data; }

  __device__ bool is_valid(size_type i) const noexcept {
    return _mask.is_valid(i);
  }

  __device__ bool is_null(size_type i) const noexcept {
    return _mask.is_null(i);
  }

  __host__ __device__ bool nullable() const noexcept {
    return _mask.nullable();
  }

  __host__ __device__ size_type null_count() const noexcept {
    return _null_count;
  }

  __host__ __device__ size_type length() const noexcept { return _length; }

  __host__ __device__ DType type() const noexcept { return _type; }

  __host__ __device__ bitmask_view mask() noexcept { return _mask; }
  __host__ __device__ bitmask_view const mask() const noexcept { return _mask; }

  __host__ __device__ column_view* other() noexcept { return _other; }
  __host__ __device__ column_view const* other() const noexcept {
    return _other;
  }

 private:
  void* _data{nullptr};
  DType _type{INVALID};
  cudf::size_type _length{0};
  bitmask_view _mask;
  cudf::size_type _null_count{0};
  column_view* _other{nullptr};
};

struct column {
  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a size, type, and option to
   * allocate bitmask.
   *
   * Both the data and bitmask are unintialized.
   *
   * @param[in] type The element type
   * @param[in] size The number of elements in the column
   * @param[in] allocate_bitmask Optionally allocate an appropriate sized
   * bitmask
   *---------------------------------------------------------------------------**/
  column(DType type, int size, bool allocate_bitmask = false);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, and a device_buffer for
   * data that will be *deep* copied.
   *
   * @param[in] dtype The element type
   * @param[in] size The number of elements in the column
   * @param[in] data device_buffer whose data will be *deep* copied
   *---------------------------------------------------------------------------**/
  column(DType dtype, int size, device_buffer data);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, and a device_buffer for
   * data that will be shallow copied.
   *
   * @param[in] dtype The element type
   * @param[in] size The number of elements in the column
   * @param[in] data device_buffer whose data will be moved into this column
   *---------------------------------------------------------------------------**/
  column(DType dtype, int size, device_buffer&& data);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, and device_buffers for data and
   * bitmask that will be *shallow* copied.
   *
   * This constructor uses move semantics to take ownership of the
   *device_buffer's device memory. The `device_buffer` passed into this
   *constructor will not longer be valid to use. Furthermore, it will result in
   *undefined behavior if the device_buffer`s associated memory is modified or
   *freed after invoking this constructor.
   *
   * @param dtype The element type
   * @param[in] size The number of elements in the column
   * @param data device_buffer whose data will be moved from into this column
   * @param mask bitmask whose data will be moved into this column
   *---------------------------------------------------------------------------**/
  column(DType dtype, int size, device_buffer&& data, bitmask&& mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, size, and deep copied device
   * buffer for data, and moved bitmask.
   *
   * @param dtype The element type
   * @param size The number of elements
   * @param data device_buffer whose data will be *deep* copied
   * @param mask bitmask whose data will be moved into this column
   *---------------------------------------------------------------------------**/
  column(DType dtype, int size, device_buffer data, bitmask&& mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, size, and moved device
   * buffer for data, and deep copied bitmask.
   *
   * @param dtype The element type
   * @param size The number of elements
   * @param data device_buffer whose data will be moved into this column
   * @param mask bitmask whose data will be deep copied into this column
   *---------------------------------------------------------------------------**/
  column(DType dtype, int size, device_buffer&& data, bitmask mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column by deep copying the device memory of another
   * column.
   *
   * @param other The other column to copy
   *---------------------------------------------------------------------------**/
  column(column const& other);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column object by moving the device memory from
   *another column.
   *
   * @param other The other column whose device memory will be moved to the new
   * column
   *---------------------------------------------------------------------------**/
  column(column&& other);

  column_view view() noexcept {}

  column_view const view() const noexcept {}

 private:
  device_buffer _data;  ///< Dense, contiguous, type erased device memory buffer
                        ///< containing the column elements
  bitmask _mask;        ///< Validity bitmask for columne elements
  DType _type{INVALID};  ///< Logical type of elements in the column
  std::unique_ptr<column> _other{
      nullptr};  ///< Depending on column's type, may point to
                 ///< another column, e.g., a Dicitionary
};
}  // namespace cudf

