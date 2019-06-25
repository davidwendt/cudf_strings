

namespace cudf {

using size_type = int32_t;

using bitmask_t = uint32_t;
struct bitmask_view {
  __device__ bool is_valid(size_type i) const noexcept {}

  __device__ bool is_null(size_type i) const noexcept {
    return not is_valid(i);
  }

  __host__ __device__ bool nullable() const noexcept {
    return nullptr != _mask;
  }

  __host__ __device__ bitmask_t* data() noexcept { return _mask; }
  __host__ __device__ bitmask_t const* data() const noexcept { return _mask; }

 private:
  bitmask_t* _mask{nullptr};
  size_type _length{0};
};

struct bitmask {
 private:
  device_buffer data;
  size_type length;
}
}  // namespace cudf
