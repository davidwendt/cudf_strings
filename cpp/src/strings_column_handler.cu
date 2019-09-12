/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/strings/strings_column_handler.hpp>
#include <cudf/strings/strings_column_factories.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <utilities/error_utils.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

namespace cudf {

#define STR_OFFSETS_CHILD_INDEX 0
#define STR_CHARS_CHILD_INDEX 1

//
strings_column_handler::strings_column_handler( const column_view& strings_column )
    : _parent(strings_column)
{
    CUDF_EXPECTS( _parent.type().id()==STRING, "string_column_view only support strings");
    CUDF_EXPECTS( _parent.num_children()>0, "string column must have children");
}

size_type strings_column_handler::count() const
{
    return _parent.child(STR_OFFSETS_CHILD_INDEX).size();
}

const char* strings_column_handler::chars_data() const
{
    return _parent.child(STR_CHARS_CHILD_INDEX).data<char>();
}

const int32_t* strings_column_handler::offsets_data() const
{
    return _parent.child(STR_OFFSETS_CHILD_INDEX).data<int32_t>();
}

size_type strings_column_handler::chars_column_size() const
{
    return _parent.child(STR_OFFSETS_CHILD_INDEX).size();
}

const bitmask_type* strings_column_handler::null_mask() const
{
    return _parent.null_mask();
}

size_type strings_column_handler::null_count() const
{
    return _parent.null_count();
}

// print strings to stdout
void strings_column_handler::print( size_type start, size_type end,
                                    size_type max_width, const char* delimiter ) const
{
    size_type count = this->count();
    if( end < 0 || end > count )
        end = count;
    if( start < 0 )
        start = 0;
    if( start >= end )
        return;
    count = end - start;

    // stick with the default stream for this odd/rare stdout function
    auto execpol = rmm::exec_policy(0);
    auto strings_column = column_device_view::create(_parent);
    auto d_column = *strings_column;
    auto d_offsets = offsets_data();
    auto d_strings = chars_data();

    // get individual strings sizes
    rmm::device_vector<size_t> output_offsets(count,0);
    thrust::transform( execpol->on(0),
        thrust::make_counting_iterator<size_type>(start), thrust::make_counting_iterator<size_type>(end),
        output_offsets.begin(),
        [d_column, d_strings, max_width, d_offsets] __device__ (size_type idx) {
            if( d_column.nullable() && d_column.is_null(idx) )
                return 0;
            size_type offset = idx ? d_offsets[idx-1] : 0; // this logic will be a template
            size_type bytes = d_offsets[idx] - offset;     // specialization on element()
            string_view dstr( d_strings + offset, bytes ); // method of column_device_view
            if( (max_width > 0) && (dstr.characters() > max_width) )
                bytes = dstr.byte_offset_for(max_width);
            return bytes+1; // allow for null-terminator on non-null strings
        });
    // convert to offsets
    thrust::inclusive_scan( execpol->on(0), output_offsets.begin(), output_offsets.end(), output_offsets.begin() );
    // build output buffer
    size_t buffer_size = output_offsets[count-1];
    if( buffer_size == 0 )
    {
        printf("all %d strings are null\n", count);
        return;
    }
    rmm::device_vector<char> buffer(buffer_size,0); // allocate and pre-null-terminate
    char* d_buffer = buffer.data().get();
    // copy strings into output buffer
    size_t* d_output_offsets = output_offsets.data().get();
    thrust::for_each_n(execpol->on(0),
        thrust::make_counting_iterator<size_type>(0), (end-start),
        [d_strings, start, d_offsets, d_output_offsets, d_buffer] __device__(size_type idx) {
            size_t output_offset = (idx ? d_output_offsets[idx-1] : 0);
            size_t length = d_output_offsets[idx] - output_offset; // bytes
            if( length ) // this is only 0 for nulls
            {
                idx += start;
                size_type offset = (idx ? d_offsets[idx-1]:0);
                memcpy(d_buffer + output_offset, d_strings + offset, length-1 );
            }
        });

    // copy output buffer to host
    std::vector<size_t> h_offsets(count);
    cudaMemcpyAsync( h_offsets.data(), d_output_offsets, count*sizeof(size_t), cudaMemcpyDeviceToHost);
    std::vector<char> h_buffer(buffer_size);
    cudaMemcpyAsync( h_buffer.data(), d_buffer, buffer_size, cudaMemcpyDeviceToHost );
    cudaStreamSynchronize(0);

    // print out the strings to stdout
    for( size_type idx=0; idx < count; ++idx )
    {
        size_t offset = (idx ? h_offsets[idx-1]:0);
        size_t length = h_offsets[idx] - offset;
        printf("%d:",idx);
        if( length )
            printf("[%s]", h_buffer.data()+offset);
        else
            printf("<null>");
        printf("%s",delimiter);
    }
}

// new strings column from subset of given strings column
std::unique_ptr<cudf::column> strings_column_handler::sublist( size_type start, size_type end, size_type step )
{
    return make_strings_column(nullptr, 0);
}

std::unique_ptr<cudf::column> strings_column_handler::gather( const column_view& indices )
{
    //
    size_type count = indices.size();
    auto d_indices = indices.data<int32_t>();

    auto execpol = rmm::exec_policy(0);
    auto strings_column = column_device_view::create(_parent);
    auto d_column = *strings_column;
    auto d_offsets = offsets_data();

    // build offsets column
    auto offsets_column = make_numeric_column( data_type{INT32}, count, mask_state::UNALLOCATED );
    auto offsets_view = offsets_column->mutable_view();
    auto d_new_offsets = offsets_view.data<int32_t>();
    // get lengths
    thrust::transform( execpol->on(0),
        thrust::make_counting_iterator<size_type>(0), thrust::make_counting_iterator<size_type>(count),
        d_new_offsets,
        [d_column, d_offsets, d_indices] __device__ (size_type idx) {
            size_type index = d_indices[idx];
            if( d_column.nullable() && d_column.is_null(index) )
                return 0;
            size_type offset = idx ? d_offsets[index-1] : 0;
            return d_offsets[index] - offset;
        });
    // convert to offsets
    thrust::inclusive_scan( execpol->on(0), d_new_offsets, d_new_offsets+count, d_new_offsets );
    // build null mask
    size_type null_count = this->null_count();
    mask_state state = mask_state::UNINITIALIZED;
    if( null_count==0 )
      state = mask_state::UNALLOCATED;
    else if( null_count==count )
      state = mask_state::ALL_NULL;
    auto null_mask = create_null_mask(count, state);
    if( (null_count > 0) && (null_count < count) )
    {
      uint8_t* d_null_mask = static_cast<uint8_t*>(null_mask.data());
      CUDA_TRY(cudaMemsetAsync(d_null_mask, 0, null_mask.size()));
      thrust::transform(execpol->on(0),
        thrust::make_counting_iterator<size_type>(0), thrust::make_counting_iterator<size_type>(count/8),
        d_null_mask,
        [d_column, count] __device__(size_type byte_idx) {
            unsigned char byte = 0; // set one byte per thread -- init to all nulls
            for( size_type i=0; i < 8; ++i )
            {
              size_type idx = i + (byte_idx*8);  // compute d_strs index
              byte = byte >> 1;                  // shift until we are done
              if( idx < count )                  // check boundary
              {
                if( d_column.is_null(idx) )
                  byte |= 128;               // string is not null, set high bit
              }
            }
            return byte; //d_null_mask[byte_idx] = byte;
        });
    }

    // build chars column
    auto chars_column = make_numeric_column( data_type{INT8}, chars_column_size(), mask_state::UNALLOCATED );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<int8_t>(); 
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_type>(0), count,
        [d_column, d_indices, d_new_offsets, d_chars] __device__(size_type idx){
            // place individual strings
            if( d_column.nullable() && d_column.is_null(idx) )
                return;
            string_view dstr = d_column.element<string_view>(d_indices[idx]);
            size_type offset = (idx ? d_new_offsets[idx-1] : 0);
            memcpy(d_chars + offset, dstr.data(), dstr.size() );
        });

  // build children vector
  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(offsets_column));
  children.emplace_back(std::move(chars_column));

  return std::make_unique<column>(
      data_type{STRING}, 0, rmm::device_buffer{0},
      null_mask, null_count,
      std::move(children));   
}

// return sorted version of the given strings column
std::unique_ptr<cudf::column> strings_column_handler::sort( sort_type stype, bool ascending, bool nullfirst )
{
    //
    auto execpol = rmm::exec_policy(0);
    auto strings_column = column_device_view::create(_parent);
    auto d_column = *strings_column;

    // lets sort indices
    size_type count = this->count();
    thrust::device_vector<size_type> indices(count);
    thrust::sequence( execpol->on(0), indices.begin(), indices.end() );
    thrust::sort( execpol->on(0), indices.begin(), indices.end(),
        [d_column, stype, ascending, nullfirst] __device__ (size_type lhs, size_type rhs) {
            bool lhs_null{d_column.nullable() && d_column.is_null(lhs)};
            bool rhs_null{d_column.nullable() && d_column.is_null(rhs)};
            if( lhs_null || rhs_null )
                return (nullfirst ? !rhs_null : !lhs_null);
            string_view lhs_str = d_column.element<string_view>(lhs);
            string_view rhs_str = d_column.element<string_view>(rhs);
            int cmp = lhs_str.compare(rhs_str);
            return (ascending ? (cmp<0) : (cmp>0));
        });

    // should have a way to create a column_view with an existing memory buffer
    auto d_indices = indices.data().get();
    // we will create an empty one and pass in this data for now
    auto indices_column = make_numeric_column( data_type{INT32}, count, mask_state::UNALLOCATED );
    auto indices_view = indices_column->mutable_view();
    cudaMemcpyAsync( indices_view.data<int32_t>(), d_indices, count*sizeof(int32_t), cudaMemcpyDeviceToDevice);

    // now build a new strings column from the indices
    return gather( indices_view );
}

// return sorted indexes only -- returns integer column
std::unique_ptr<cudf::column> strings_column_handler::order( sort_type stype, bool ascending, bool nullfirst )
{
    return make_strings_column(nullptr, 0);
}

}  // namespace cudf