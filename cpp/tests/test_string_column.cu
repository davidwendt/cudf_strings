
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

//#include <bitmask/valid_if.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strings_column_factories.hpp>
#include <rmm/thrust_rmm_allocator.h>

// nvcc -w -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_70,code=sm_70 test_string_column.cu -I/usr/local/cuda/include -I../../../cudf/cpp/include -I../../../rmm/include -L../../../cudf/cpp/build -L../../../rmm/build -lcudf -lrmm -o test_string_column

std::vector<const char*> hstrs{ "the quick brown fox jumps over the lazy dog",
                                "the fat cat lays next to the other accénted cat",
                                 "a slow moving turtlé cannot catch the bird",
                                 "which can be composéd together to form a more complete",
                                 "thé result does not include the value in the sum in",
                                 "", "absent stop words" };

std::unique_ptr<cudf::column> create_column()
{
   cudf::size_type count = hstrs.size();
   // build offsets vector
   thrust::host_vector<cudf::size_type> offsets(hstrs.size()+1);
   cudf::size_type offset = 0;
   for( int idx=0; idx < hstrs.size(); ++idx )
   {
      offsets[idx] = offset;
      const char* str = hstrs[idx];
      if( str )
         offset += strlen(str);
   }
   offsets[hstrs.size()] = offset;
   cudf::size_type memsize = offset;

   // build null mask
   //rmm::device_vector<const char*> dstrs(hstrs);
   //auto valid_mask = cudf::valid_if( static_cast<const bit_mask_t*>(nullptr),
   //     [dstrs] __device__ (cudf::size_type idx) { return dstrs[idx]!=nullptr; }, hstrs.size() );
   //auto null_count = valid_mask.second;
   rmm::device_vector<cudf::bitmask_type> null_mask;//gdf_valid_allocation_size(count),0);
   //cudaMemcpy( null_mask.data().get(), valid_mask.first, gdf_valid_allocation_size(count), cudaMemcpyDeviceToDevice );
   //RMM_FREE(valid_mask.first,0);

   // build chars vector
   thrust::host_vector<char> buffer(offset);
   for( int idx=0; idx < hstrs.size(); ++idx )
   {
      const char* str = hstrs[idx];
      if( !str )
         continue;
      offset = offsets[idx];
      cudf::size_type length = offsets[idx+1] - offset;
      char* ptr = buffer.data() + offset;
      memcpy( ptr, str, length );
   }

   // copy to device
   rmm::device_vector<char> d_chars(buffer);
   rmm::device_vector<cudf::size_type> d_offsets(offsets);
   rmm::device_vector<cudf::bitmask_type> d_nulls;
   //
   return cudf::make_strings_column( d_chars, d_offsets, null_mask, 0 );
}

int main(int argc, const char** argv)
{
   auto column = create_column();
   cudf::strings_column_view strings_view(column->view());
   cudf::strings::print(strings_view);

   auto arrow = cudf::strings::create_offsets(strings_view);
   //printf("chars=%d, offsets=%d\n", (int)arrow.first.size(), (int)arrow.second.size());
   thrust::host_vector<char> h_chars(arrow.first);
   thrust::host_vector<cudf::size_type> h_offsets(arrow.second);
   printf("chars=%d, offsets=%d\n", (int)h_chars.size(), (int)h_offsets.size());
   for( int idx=0; idx < (int)strings_view.size(); ++idx )
   {
      int offset = (int)h_offsets[idx];
      int length = (int)h_offsets[idx+1] - offset;
      std::string str(h_chars.data()+offset,length);
      printf("%d:[%s]\n",idx, str.c_str());
   }

   return 0;
}
