
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strings_column_factories.hpp>
#include <rmm/thrust_rmm_allocator.h>

// nvcc -w -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_70,code=sm_70 test_string_column.cu -I/usr/local/cuda/include -I../../../cudf/cpp/include -I../../../rmm/include -L../../../cudf/cpp/build -L../../../rmm/build -lcudf -lrmm -o test_string_column

std::vector<const char*> hstrs{ "the quick brown fox jumps over the lazy dog",
                                "the fat cat lays next to the other accénted cat",
                                 "a slow moving turtlé cannot catch the bird",
                                 "which can be composéd together to form a more complete",
                                 "thé result does not include the value in the sum in",
                                 "", nullptr, "absent stop words" };

int main(int argc, const char** argv)
{
   // first, copy strings to device memory
   rmm::device_vector<thrust::pair<const char*,size_t> > strings;
   size_t memsize = 0;
   for( auto itr=hstrs.begin(); itr!=hstrs.end(); ++itr )
   {
      if( *itr )
      {
         std::string str = *itr;
         memsize += str.size();
      }
   }
   thrust::device_vector<char> buffer(memsize);
   size_t offset = 0;
   for( size_t idx=0; idx < hstrs.size(); ++idx )
   {
      if( !hstrs[idx] )
      {
         strings.push_back( thrust::pair<const char*,size_t>{nullptr,0} );
         continue;
      }
      std::string str = hstrs[idx];
      size_t length = str.size();
      char* ptr = buffer.data().get() + offset;
      cudaMemcpy( ptr, str.c_str(), length, cudaMemcpyHostToDevice );
      strings.push_back( thrust::pair<const char*,size_t>{ptr,length} );
      offset += length;
   }
   
   //
   auto column = cudf::make_strings_column( strings );
   cudf::strings::print(cudf::strings_column_view(column->view()));

   return 0;
}
