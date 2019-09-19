
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strings_column_factories.hpp>
#include <rmm/thrust_rmm_allocator.h>

// nvcc -w -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_70,code=sm_70 test_array.cu -I/usr/local/cuda/include -I../../../cudf/cpp/include -I../../../rmm/include -L../../../cudf/cpp/build -L../../../rmm/build -lcudf -lrmm  -o test_array

std::vector<const char*> hstrs1{ "eee", "bbb", "cccc", "aa", "bbb", "ééé" };
std::vector<const char*> hstrs2{ "1", "22" };


auto create_strings_column( std::vector<const char*> hstrs )
{
   // first, copy strings to device memory
   rmm::device_vector<thrust::pair<const char*,size_t> > strings;
   size_t memsize = 0;
   for( auto itr=hstrs.begin(); itr!=hstrs.end(); ++itr )
   {
      std::string str = *itr;
      memsize += str.size();
   }
   thrust::device_vector<char> buffer(memsize);
   size_t offset = 0;
   for( size_t idx=0; idx < hstrs.size(); ++idx )
   {
      std::string str = hstrs[idx];
      size_t length = str.size();
      char* ptr = buffer.data().get() + offset;
      cudaMemcpy( ptr, str.c_str(), length, cudaMemcpyHostToDevice );
      strings.push_back( thrust::pair<const char*,size_t>{ptr,length} );
      offset += length;
   }

   return cudf::make_strings_column( strings );
}

int main(int argc, const char** argv)
{
   
   //
   auto column1 = create_strings_column(hstrs1);
   auto strings1 = cudf::strings_column_view(column1->view());
   printf("column1\n");
   cudf::strings::print(strings1);
   printf("\n");
   auto column2 = create_strings_column(hstrs2);
   auto strings2 = cudf::strings_column_view(column2->view());
   printf("column2\n");
   cudf::strings::print(strings2);
   printf("\n");

   thrust::device_vector<int32_t> gather_map(2,0);
   gather_map[0] = 1;
   gather_map[1] = 3;
   cudf::column_view gather_map_view( cudf::data_type{cudf::INT32}, gather_map.size(),
                                      gather_map.data().get(), nullptr, 0);

   auto column3 = cudf::strings::gather(strings1,gather_map_view);
   auto strings3 = cudf::strings_column_view(column3->view());
   printf("column3 (gather(column1,[1,3]))\n");
   cudf::strings::print(strings3);

   auto column4 = cudf::strings::scatter(strings1,strings2,gather_map_view);
   auto strings4 = cudf::strings_column_view(column4->view());
   printf("column4 (scatter(column1,column2,[1,3])\n");
   cudf::strings::print(strings4);

   return 0;
}
