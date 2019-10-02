
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

#include "./test_utils.inl"

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
   {
      auto results = cudf::strings::gather(strings1,gather_map_view);
      auto strings = cudf::strings_column_view(results->view());
      printf("results (gather(column1,[1,3]))\n");
      cudf::strings::print(strings);
   }
   {
      auto results = cudf::strings::scatter(strings1,strings2,gather_map_view);
      auto strings = cudf::strings_column_view(results->view());
      printf("results (scatter(column1,column2,[1,3])\n");
      cudf::strings::print(strings);
   }
   {
      auto results = cudf::strings::scatter(strings1,"---",gather_map_view);
      auto strings = cudf::strings_column_view(results->view());
      printf("results (scatter(column1,'---',[1,3])\n");
      cudf::strings::print(strings);
   }

#if 0
   thrust::device_vector<int64_t> map64(2,0);
   map64[0] = 4;
   map64[1] = 2;
   cudf::column_view map64_view( cudf::data_type{cudf::INT64}, map64.size(),
                                 map64.data().get(), nullptr, 0);
   {
      auto results = cudf::strings::gather(strings1,map64_view);
      auto strings = cudf::strings_column_view(results->view());
      printf("results (gather(column1,[4,2]))\n");
      cudf::strings::print(strings);
   }

   thrust::device_vector<int8_t> map8(2,0);
   map8[0] = 5;
   map8[1] = 0;
   cudf::column_view map8_view( cudf::data_type{cudf::INT8}, map8.size(),
                                map8.data().get(), nullptr, 0);
   {
      auto results = cudf::strings::gather(strings1,map8_view);
      auto strings = cudf::strings_column_view(results->view());
      printf("results (gather(column1,[5,0]))\n");
      cudf::strings::print(strings);
   }
#endif
   return 0;
}
