
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

// nvcc -w -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_70,code=sm_70 test_combine.cu -I/usr/local/cuda/include -I../../../cudf/cpp/include -I../../../rmm/include -L../../../cudf/cpp/build -L../../../rmm/build -lcudf -lrmm  -o test_combine

std::vector<const char*> hstrs1{ "xyz", "", "aé", nullptr, "bbb", "éé" };
std::vector<const char*> hstrs2{ "abc", "d","éa", "", nullptr, "f" };

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

   //
   auto results = cudf::strings::concatenate(strings1,strings2);
   printf("concat(column1,column2):\n");
   cudf::strings::print(results->view());

   results = cudf::strings::concatenate(strings1,strings2,":");
   printf("concat(column1,column2,':'):\n");
   cudf::strings::print(results->view());

   results = cudf::strings::concatenate(strings1,strings2,":","<null>");
   printf("concat(column1,column2,':','<null>'):\n");
   cudf::strings::print(results->view());

   return 0;
}
