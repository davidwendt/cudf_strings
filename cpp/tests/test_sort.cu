
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strings_column_factories.hpp>
#include <rmm/thrust_rmm_allocator.h>

// nvcc -w -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_70,code=sm_70 test_sort.cu -I/usr/local/cuda/include -I../../../cudf/cpp/include -I../../../rmm/include -L../../../cudf/cpp/build -L../../../rmm/build -lcudf -lrmm  -o test_sort

std::vector<const char*> hstrs1{ "eee", "bbb", "", "cccc", nullptr, "aa", "bbb", "ééé" };

#include "./test_utils.inl"

int main(int argc, const char** argv)
{
   auto column1 = create_strings_column(hstrs1);
   auto strings1 = cudf::strings_column_view(column1->view());
   printf("column1\n");
   cudf::strings::print(strings1);

   auto column2 = cudf::strings::sort(strings1, cudf::strings::name);
   auto strings2 = cudf::strings_column_view(column2->view());
   printf("column2 (sorted column1)\n");
   cudf::strings::print(strings2);

   return 0;
}
