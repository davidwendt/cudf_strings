
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

std::vector<const char*> hstrs1{ "xyz", "", "aé", nullptr, "bbb", "éé" };

#include "./test_utils.inl"

void print_int_column( cudf::column_view view )
{
   std::vector<unsigned int> cps(view.size());
   cudaMemcpy( cps.data(), view.data<uint32_t>(), view.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost );
   for( auto itr=cps.begin(); itr!=cps.end(); ++itr )
        printf(" %d", *itr);
    printf("\n");
}

int main(int argc, const char** argv)
{
   //
   auto column1 = create_strings_column(hstrs1);
   auto strings1 = cudf::strings_column_view(column1->view());
   printf("column1\n");
   cudf::strings::print(strings1);
   printf("\n");

   //
   auto results = cudf::strings::bytes_counts(strings1);
   printf("bytes-counts:\n");
   print_int_column(results->view());
   results = cudf::strings::characters_counts(strings1);
   printf("characters-counts:\n");
   print_int_column(results->view());

   results = cudf::strings::code_points(strings1);
   printf("code-points:\n");
   print_int_column(results->view());

   return 0;
}
