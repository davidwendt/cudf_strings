
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

// nvcc -w -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_70,code=sm_70 timing_combine.cu -I/usr/local/cuda/include -I../../../cudf/cpp/include -I../../../rmm/include -L../../../cudf/cpp/build -L../../../rmm/build -lcudf -lrmm  -o timing_combine

std::vector<const char*> hstrs1{ "TUVWXYZ", "1234567890", "abcdéfgij", nullptr, "" };
std::vector<const char*> hstrs2{ nullptr, "ABC", "","9=4+5", "éa" };

#include "./test_utils.inl"


int main(int argc, const char** argv)
{

    // create large string arrays
    std::vector<const char*> strs1, strs2;
    for( int idx=0; idx < 1000000; ++idx )
    {
        strs1.push_back(hstrs1[idx % hstrs1.size()]);
        strs2.push_back(hstrs2[idx % hstrs2.size()]);
    }
    printf("build columns\n");
    //
    double st_c1 = GetTime();
    auto column1 = create_strings_column(strs1);
    auto strings1 = cudf::strings_column_view(column1->view());
    double et_c1 = GetTime();
    printf("column1\n");
    cudf::strings::print(strings1,0,10);
    printf("%g seconds\n",(et_c1-st_c1));
    double st_c2 = GetTime();
    auto column2 = create_strings_column(strs2);
    auto strings2 = cudf::strings_column_view(column2->view());
    double et_c2 = GetTime();
    printf("column2\n");
    cudf::strings::print(strings2,0,10);
    printf("%g seconds\n",(et_c2-st_c2));

    double st_cat = GetTime();
    auto results = cudf::strings::concatenate(strings1,strings2,":");
    double et_cat = GetTime();
    printf("concat(column1,column2,':'):\n");
    cudf::strings::print(results->view(),0,10);
    printf("%g seconds\n",(et_cat-st_cat));

    return 0;
}
