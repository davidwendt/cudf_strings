
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include "../include/string_view.h"


// nvcc -w -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_70,code=sm_70 test_string_view.cu -I/usr/local/cuda/include -o test_string_view

int main(int argc, const char** argv)
{

   thrust::for_each_n( thrust::device, thrust::make_counting_iterator<int>(0), 1,
         [] __device__ (int idx) {

            cudf::string_view dstr1("abcdéfghijklmnopqrstuvwxyz",13); // é
            printf("%d:%s,bytes=%d,chars=%d\n",idx,dstr1.data(),dstr1.size(),dstr1.chars_count());
            cudf::string_view dstr2 = dstr1;
            printf("%d:%s,length=%d,chars=%d\n",idx,dstr2.data(),dstr2.length(),dstr2.chars_count());
            
            printf("%d:compare=%d\n", idx, dstr1.compare("abcdéfghij") ); // 10 chars, 11 bytes
            printf("%d:find(f)=%d\n", idx, dstr2.find("f",1) );

            cudf::string_view dstr = dstr1.substr(2,10);
            printf("%d:substr(2,10):%s,bytes=%d,chars=%d\n",idx,dstr.data(),dstr.size(),dstr.chars_count());

            cudf::string_view dstr3("cdéfghijkl");
            printf("%d:%s < %s = %d\n", idx, dstr.data(), dstr3.data(), (int)(dstr < dstr3) );
      });


   cudaError_t err = cudaDeviceSynchronize();
   printf("error = %d\n", (int)err);
   return 0;
}
