
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include <cudf/strings/strings_column_handler.hpp>
#include <cudf/strings/strings_column_factories.hpp>

// nvcc -w -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_70,code=sm_70 test_sort.cu -I/usr/local/cuda/include -I../../../cudf/cpp/include -I../../../rmm/include -L../../../cudf/cpp/build -L../../../rmm/build -lcudf -lrmm  -o test_sort

std::vector<const char*> hstrs1{ "aaa", "ccc", "eee", "ddd", "ccc" };
std::vector<const char*> hstrs2{ "eee", "bbb", "ccc", "aaa", "bbb" };

auto create_strings_column( std::vector<const char*> hstrs )
{
   // first, copy strings to device memory
   thrust::device_vector<thrust::pair<const char*,size_t> > strings;
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

   return cudf::make_strings_column( reinterpret_cast<std::pair<const char*,size_t>*>(strings.data().get()), strings.size() );
}

int main(int argc, const char** argv)
{
   
   //
   auto column1 = create_strings_column(hstrs1);
   auto handler1 = cudf::strings_column_handler(column1->view());
   handler1.print();
   auto column2 = create_strings_column(hstrs2);
   auto handler2 = cudf::strings_column_handler(column2->view());
   handler2.print();

   auto column3 = handler2.sort(cudf::strings_column_handler::name);
   auto handler3 = cudf::strings_column_handler(column3->view());
   handler3.print();

   return 0;
}
