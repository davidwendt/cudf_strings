#include <sys/time.h>
#include <unistd.h>

double GetTime()
{
	timeval tv;
	gettimeofday( &tv, NULL );
	return (double)(tv.tv_sec*1000000+tv.tv_usec)/1000000.0;
}

auto create_strings_column( std::vector<const char*> hstrs )
{
   // first, copy strings to device memory
   thrust::host_vector<thrust::pair<const char*,size_t> > strings(hstrs.size());
   size_t memsize = 0;
   for( auto itr=hstrs.begin(); itr!=hstrs.end(); ++itr )
   {
      if( *itr )
      {
         std::string str = *itr;
         memsize += str.size();
      }
   }
   double st_hm = GetTime();
   thrust::host_vector<char> buffer(memsize);
   thrust::device_vector<char> d_buffer(memsize);
   size_t offset = 0;
   for( size_t idx=0; idx < hstrs.size(); ++idx )
   {
      const char* str = hstrs[idx];
      if( !str )
      {
         strings[idx] = thrust::pair<const char*,size_t>{nullptr,0};
         continue;
      }
      //std::string str = hstrs[idx];
      size_t length = strlen(str);
      char* ptr = buffer.data() + offset;
      memcpy( ptr, str, length );
      char* d_ptr = d_buffer.data().get() + offset;
      strings[idx] = thrust::pair<const char*,size_t>{d_ptr,length};
      offset += length;
   }
   double et_hm = GetTime();
   double st_dm = et_hm;
   rmm::device_vector<thrust::pair<const char*,size_t> > d_strings(strings);
   //cudaMemcpy( d_strings.data().get(), strings.data(), hstrs.size()*sizeof(thrust::pair<const char*,size_t>), cudaMemcpyHostToDevice);
   cudaMemcpy( d_buffer.data().get(), buffer.data(), memsize, cudaMemcpyHostToDevice );
   double et_dm = GetTime();
   printf("hm(%g), dm(%g)\n", (et_hm-st_hm), (et_dm-st_dm));
   return cudf::make_strings_column( d_strings );
}

void print_int_column( cudf::column_view view )
{
   std::vector<unsigned int> cps(view.size());
   cudaMemcpy( cps.data(), view.data<uint32_t>(), view.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost );
   for( auto itr=cps.begin(); itr!=cps.end(); ++itr )
        printf(" %d", *itr);
    printf("\n");
}