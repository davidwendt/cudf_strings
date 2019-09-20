

auto create_strings_column( std::vector<const char*> hstrs )
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

   return cudf::make_strings_column( strings );
}

