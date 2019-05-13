
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

namespace cudf
{

template<typename T>
inline void category_impl<T>::create_keys(const T* items, const int* indexes, size_t ucount, bool includes_null, thrust::host_vector<T>& keys)
{
    keys.resize(ucount);
    thrust::gather(thrust::host, indexes, indexes+ucount, items, keys.data());
}

template<typename T>
inline void category_impl<T>::create_keys(const T* items, size_t ucount, bool includes_null, thrust::host_vector<T>& keys)
{
    keys.resize(ucount);
    memcpy( keys.data(), items, ucount*sizeof(T) );
}

//template<>
inline void category_impl<dstring>::create_keys(const dstring* items, const int* indexes, size_t ucount, bool includes_null, thrust::host_vector<dstring>& keys)
{
    keys.resize(ucount);
    cudf::dstring* d_keys = keys.data();
    // build memory for the keys
    thrust::host_vector<size_t> sizes(ucount);
    size_t* d_sizes = sizes.data();
    thrust::for_each_n(thrust::host, thrust::make_counting_iterator<size_t>(0), ucount,
        [items, indexes, d_sizes] (size_t idx) {
            d_sizes[idx] = items[indexes[idx]].size();
        });
    size_t memory_size = thrust::reduce(sizes.begin(),sizes.end());
    char* d_buffer = (char*)malloc(memory_size);
    thrust::host_vector<size_t> offsets(ucount);
    thrust::exclusive_scan(sizes.begin(),sizes.end(),offsets.begin());
    size_t* d_offsets = offsets.data();
    thrust::for_each_n(thrust::host, thrust::make_counting_iterator<size_t>(0), ucount,
        [items, indexes, d_buffer, d_offsets, d_keys] (size_t idx) {
            char* buffer = d_buffer + d_offsets[idx];
            dstring dstr = items[indexes[idx]];
            memcpy(buffer,dstr.data(),dstr.size());
            d_keys[idx] = {buffer,dstr.size()};
        });
    //
    the_keys_memory = d_buffer;
}

//template<>
inline void category_impl<dstring>::create_keys(const dstring* items, size_t ucount, bool includes_null, thrust::host_vector<dstring>& keys)
{
    keys.resize(ucount);
    cudf::dstring* d_keys = keys.data();
    // build memory for the keys
    thrust::host_vector<size_t> sizes(ucount);
    size_t* d_sizes = sizes.data();
    thrust::for_each_n(thrust::host, thrust::make_counting_iterator<size_t>(0), ucount,
        [items, d_sizes] (size_t idx) {
            d_sizes[idx] = items[idx].size();
        });
    size_t memory_size = thrust::reduce(sizes.begin(),sizes.end());
    char* d_buffer = (char*)malloc(memory_size);
    thrust::host_vector<size_t> offsets(ucount);
    thrust::exclusive_scan(sizes.begin(),sizes.end(),offsets.begin());
    size_t* d_offsets = offsets.data();
    thrust::for_each_n(thrust::host, thrust::make_counting_iterator<size_t>(0), ucount,
        [items, d_buffer, d_offsets, d_keys] (size_t idx) {
            char* buffer = d_buffer + d_offsets[idx];
            dstring dstr = items[idx];
            memcpy(buffer,dstr.data(),dstr.size());
            d_keys[idx] = {buffer,dstr.size()};
        });
    //
    the_keys_memory = d_buffer;
}

template<typename T, class Impl>
inline category<T,Impl>::category( const T* items, size_t count ) // add null-bitmask
{
    _bitmask.resize(count);
    thrust::host_vector<int> indexes(count);
    thrust::sequence(indexes.begin(), indexes.end()); // 0,1,2,3,4,5,6,7,8
    thrust::sort(thrust::host, indexes.begin(), indexes.end(),
        [items] (int lhs, int rhs) { return items[lhs]<items[rhs]; } ); // handle nulls
    _values.resize(count);
    int* d_values = _values.data();
    int* d_indexes = indexes.data();
    thrust::for_each_n(thrust::host, thrust::make_counting_iterator<size_t>(0), count,
        [items, d_indexes, d_values] (size_t idx) {
            d_values[idx] = (int)(idx && items[d_indexes[idx-1]]!=items[d_indexes[idx]]); // handle nulls
        });
    int ucount = thrust::reduce(thrust::host, d_values, d_values+count) +1;
    thrust::host_vector<int> keys_indexes(ucount);
    thrust::unique_copy(thrust::host, indexes.begin(), indexes.end(), keys_indexes.begin(),
        [items] ( int lhs, int rhs ) {
            return items[lhs]==items[rhs]; // handle nulls
        });
    thrust::inclusive_scan(thrust::host, d_values, d_values+count, d_values);
    thrust::sort_by_key(thrust::host, indexes.begin(), indexes.end(), d_values);
    impl.create_keys(items,keys_indexes.data(),ucount,false,_keys);
}

template<typename T, class Impl>
inline category<T>* category<T,Impl>::copy()
{
    category<T>* result = new category<T>;
    result->impl.create_keys(_keys.data(),_keys.size(),false,result->_keys);
    result->_values.assign(_values.begin(),_values.end());
    result->_bitmask.assign(_bitmask.begin(),_bitmask.end());
    return result;
}

template<typename T, class Impl>
inline int category<T,Impl>::get_value_for(T key)
{
    int* d_values = _values.data();
    int index = -1, count = _keys.size();
    T* d_keys = keys.data();
    thrust::copy_if( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(count), &index,
        [d_keys, key] (int idx) { return key == d_keys[idx]; });
    return index;
}

template<typename T, class Impl>
inline int* category<T,Impl>::get_indexes_for(T key)
{
    int index = get_value_for(key);
    int count = thrust::count( _values.begin(), _values.end(), index);
    int* results = new int[count];
    int* d_values = _values.data();
    thrust::copy_if( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(count), results, 
        [index, d_values] (int idx) { return d_values[idx]==index; });
    return results;
}

template<typename T, class Impl>
inline void category<T,Impl>::to_type( T* results )
{
    thrust::gather(thrust::host, _values.begin(), _values.end(), _keys.begin(), results);
}

template<typename T, class Impl>
inline void category<T,Impl>::gather_type( const int* indexes, size_t count, T* results )
{
    // need to account for negative index values
    size_t sz = size();
    int check = thrust::count_if( indexes, indexes+count, [sz] (int val) { return (val<0) || (val>=sz);});
    if( check > 0 )
        throw std::out_of_range("gather_type invalid index value");
    thrust::gather(thrust::host, indexes, indexes+count, _keys.begin(), results);
}

template<typename T, class Impl>
inline category<T>* category<T,Impl>::add_keys( const T* items, size_t count )
{
    if( count==0 )
        return copy();
    category<T>* result = new category<T>;
    //result->_bitmask.insert(_bitmask.begin(),_bitmask.end() ); // this does not change
    // shortcut for category with no keys
    size_t kcount = keys_size();
    if( keys_size()==0 )
    {
        thrust::host_vector<int> indexes(keys_size());
        thrust::sequence( indexes.begin(), indexes.end() );
        thrust::sort(thrust::host, indexes.begin(), indexes.end(),
            [items] (int lhs, int rhs) { return items[lhs]<items[rhs]; } ); // handle nulls too
        int* d_indexes = indexes.data();
        int* d_end = thrust::unique(thrust::host, d_indexes, d_indexes+indexes.size(),
            [items] (int lhs, int rhs ) {
                return items[lhs]==items[rhs];
            });
        result->impl.create_keys(items,indexes.data(),(size_t)(d_end-d_indexes),false,result->_keys);
        if( size() )  // just copy the values if we have them
            result->_values.assign(_values.begin(),_values.end());
        return result;
    }
    // incorporating the keys and adjust the values
    size_t both_count = kcount + count; // this is not the unique count
    thrust::host_vector<T> both_keys(both_count);  // first combine both keysets
    T* d_both_keys = both_keys.data();
    memcpy( d_both_keys, keys(), kcount*sizeof(T) );         // these keys
    memcpy( d_both_keys + kcount, items, count*sizeof(T) );  // and those keys
    thrust::host_vector<int> xvals(both_count); 
    int* d_xvals = xvals.data(); // build vector like: 0,...,(kcount-1),-1,...,-count
    thrust::for_each_n(thrust::host, thrust::make_counting_iterator<size_t>(0), both_count,
        [d_xvals, kcount] (size_t idx) {
            if( idx < kcount )
                d_xvals[idx] = idx;             //  0 ... (kcount-1)
            else
                d_xvals[idx]= kcount - idx -1;  // -1 ... -count
        });
    // compute the new keyset by doing sort/unique
    // stable-sort preserves order for keys that match
    thrust::stable_sort_by_key( d_both_keys, d_both_keys + both_count, d_xvals );
    auto nend = thrust::unique_by_key( d_both_keys, d_both_keys + both_count, d_xvals );
    size_t unique_count = nend.second - d_xvals;
    result->impl.create_keys(d_both_keys,unique_count,false,result->_keys);
    // done with keys
    // update the values to their new positions using the xvals created above
    if( size() )
    {
        size_t vcount = size();
        const int* d_values = values();
        result->_values.resize(vcount);
        int* d_new_values = const_cast<int*>(result->values());
        // map the new positions
        thrust::host_vector<int> yvals(kcount,-1);
        int* d_yvals = yvals.data();
        thrust::for_each_n(thrust::host, thrust::make_counting_iterator<size_t>(0), unique_count,
            [d_yvals, d_xvals] (size_t idx) {
                int map_id = d_xvals[idx];
                if( map_id >= 0 )
                    d_yvals[map_id] = idx;
            });
        // apply new positions to new category values
        thrust::for_each_n( thrust::host, thrust::make_counting_iterator<size_t>(0), vcount,
            [d_values, d_yvals, d_new_values] (size_t idx) {
                int value = d_values[idx];
                d_new_values[idx] = (value < 0 ? value : d_yvals[value]);
            });
    }

    return result;
}

template<typename T, class Impl>
inline category<T>* category<T,Impl>::remove_keys( const T* items, size_t count )
{
    size_t kcount = keys_size();
    if( count==0 || kcount==0 )
        return copy();
    category<T>* result = new category<T>;
    //result->_bitmask.insert(_bitmask.begin(),_bitmask.end() ); // this does not change
    size_t both_count = kcount + count;
    thrust::host_vector<T> both_keys(both_count);  // first combine both keysets
    T* d_both_keys = both_keys.data();
    memcpy( d_both_keys, keys(), kcount*sizeof(T) );         // these keys
    memcpy( d_both_keys + kcount, items, count*sizeof(T) );  // and those keys
    thrust::host_vector<int> xvals(both_count); 
    int* d_xvals = xvals.data(); // build vector like: 0,...,(kcount-1),-1,...,-count
    thrust::for_each_n(thrust::host, thrust::make_counting_iterator<int>(0), (int)(both_count-1),
        [d_xvals, kcount] (int idx) {
            if( idx < kcount )
                d_xvals[idx] = idx;             //  0 ... (kcount-1)
            else
                d_xvals[idx]= kcount - idx -1;  // -1 ... -count
        });
    // compute the new keyset by doing sort/unique
    // stable-sort preserves order for keys that match
    thrust::stable_sort_by_key( d_both_keys, d_both_keys + both_count, d_xvals );
    thrust::host_vector<int> yvals(both_count);
    int* d_yvals = yvals.data();
    thrust::for_each_n( thrust::host, thrust::make_counting_iterator<int>(0), (int)(both_count-1),
        [d_yvals, d_both_keys] (int idx) { d_yvals[idx] = (int)(d_both_keys[idx]==d_both_keys[idx+1]); });
    size_t unique_count = both_count;
    {
        thrust::host_vector<int> indexes(both_count);
        int* d_indexes = indexes.data();
        int* d_end = thrust::copy_if( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(both_count), d_indexes,
            [d_xvals, d_yvals] (int idx) { return (d_xvals[idx]>=0) && (d_yvals[idx]==0); });
        unique_count = (size_t)(d_end - d_indexes);
        result->impl.create_keys( d_both_keys, d_indexes, unique_count, false, result->_keys);
        //
        thrust::host_vector<int> new_xvals(unique_count);
        thrust::gather( d_indexes, d_indexes+unique_count, d_xvals, new_xvals.data() );
        xvals.swap(new_xvals);
        d_xvals = xvals.data();
    }
    // done with the keys
    // now remap values to their new positions
    if( size() )
    {
        size_t vcount = size();
        const int* d_values = values();
        result->_values.resize(vcount);
        int* d_new_values = const_cast<int*>(result->values());
        // values pointed to removed keys will now have index=-1
        thrust::fill( d_yvals, d_yvals + kcount, -1 );
        thrust::for_each_n( thrust::host, thrust::make_counting_iterator<int>(0), (int)unique_count,
            [d_yvals, d_xvals] (int idx) { d_yvals[d_xvals[idx]] = idx; });
        thrust::for_each_n( thrust::host, thrust::make_counting_iterator<int>(0), (int)vcount, 
            [d_values, d_yvals, d_new_values] (int idx) {
                int value = d_values[idx];
                d_new_values[idx] = ( value < 0 ? value : d_yvals[value] );
            });
    }
    return result;
}

template<typename T, class Impl>
inline category<T>* category<T,Impl>::remove_unused_keys( const T* items_in, size_t count )
{
    return nullptr;
}

template<typename T, class Impl>
inline category<T>* category<T,Impl>::set_keys( const T* items_in, size_t count )
{
    return nullptr;
}


template<typename T, class Impl>
inline category<T>* category<T,Impl>::merge_category( category<T>& cat )
{
    return nullptr;
}

template<typename T, class Impl>
inline category<T>* category<T,Impl>::gather(const int* pos, size_t elements )
{
    return nullptr;
}

}