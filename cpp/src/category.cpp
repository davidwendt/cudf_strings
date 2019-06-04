
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
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
#include <thrust/tabulate.h>
#include "../include/category.h"

void printValues( const int* values, int count );

void printInts( const char* title, int* pMap, int count )
{
    if( title )
        printf("%s",title);
    for( int i=0; i < count; ++i )
        printf("%2d ",pMap[i]);
    printf("\n");
}


bool is_item_null( const BYTE* nulls, int idx )
{
    return nulls && ((nulls[idx/8] & (1 << (idx % 8)))==0);
}

size_t count_nulls( const BYTE* nulls, size_t count )
{
    return thrust::count_if( thrust::host, thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(count),
            [nulls] (size_t idx) { return ((nulls[idx/8] & (1 << (idx % 8)))==0); });
}

namespace cudf
{

template<typename T>
class category_impl
{
public:
    thrust::host_vector<T> _keys;
    thrust::host_vector<int> _values;
    thrust::host_vector<BYTE> _bitmask;

    void init_keys( const T* items, size_t count )
    {
        _keys.resize(count);
        //memcpy( keys.data(), items, ucount*sizeof(T) );
        thrust::copy( items, items + count, _keys.data() );
    }

    void init_keys( const T* items, const int* indexes, size_t count )
    {
        _keys.resize(count);
        thrust::gather(thrust::host, indexes, indexes + count, items, _keys.data());
    }

    const T* get_keys()
    {
        return _keys.data();
    }

    size_t keys_count()
    {
        return _keys.size();
    }

    int* get_values(size_t count)
    {
        _values.resize(count);
        return _values.data();
    }

    const int* get_values()
    {
        return _values.data();
    }

    size_t values_count()
    {
        return _values.size();
    }

    void set_values( const int* vals, size_t count )
    {
        _values.resize(count);
        thrust::copy( vals, vals+count, _values.data());
    }

    BYTE* get_nulls(size_t count)
    {
        if( count==0 )
            return nullptr;
        size_t byte_count = (count+7)/8;
        _bitmask.resize(byte_count);
        return _bitmask.data();
    }

    const BYTE* get_nulls()
    {
        if( _bitmask.empty() )
            return nullptr;
        return _bitmask.data();
    }

    void set_nulls( const BYTE* nulls, size_t count )
    {
        if( nulls==nullptr || count==0 )
            return;
        size_t byte_count = (count+7)/8;
        _bitmask.resize(byte_count);
        thrust::copy(nulls, nulls + byte_count, _bitmask.data());
    }

    void reset_nulls()
    {
        _bitmask.resize(0);
    }
};


template<typename T>
category<T>::category()
{
    pImpl = new category_impl<T>;
}

template<typename T>
category<T>::category( const category& ) : pImpl(nullptr)
{}

template<typename T>
category<T>::~category()
{
    delete pImpl;
}

template<typename T>
category<T>::category( const T* items, size_t count, BYTE* nulls )
            : pImpl(nullptr)
{
    pImpl = new category_impl<T>;

    thrust::host_vector<int> indexes(count);
    thrust::sequence(indexes.begin(), indexes.end()); // 0,1,2,3,4,5,6,7,8
    thrust::sort(thrust::host, indexes.begin(), indexes.end(),
        [items, nulls] (int lhs, int rhs) {
            bool lhs_null = is_item_null(nulls,lhs);
            bool rhs_null = is_item_null(nulls,rhs);
            if( lhs_null || rhs_null )
                return !rhs_null; // sorts: null < non-null
            return items[lhs] < items[rhs];
        });
    int* d_values = pImpl->get_values(count);
    int* d_indexes = indexes.data();

#if OLDWAY
    printf("(old way)\n");
    // this will set d_values to 0 for matches and 1 for no match
    thrust::for_each_n(thrust::host, thrust::make_counting_iterator<size_t>(0), count,
        [items, nulls, d_indexes, d_values] (size_t idx) {
            if( idx==0 )
            {
                d_values[0] = 0;
                return;
            }
            bool lhs_null = is_item_null(nulls,idx-1);
            bool rhs_null = is_item_null(nulls,idx);
            if( lhs_null || rhs_null )
                d_values[idx] = (int)(lhs_null != rhs_null);
            else
                d_values[idx] = (int)(items[d_indexes[idx-1]] != items[d_indexes[idx]]);
        });
    int ucount = thrust::reduce(thrust::host, d_values, d_values+count) +1;
#else        
    //printf("(new way)\n");
    thrust::host_vector<int> map_indexes(count);
    int* d_map_indexes = map_indexes.data();
    int* d_nend = thrust::copy_if( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(count), d_map_indexes,
        [items, nulls, d_indexes, d_values] (int idx) {
            if( idx==0 )
            {
                d_values[0] = 0;
                return true;
            }
            bool lhs_null = is_item_null(nulls,idx-1);
            bool rhs_null = is_item_null(nulls,idx);
            bool isunique = true;
            if( lhs_null || rhs_null )
                isunique = (lhs_null != rhs_null);
            else
                isunique = (items[d_indexes[idx-1]] != items[d_indexes[idx]]);
            d_values[idx] = (int)isunique;
            return isunique;
        });
    int ucount = (int)(d_nend - d_map_indexes);
#endif    
    thrust::host_vector<int> keys_indexes(ucount);
#ifdef OLDWAY    
    // make a copy of just the unique values
    thrust::unique_copy(thrust::host, indexes.begin(), indexes.end(), keys_indexes.begin(),
        [items, nulls] ( int lhs, int rhs ) {
            bool lhs_null = is_item_null(nulls,lhs);
            bool rhs_null = is_item_null(nulls,rhs);
            if( lhs_null || rhs_null )
                return lhs_null==rhs_null;
            return items[lhs]==items[rhs];
        });
    // next 3 lines replace unique above but avoids a comparison and operates only on integers
    //thrust::host_vector<int> map_indexes(ucount);
    //thrust::copy_if( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(count), map_indexes.begin(), [d_values] (int idx) { return (idx==0) || d_values[idx]; });
    //thrust::gather( map_indexes.begin(), map_indexes.end(), indexes.begin(), keys_indexes.begin() );
#else    
    thrust::gather( d_map_indexes, d_nend, indexes.begin(), keys_indexes.begin() );
#endif    
    // scan will produce the resulting values
    thrust::inclusive_scan(thrust::host, d_values, d_values+count, d_values);
    // sort will put them in the correct order
    thrust::sort_by_key(thrust::host, indexes.begin(), indexes.end(), d_values);
    // gather the keys for this category
    pImpl->init_keys(items,keys_indexes.data(),ucount);
    // just make a copy of the nulls bitmask
    if( nulls && count_nulls(nulls,count) )
        pImpl->set_nulls(nulls,count);
}

template<typename T>
size_t category<T>::size()
{
    return pImpl->values_count();
}

template<typename T>
size_t category<T>::keys_size()
{
    return pImpl->keys_count();
}

template<typename T>
const T* category<T>::keys()
{
    return pImpl->get_keys();
}

template<typename T>
const int* category<T>::values()
{
    return pImpl->get_values();
}

template<typename T>
const T category<T>::get_key_for(int idx)
{
    return pImpl->_keys[idx];
} //

template<typename T>
const BYTE* category<T>::nulls_bitmask()
{
    return pImpl->get_nulls();
}

template<typename T>
void category<T>::printType( T* items, size_t count )
{
    for( size_t idx=0; idx < count; ++idx )
        std::cout << items[idx] << " ";
    std::cout << "\n";
}

template<typename T>
category<T>* category<T>::copy()
{
    category<T>* result = new category<T>;
    result->pImpl->init_keys(pImpl->get_keys(), pImpl->keys_count());
    result->pImpl->set_values(pImpl->get_values(),pImpl->values_count());
    result->pImpl->set_nulls(pImpl->get_nulls(),pImpl->values_count());
    return result;
}

template<typename T>
int category<T>::get_index_for(T key)
{
    const int* d_values = pImpl->get_values();
    int index = -1, count = keys_size();
    const T* d_keys = pImpl->get_keys();
    thrust::copy_if( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(count), &index,
        [d_keys, key] (int idx) { return key == d_keys[idx]; });
    return index;
}

template<typename T>
int* category<T>::get_indexes_for(T key)
{
    int index = get_index_for(key);
    const int* d_values = pImpl->get_values();
    int count = thrust::count( d_values, d_values + size(), index);
    int* results = new int[count];
    thrust::copy_if( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(count), results, 
        [index, d_values] (int idx) { return d_values[idx]==index; });
    return results;
}

template<typename T>
int* category<T>::get_indexes_for_null_key()
{
    if( pImpl->get_nulls()==nullptr )
        return nullptr; // there are no null entries
    int index = 0; // null key is always index 0
    const int* d_values = pImpl->get_values();
    int count = thrust::count( d_values, d_values + size(), index);
    int* results = new int[count];
    thrust::copy_if( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(count), results, 
        [index, d_values] (int idx) { return d_values[idx]==index; });
    return results;
}

template<typename T>
void category<T>::to_type( T* results, BYTE* nulls )
{
    const int* d_values = pImpl->get_values();
    thrust::gather(thrust::host, d_values, d_values + size(), pImpl->get_keys(), results);
    const BYTE* d_nulls = pImpl->get_nulls();
    if( d_nulls && nulls )
        thrust::copy(d_nulls, d_nulls + ((size()+7)/8), nulls);
}

template<typename T>
void category<T>::gather_type( const int* indexes, size_t count, T* results, BYTE* nulls )
{
    // should these be indexes of the values and not values themselves?
    size_t kcount = keys_size();
    int check = thrust::count_if( indexes, indexes+count, [kcount] (int val) { return (val<0) || (val>=kcount);});
    if( check > 0 )
        throw std::out_of_range("gather_type invalid index value");
    thrust::gather(thrust::host, indexes, indexes+count, pImpl->get_keys(), results);
    // need to also gather the null bits
    const BYTE* d_nulls = pImpl->get_nulls();
    if( d_nulls && nulls )
    {
        thrust::for_each_n(thrust::host, thrust::make_counting_iterator<int>(0), count,
            [indexes, d_nulls, nulls] (int idx) {
                int position = indexes[idx];
                int flag = (int)((d_nulls[position/8] & (1 << (position % 8)))>0);
                nulls[idx/8] |= (flag << (idx % 8));
            });
    }
}

template<typename T>
category<T>* category<T>::add_keys( const T* items, size_t count, const BYTE* nulls )
{
    if( count==0 )
        return copy();
    category<T>* result = new category<T>;
    // the values which are null do not change position so this is just a copy
    result->pImpl->set_nulls( pImpl->get_nulls(), size() );
    size_t kcount = keys_size();
    // incorporating the keys and adjust the values
    const BYTE* d_nulls = pImpl->get_nulls();
    const T* d_keys = pImpl->get_keys();
    size_t both_count = kcount + count; // this is not the unique count
    thrust::host_vector<T> both_keys(both_count);  // first combine both keysets
    T* d_both_keys = both_keys.data();
    thrust::copy( d_keys, d_keys + kcount, d_both_keys );
    thrust::copy( items, items + count, d_both_keys + kcount );
    thrust::host_vector<int> xvals(both_count); 
    int* d_xvals = xvals.data(); // build vector like: 0,...,(kcount-1),-1,...,-count
    thrust::tabulate(thrust::host, d_xvals, d_xvals + both_count,
        [kcount] (int idx) { return (idx < kcount) ? idx : (kcount - idx - 1); });
    // compute the new keyset by doing sort/unique
    thrust::host_vector<int> indexes(both_count);
    thrust::sequence( indexes.begin(), indexes.end() );
    int* d_indexes = indexes.data();
    // stable-sort preserves order for keys that match
    thrust::stable_sort_by_key( d_indexes, d_indexes + both_count, d_xvals, 
        [d_both_keys, kcount, d_nulls, nulls] (int lhs, int rhs) {
            bool lhs_null = ((lhs==0) && d_nulls) || ((lhs >= kcount) && is_item_null(nulls,lhs-kcount));
            bool rhs_null = ((rhs==0) && d_nulls) || ((rhs >= kcount) && is_item_null(nulls,rhs-kcount));
            if( lhs_null || rhs_null )
                return !rhs_null; // sorts: null < non-null
            return d_both_keys[lhs] < d_both_keys[rhs];
        });
    auto nend = thrust::unique_by_key( d_indexes, d_indexes + both_count, d_xvals,
        [d_both_keys, kcount, d_nulls, nulls] (int lhs, int rhs) {
            bool lhs_null = ((lhs==0) && d_nulls) || ((lhs >= kcount) && is_item_null(nulls,lhs-kcount));
            bool rhs_null = ((rhs==0) && d_nulls) || ((rhs >= kcount) && is_item_null(nulls,rhs-kcount));
            if( lhs_null || rhs_null )
                return lhs_null==rhs_null;
            return d_both_keys[lhs]==d_both_keys[rhs];
        });
    size_t unique_count = nend.second - d_xvals;
    result->pImpl->init_keys(d_both_keys,d_indexes,unique_count);
    // done with keys
    // update the values to their new positions using the xvals created above
    if( size() )
    {
        size_t vcount = size();
        const int* d_values = pImpl->get_values();
        int* d_new_values = result->pImpl->get_values(vcount);
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

template<typename T>
category<T>* category<T>::remove_keys( const T* items, size_t count, const BYTE* nulls )
{
    category<T>* result = new category<T>;
    const BYTE* d_nulls = pImpl->get_nulls();
    size_t kcount = keys_size();
    size_t both_count = kcount + count;
    const T* d_keys = pImpl->get_keys();
    thrust::host_vector<T> both_keys(both_count);  // first combine both keysets
    T* d_both_keys = both_keys.data();
    thrust::copy( d_keys, d_keys + kcount, d_both_keys );         // these keys
    thrust::copy( items, items + count, d_both_keys + kcount );  // and those keys
    thrust::host_vector<int> xvals(both_count); 
    int* d_xvals = xvals.data(); // build vector like: 0,...,(kcount-1),-1,...,-count
    thrust::tabulate(thrust::host, d_xvals, d_xvals + both_count, [kcount] (int idx) { return (idx < kcount) ? idx : (kcount - idx - 1); });
    // compute the new keyset by doing sort/unique
    thrust::host_vector<int> indexes(both_count);
    thrust::sequence( indexes.begin(), indexes.end() );
    int* d_indexes = indexes.data();
    // stable-sort preserves order for keys that match
    thrust::stable_sort_by_key( d_indexes, d_indexes + both_count, d_xvals,
        [d_both_keys, kcount, d_nulls, nulls] (int lhs, int rhs) {
            bool lhs_null = ((lhs==0) && d_nulls) || ((lhs >= kcount) && is_item_null(nulls,lhs-kcount));
            bool rhs_null = ((rhs==0) && d_nulls) || ((rhs >= kcount) && is_item_null(nulls,rhs-kcount));
            if( lhs_null || rhs_null )
                return !rhs_null; // sorts: null < non-null
            return d_both_keys[lhs] < d_both_keys[rhs];
        });
    size_t unique_count = both_count;
    {
        thrust::host_vector<int> map_indexes(both_count);
        int* d_map_indexes = map_indexes.data();
        int* d_end = thrust::copy_if( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(both_count), d_map_indexes,
            [d_both_keys, kcount, both_count, d_indexes, d_xvals, d_nulls, nulls] (int idx) {
                if( d_xvals[idx] < 0 )
                    return false;
                if( idx == both_count-1 )
                    return true;
                int lhs = d_indexes[idx], rhs = d_indexes[idx+1];
                bool lhs_null = ((lhs==0) && d_nulls) || ((lhs >= kcount) && is_item_null(nulls,lhs-kcount));
                bool rhs_null = ((rhs==0) && d_nulls) || ((rhs >= kcount) && is_item_null(nulls,rhs-kcount));
                if( lhs_null || rhs_null )
                    return lhs_null != rhs_null;
                return (d_both_keys[d_indexes[idx]] != d_both_keys[d_indexes[idx+1]]);
            });
        unique_count = (size_t)(d_end - d_map_indexes);
        thrust::host_vector<int> keys_indexes(unique_count);
        thrust::gather( d_map_indexes, d_end, d_indexes, keys_indexes.data() );
        result->pImpl->init_keys( d_both_keys, keys_indexes.data(), unique_count );
        // setup for the value remap
        thrust::host_vector<int> new_xvals(unique_count);
        thrust::gather( d_map_indexes, d_end, d_xvals, new_xvals.data() );
        xvals.swap(new_xvals);
        d_xvals = xvals.data();
    }
    // done with the keys
    // now remap values to their new positions
    size_t vcount = size();
    if( vcount )
    {
        const int* d_values = values();
        int* d_new_values = result->pImpl->get_values(vcount);
        // values pointed to removed keys will now have index=-1
        thrust::host_vector<int> yvals(kcount,-1);
        int* d_yvals = yvals.data();
        //thrust::fill( d_yvals, d_yvals + kcount, -1 );
        thrust::for_each_n( thrust::host, thrust::make_counting_iterator<int>(0), (int)unique_count,
            [d_yvals, d_xvals] (int idx) { d_yvals[d_xvals[idx]] = idx; });
        thrust::for_each_n( thrust::host, thrust::make_counting_iterator<int>(0), (int)vcount, 
            [d_values, d_yvals, d_new_values] (int idx) {
                int value = d_values[idx];
                d_new_values[idx] = ( value < 0 ? value : d_yvals[value] );
            });
    }
    // finally, handle the nulls
    // if null key is removed, then null values are abandoned (become -1)
    // so the bitmask can become undefined perhaps
    if( d_nulls )
    {
        //if( count_nulls(nulls,count) )    // remove null key
        //    result->pImpl->reset_nulls();
        //else                              // otherwise just copy them
            result->pImpl->set_nulls(d_nulls,vcount);
    }
    return result;
}

template<typename T>
category<T>* category<T>::remove_unused_keys()
{
    size_t kcount = keys_size();
    if( kcount==0 )
        return copy();
    const int* d_values = values();
    thrust::host_vector<int> usedkeys(kcount,0);
    int* d_usedkeys = usedkeys.data();
    // find the keys that not being used
    thrust::for_each_n( thrust::host, thrust::make_counting_iterator<int>(0), (int)size(),
        [d_values, d_usedkeys] (int idx) {
            int pos = d_values[idx];
            if( pos >= 0 )
                d_usedkeys[pos] = 1; // race condition not important
        });
    // compute how many are not used
    size_t count = kcount - thrust::reduce(d_usedkeys,d_usedkeys+kcount,(int)0);
    if( count==0 )
        return copy();
    //
    thrust::host_vector<T> rmv_keys(count);
    T* d_rmv_keys = rmv_keys.data();
    thrust::host_vector<int> indexes(count);
    int* d_indexes = indexes.data();
    thrust::copy_if( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(kcount), d_indexes,
        [d_usedkeys] (int idx) { return (d_usedkeys[idx]==0); });
    thrust::gather( d_indexes, d_indexes+count, pImpl->get_keys(), d_rmv_keys );
    return remove_keys( d_rmv_keys, count );
}

template<typename T>
category<T>* category<T>::set_keys( const T* items, size_t count, const BYTE* nulls )
{
    size_t kcount = keys_size();
    size_t both_count = kcount + count; // this is not the unique count
    const T* d_keys = pImpl->get_keys();
    const BYTE* d_nulls = pImpl->get_nulls();
    thrust::host_vector<T> both_keys(both_count);  // first combine both keysets
    T* d_both_keys = both_keys.data();
    thrust::copy( d_keys, d_keys + kcount, d_both_keys );        // these keys
    thrust::copy( items, items + count, d_both_keys + kcount );  // and those keys
    thrust::host_vector<int> xvals(both_count); // seq-vector for resolving old/new keys
    int* d_xvals = xvals.data(); // build vector like: 0,...,(kcount-1),-1,...,-count
    thrust::tabulate(thrust::host, d_xvals, d_xvals + both_count, [kcount] (int idx) { return (idx < kcount) ? idx : (kcount - idx - 1); });
    // sort the combined keysets
    thrust::host_vector<int> indexes(both_count);
    thrust::sequence( indexes.begin(), indexes.end() );
    int* d_indexes = indexes.data();
    // stable-sort preserves order for keys that match
    thrust::stable_sort_by_key( d_indexes, d_indexes + both_count, d_xvals, 
        [d_both_keys, kcount, d_nulls, nulls] (int lhs, int rhs) {
            bool lhs_null = ((lhs==0) && d_nulls) || ((lhs >= kcount) && is_item_null(nulls,lhs-kcount));
            bool rhs_null = ((rhs==0) && d_nulls) || ((rhs >= kcount) && is_item_null(nulls,rhs-kcount));
            if( lhs_null || rhs_null )
                return !rhs_null; // sorts: null < non-null
            return d_both_keys[lhs] < d_both_keys[rhs];
        });
    thrust::host_vector<int> map_indexes(both_count); // needed for gather methods
    int* d_map_indexes = map_indexes.data(); // indexes of keys from key1 not in key2
    int* d_copy_end = thrust::copy_if( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(both_count), d_map_indexes,
            [d_both_keys, kcount, both_count, d_indexes, d_xvals, d_nulls, nulls] (int idx) {
                if( d_xvals[idx] < 0 )
                    return true;
                if( idx == (both_count-1) )
                    return false;
                int lhs = d_indexes[idx], rhs = d_indexes[idx+1];
                bool lhs_null = ((lhs==0) && d_nulls) || ((lhs >= kcount) && is_item_null(nulls,lhs-kcount));
                bool rhs_null = ((rhs==0) && d_nulls) || ((rhs >= kcount) && is_item_null(nulls,rhs-kcount));
                if( lhs_null || rhs_null )
                    return lhs_null == rhs_null;
                return (d_both_keys[d_indexes[idx]] == d_both_keys[d_indexes[idx+1]]);
            });
    int copy_count = d_copy_end - d_map_indexes;
    if( copy_count < both_count )
    {   // if keys are removed, we need new keyset; the gather()s here will select the remaining keys
        thrust::host_vector<int> copy_indexes(copy_count);
        thrust::host_vector<int> copy_xvals(copy_count);
        thrust::gather( d_map_indexes, d_map_indexes + copy_count, d_indexes, copy_indexes.data() );  // likely, these 2 lines can be
        thrust::gather( d_map_indexes, d_map_indexes + copy_count, d_xvals, copy_xvals.data() );      // combined with a zip-iterator
        indexes.swap(copy_indexes);
        xvals.swap(copy_xvals);
        d_indexes = indexes.data();
        d_xvals = xvals.data();
        both_count = copy_count;
    }
    // resolve final key-set
    auto d_unique_end = thrust::unique_by_key( d_indexes, d_indexes + both_count, d_xvals,
        [d_both_keys, kcount, d_nulls, nulls] (int lhs, int rhs) {
            bool lhs_null = ((lhs==0) && d_nulls) || ((lhs >= kcount) && is_item_null(nulls,lhs-kcount));
            bool rhs_null = ((rhs==0) && d_nulls) || ((rhs >= kcount) && is_item_null(nulls,rhs-kcount));
            if( lhs_null || rhs_null )
                return lhs_null==rhs_null;
            return d_both_keys[lhs]==d_both_keys[rhs];
        });
    size_t unique_count = d_unique_end.second - d_xvals;//both_count - matched;
    category<T>* result = new category<T>;
    result->pImpl->init_keys( d_both_keys, d_indexes, unique_count );
    // done with keys, remap the values
    size_t vcount = size();
    if( vcount )
    {
        const int* d_values = values();
        thrust::host_vector<int> yvals(kcount,-1); // create map/stencil from old key positions
        int* d_yvals = yvals.data();
        thrust::for_each_n( thrust::host, thrust::make_counting_iterator<int>(0), (int)unique_count,
            [d_xvals, d_yvals] (int idx) {
                int value = d_xvals[idx];
                if( value >= 0 )
                    d_yvals[value] = idx; // map to new position
            });
        // create new values using the map in yvals
        int* d_new_values = result->pImpl->get_values(vcount);
        thrust::for_each_n( thrust::host, thrust::make_counting_iterator<int>(0), (int)vcount, 
            [d_values, d_yvals, d_new_values] (int idx) {
                int value = d_values[idx];
                d_new_values[idx] = ( value < 0 ? value : d_yvals[value] );
            });
        // nulls do not change
        result->pImpl->set_nulls( d_nulls, vcount ); // nulls do not change
    }
    return result;
}

template<typename T>
category<T>* category<T>::merge( category<T>& cat )
{
    // first, copy keys so we can sort/unique
    size_t kcount = keys_size();
    size_t count = kcount + cat.keys_size();
    const T* d_keys = keys();
    const T* d_catkeys = cat.keys();
    thrust::host_vector<T> keyset(count);
    T* d_keyset = keyset.data();
    thrust::copy( d_keys, d_keys + kcount, d_keyset );
    thrust::copy( d_catkeys, d_catkeys + cat.keys_size(), d_keyset + kcount );
    // build sequence vector and sort positions
    thrust::host_vector<int> xvals(count);
    int* d_xvals = xvals.data();
    thrust::sequence( xvals.begin(), xvals.end() );
    thrust::stable_sort_by_key( d_keyset, d_keyset+count, d_xvals );
    // build anti-matching indicator vector
    thrust::host_vector<int> yvals(count,0);
    int* d_yvals = yvals.data();
    thrust::for_each_n( thrust::host, thrust::make_counting_iterator<int>(1), (int)(count-1),
        [d_yvals, d_keyset] (int idx) { d_yvals[idx] = (int)(d_keyset[idx]!=d_keyset[idx-1]); });
    size_t unique_count = thrust::reduce( d_yvals, d_yvals+count ) +1;
    thrust::unique( d_keyset, d_keyset+count );
    // this is now the new keyset
    category<T>* result = new category<T>;
    result->pImpl->init_keys(d_keyset, unique_count );
    // done with keys
    // create index to map old positions to their new indexes
    thrust::inclusive_scan( d_yvals, d_yvals+count, d_yvals ); // new positions
    thrust::sort_by_key( d_xvals, d_xvals+count, d_yvals );    // creates map
    int* d_new_values = result->pImpl->get_values(size()+cat.size()); // output values
    if( size() )
    {   // remap our values
        const int* d_values = values();
        thrust::for_each_n( thrust::host, thrust::make_counting_iterator<int>(0), (int)size(),
            [d_values, d_yvals, d_new_values] (int idx) {
                int value = d_values[idx];
                d_new_values[idx] = ( value < 0 ? value : d_yvals[value]);
            });
        d_new_values += size();
        d_yvals += keys_size();
    }
    if( cat.size() )
    {   // remap arg's values
        const int* d_values = cat.values();
        thrust::for_each_n( thrust::host, thrust::make_counting_iterator<int>(0), (int)cat.size(),
            [d_values, d_yvals, d_new_values] (int idx) {
                int value = d_values[idx];
                d_new_values[idx] = ( value < 0 ? value : d_yvals[value]);
            });
    }
    // the nulls are just appended
    const BYTE* d_nulls = nulls_bitmask();
    const BYTE* d_cat_nulls = cat.nulls_bitmask();
    if( d_nulls || d_cat_nulls )
    {
        size_t vcount = size();
        size_t ncount = vcount + cat.size();
        BYTE* d_result_nulls = result->pImpl->get_nulls(ncount);
        thrust::for_each_n( thrust::host, thrust::make_counting_iterator<int>(0), ncount,
            [d_nulls, d_cat_nulls, vcount, d_result_nulls] (int idx) {
                if( idx < vcount )
                    d_result_nulls[idx/8] = d_nulls[idx/8]; // may be garbage in here
                else
                {
                    int nidx = idx - vcount;
                    int flag = (int)((d_cat_nulls[nidx/8] & (1 << (nidx % 8)))>0);
                    d_result_nulls[idx/8] |= (flag << (idx % 8)); // garbage above may impact this
                }
            });
    }
    return result;
}

template<typename T>
category<T>* category<T>::gather(const int* indexes, size_t count )
{
    size_t kcount = keys_size();
    int check = thrust::count_if( indexes, indexes + count, [kcount] (int val) { return (val<0) || (val>=kcount);});
    if( check > 0 )
        throw std::out_of_range("gather: invalid index value");
    // create histogram-ish record of keys for this gather
    thrust::host_vector<int> xvals(kcount,0);
    int* d_xvals = xvals.data();
    thrust::for_each_n( thrust::host, thrust::make_counting_iterator<size_t>(0), count,
        [d_xvals, indexes] (size_t idx) { d_xvals[indexes[idx]] = 1; });
    // create indexes of our keys for the new category
    thrust::host_vector<int> yvals(kcount,0);
    int* d_yvals = yvals.data();
    auto d_new_end = thrust::copy_if( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(kcount), d_yvals,
        [d_xvals] (int idx) { return d_xvals[idx]==1;} );
    size_t unique_count = (size_t)(d_new_end - d_yvals);
    // create new category and set the keys
    category<T>* result = new category<T>;
    result->pImpl->init_keys( keys(), d_yvals, unique_count );
    // now create values by mapping our values over to the new key positions
    thrust::exclusive_scan( d_xvals, d_xvals + kcount, d_yvals ); // reuse yvals for the map
    int* d_new_values = result->pImpl->get_values(count);
    thrust::gather( thrust::host, indexes, indexes + count, d_yvals, d_new_values );
    // also need to gather nulls
    return result;
}

// pre-define these types
template class category<int>;
//template class category<float>;
//template class category<long>;
//template class category<double>;
//template class category<std::string>;

}
