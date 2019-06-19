
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <typeinfo>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "include/category.h"

//
// nvcc -w -std=c++11 --expt-extended-lambda -gencode arch=compute_70,code=sm_70 test.cu src/category.cu src/category_int.cu src/category_float.cu -o test
//

static bool is_item_null( const BYTE* nulls, int idx )
{
    return nulls && ((nulls[idx/8] & (1 << (idx % 8)))==0);
}


int g_ivals[] = { 4,1,2,3,2,1,4,1,1 };
int g_ivals2[] = { 2,4,3,0 };
long g_lvals[] = { 1,1,4,1,2,3,2,1,4 };
long g_lvals2[] = { 3,4,0,2 };
float g_fvals[] = { 2.0,1.0,1.25,1.50,1.0,1.25,1.0,1.0,2.0 };
float g_fvals2[] = { 2.0,1.0,1.75,0.0 };
double g_dvals[] = { 2.0,1.0,1.0,1e25,1.0,1e25,1e50,1.0,2.0 };
double g_dvals2[] = { 2.0,1.0,1e75,0.0 };


void printValues( const int* values, int count )
{
    std::vector<int> h_values(count);
    cudaMemcpy(h_values.data(), values, count*sizeof(int), cudaMemcpyDeviceToHost);
    for( int i=0; i < count; ++i )
        std::cout << " " << h_values[i];
    std::cout << "\n";
}

template<typename T>
void printType( const T* keys, size_t count, const BYTE* nulls=nullptr )
{
    std::vector<T> h_keys(count,1);
    cudaMemcpy(h_keys.data(), keys, count*sizeof(T), cudaMemcpyDeviceToHost);
    size_t byte_count = (count+7)/8;
    std::vector<BYTE> hnulls(byte_count);
    BYTE* h_nulls = nullptr;
    if( nulls )
    {
        h_nulls = hnulls.data();
        cudaMemcpy(h_nulls,nulls,byte_count,cudaMemcpyDeviceToHost);
    }
    for( size_t idx=0; idx < count; ++idx )
    {
        if( is_item_null(h_nulls,idx) )
            std::cout << "- ";
        else
            std::cout << h_keys[idx] << " ";
    }
    std::cout << "\n";
}


template<typename T>
void testcat( const T* data1, const T* data2 )
{
    std::cout << "-----------------\n";
    printType(data1,9);

    custr::category<T> cat( data1,9 );
    std::cout << typeid(T).name() << " cat("<< cat.keys_size() << "," << cat.size() << ")\n";
    cat.print();

    // add keys
    std::cout << " add_keys: ";
    printType(data2,4);
    custr::category<T>* addcat = cat.add_keys( data2, 4 );
    addcat->print(" ");

    // remove unused keys
    std::cout << " remove_unused\n";
    custr::category<T>* unucat = addcat->remove_unused_keys();
    unucat->print(" ");
    delete unucat;
    delete addcat;

    // remove keys
    std::cout << " remove_keys: ";
    printType(data2+2,2);
    custr::category<T>* rmvcat = cat.remove_keys( data2+2, 2 );
    rmvcat->print(" ");
    delete rmvcat;

    // set keys
    std::cout << " set_keys: ";
    printType(data2,4);
    custr::category<T>* setcat = cat.set_keys( data2, 4 );
    setcat->print(" ");
    delete setcat;
    // null keyset
    custr::category<T>* nullcat = cat.set_keys( nullptr, 0 );
    std::cout << " null keyset size = " << nullcat->keys_size() << "\n";
    printValues(nullcat->values(),nullcat->size());
    nullcat->print(" ");
    std::cout << " set_keys on nullset: ";
    printType(data2,4);
    setcat = nullcat->set_keys( data2, 4 );
    setcat->print(" ");
    delete setcat;
    delete nullcat;

    // gather
    thrust::device_vector<int> gatidxs(6,3);
    //int gatidxs[] = { 2,3,1,1,3,3 };
    gatidxs[0] = 2; gatidxs[2] = 1; gatidxs[3] = 1;
    std::cout << " gather: ";
    printValues(gatidxs.data().get(),6);
    custr::category<T>* gatcat = cat.gather( gatidxs.data().get(), 6 );
    gatcat->print(" ");
    delete gatcat;
    std::cout << " gather_and_remap: ";
    printValues(gatidxs.data().get(),6);
    gatcat = cat.gather_and_remap( gatidxs.data().get(), 6 );
    gatcat->print(" ");
    delete gatcat;
    std::cout << " gather_values: ";
    printValues(gatidxs.data().get(),6);
    gatcat = cat.gather_values( gatidxs.data().get(), 6 );
    gatcat->print(" ");
    delete gatcat;
    std::cout << " gather_type: ";
    printValues(gatidxs.data().get(),6);
    thrust::device_vector<T> gout(6);
    cat.gather_type( gatidxs.data().get(), 6, gout.data().get() );
    std::cout << " "; printType(gout.data().get(),6);

    // merge
    std::cout << " merge ";
    custr::category<T> twocat( data2, 4 );
    twocat.print(" ");
    custr::category<T>* mrgcat = cat.merge(twocat);
    mrgcat->print(" ");
    delete mrgcat;
}

//BYTE g_nulls1[] = { '\x5D', '\x00' };
//BYTE g_nulls2[] = { '\x0c' };
//BYTE g_nulls3[] = { '\x03' };
BYTE g_nulls[] = { '\x5D', '\x00',
                   '\x0C',
                   '\x03' };

template<typename T>
void testnulls( const T* data1, const T* data2 )
{
    BYTE* d_nulls = 0;
    cudaMalloc(&d_nulls, sizeof(g_nulls));
    cudaMemcpy(d_nulls, g_nulls, sizeof(g_nulls), cudaMemcpyHostToDevice);
    BYTE* d_nulls1 = d_nulls;
    BYTE* d_nulls2 = d_nulls+2;
    BYTE* d_nulls3 = d_nulls+3;

    std::cout << "nulls-----------------\n";
    printType(data1,9,d_nulls1);
    custr::category<T> cat( data1, 9, d_nulls1 );
    const int* values = cat.values();
    int count = (int)cat.size();
    std::cout << typeid(T).name() << " cat("<< cat.keys_size() << "," << count << ")\n";
    cat.print();

    // add keys
    std::cout << " add_keys: ";
    printType(data2,4,d_nulls2);
    custr::category<T>* addcat = cat.add_keys( data2, 4, d_nulls2 );
    addcat->print(" ");

    // remove unused keys
    std::cout << " remove_unused\n";
    custr::category<T>* unucat = addcat->remove_unused_keys();
    unucat->print(" ");
    delete unucat;
    delete addcat;

    // remove keys
    std::cout << " remove_keys: ";
    printType(data2+2,2,d_nulls3);
    custr::category<T>* rmvcat = cat.remove_keys( data2+2, 2, d_nulls3 );
    rmvcat->print(" ");
    delete rmvcat;

    // set keys
    std::cout << " set_keys: ";
    printType(data2,4,d_nulls3);
    custr::category<T>* setcat = cat.set_keys( data2, 4, d_nulls3 );
    setcat->print(" ");
    delete setcat;

    // gather
    thrust::device_vector<int> gatidxs(6,3);
    //int gatidxs[] = { 2,3,1,1,3,3 };
    gatidxs[0] = 2; gatidxs[2] = 1; gatidxs[3] = 1;
    std::cout << " gather: "; printValues(gatidxs.data().get(),6);
    custr::category<T>* gatcat = cat.gather( gatidxs.data().get(), 6 );
    gatcat->print(" ");
    delete gatcat;
    gatidxs[3] = 1;
    std::cout << " gather_and_remap: "; printValues(gatidxs.data().get(),6);
    gatcat = cat.gather_and_remap( gatidxs.data().get(), 6 );
    gatcat->print(" ");
    delete gatcat;
    std::cout << " gather_values: "; printValues(gatidxs.data().get(),6);
    gatcat = cat.gather_values( gatidxs.data().get(), 6 );
    gatcat->print(" ");
    delete gatcat;
    gatidxs[3] = 0;
    std::cout << " gather_type: "; printValues(gatidxs.data().get(),6);
    thrust::device_vector<T> gout(6);
    thrust::device_vector<BYTE> gatnulls(1);
    cat.gather_type( gatidxs.data().get(), 6, gout.data().get(), gatnulls.data().get() );
    std::cout << " "; printType(gout.data().get(),6,gatnulls.data().get());

    // merge
    std::cout << " merge ";
    custr::category<T> twocat( data2, 4, d_nulls3 );
    twocat.print(" ");
    custr::category<T>* mrgcat = cat.merge(twocat);
    mrgcat->print(" ");
    delete mrgcat;
}

template<typename T>
T* get_dev_data(T* data, size_t count)
{
    T* d_data = nullptr;
    cudaMalloc(&d_data, count*sizeof(T));
    cudaMemcpy(d_data, data, count*sizeof(T), cudaMemcpyHostToDevice);
    return d_data;
}

int main( int argc, const char** argv )
{

    testcat<int>( get_dev_data(g_ivals,9), get_dev_data(g_ivals2,4) );
    testnulls<int>( get_dev_data(g_ivals,9), get_dev_data(g_ivals2,4) );
    testcat<float>( get_dev_data(g_fvals,9), get_dev_data(g_fvals2,4) );
    testnulls<float>( get_dev_data(g_fvals,9), get_dev_data(g_fvals2,4) );
    //testcat<std::string>( g_svals, g_svals2 );
    //testnulls<std::string>( g_svals, g_svals2 );

    //testcat<long>( g_lvals, g_lvals2 );
    //testcat<double>( g_dvals, g_dvals2 );

    
    return 0;
}
