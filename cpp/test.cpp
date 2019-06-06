
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <typeinfo>
#include "include/category.h"

// nvcc -w -std=c++11 --expt-extended-lambda test.cpp src/category.cpp -o test
bool is_item_null( const BYTE* nulls, int idx );

void printValues( const int* values, int count )
{
    for( int i=0; i < count; ++i )
        std::cout << " " << values[i];
    std::cout << "\n";
}


int g_ivals[] = { 4,1,2,3,2,1,4,1,1 };
int g_ivals2[] = { 2,4,3,0 };
long g_lvals[] = { 1,1,4,1,2,3,2,1,4 };
long g_lvals2[] = { 3,4,0,2 };
float g_fvals[] = { 2.0,1.0,1.25,1.50,1.0,1.25,1.0,1.0,2.0 };
float g_fvals2[] = { 2.0,1.0,1.75,0.0 };
double g_dvals[] = { 2.0,1.0,1.0,1e25,1.0,1e25,1e50,1.0,2.0 };
double g_dvals2[] = { 2.0,1.0,1e75,0.0 };
std::string g_svals[] = { "e", "a", "d", "b", "c", "c", "c", "e", "a" };
std::string g_svals2[] = { "b", "c", "d", "f" };


template<typename T>
void printKeys( cudf::category<T>& cat )
{
    const T* keys = cat.keys();
    int count = (int)cat.keys_size();
    const BYTE* nulls = cat.nulls_bitmask();
    for( size_t idx=0; idx < count; ++idx )
    {
        if( is_item_null(nulls,idx) )
            std::cout << " -";
        else
            std::cout << " " << keys[idx];
    }
    std::cout << "\n";
}

template<typename T>
void printType( const T* keys, size_t count, const BYTE* nulls=nullptr )
{
    for( size_t idx=0; idx < count; ++idx )
    {
        if( is_item_null(nulls,idx) )
            std::cout << "- ";
        else
            std::cout << keys[idx] << " ";
    }
    std::cout << "\n";
}

void printNulls( BYTE* nulls, int count )
{
    for( int idx=0; idx < count; ++idx )
    {
        int flag = (int)((nulls[idx/8] & (1 << (idx % 8)))>0);
        std::cout << flag << " ";
    }
    std::cout << "\n";
}

template<typename T>
void testcat( const T* data1, const T* data2 )
{
    std::cout << "-----------------\n";
    printType(data1,9);

    cudf::category<T> cat( data1, 9 );
    std::cout << typeid(T).name() << " cat("<< cat.keys_size() << "," << cat.size() << ")\n";
    cat.print();

    // add keys
    std::cout << " add_keys: ";
    printType(data2,4);
    cudf::category<T>* addcat = cat.add_keys( data2, 4 );
    addcat->print(" ");

    // remove unused keys
    std::cout << " remove_unused\n";
    cudf::category<T>* unucat = addcat->remove_unused_keys();
    unucat->print(" ");
    delete unucat;
    delete addcat;

    // remove keys
    std::cout << " remove_keys: ";
    printType(data2+2,2);
    cudf::category<T>* rmvcat = cat.remove_keys( data2+2, 2 );
    rmvcat->print(" ");
    delete rmvcat;

    // set keys
    std::cout << " set_keys: ";
    printType(data2,4);
    cudf::category<T>* setcat = cat.set_keys( data2, 4 );
    setcat->print(" ");
    delete setcat;
    // null keyset
    cudf::category<T>* nullcat = cat.set_keys( nullptr, 0 );
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
    int gatidxs[] = { 2,3,1,1,3,3 };
    std::cout << " gather: ";
    printValues(gatidxs,6);
    cudf::category<T>* gatcat = cat.gather( gatidxs, 6 );
    gatcat->print(" ");
    delete gatcat;
    std::cout << " gather_and_remap: ";
    printValues(gatidxs,6);
    gatcat = cat.gather_and_remap( gatidxs, 6 );
    gatcat->print(" ");
    delete gatcat;
    std::cout << " gather_values: ";
    printValues(gatidxs,6);
    gatcat = cat.gather_values( gatidxs, 6 );
    gatcat->print(" ");
    delete gatcat;
    std::cout << " gather_type: ";
    printValues(gatidxs,6);
    T* gout = new T[6];
    cat.gather_type( gatidxs, 6, gout );
    //for( int idx=0; idx < 6; ++idx )
    //    std::cout << " " << gout[idx];
    //std::cout << "\n";
    std::cout << " "; printType(gout,6);
    delete[] gout;

    // merge
    std::cout << " merge ";
    cudf::category<T> twocat( data2, 4 );
    twocat.print(" ");
    //printKeys(twocat);
    cudf::category<T>* mrgcat = cat.merge(twocat);
    mrgcat->print(" ");
    delete mrgcat;
}

BYTE g_nulls1[] = { '\x5D', '\x00' };
BYTE g_nulls2[] = { '\x0c' };
BYTE g_nulls3[] = { '\x03' };

template<typename T>
void testnulls( const T* data1, const T* data2 )
{
    std::cout << "nulls-----------------\n";
    printType(data1,9,g_nulls1);
    cudf::category<T> cat( data1, 9, g_nulls1 );
    const int* values = cat.values();
    int count = (int)cat.size();
    std::cout << typeid(T).name() << " cat("<< cat.keys_size() << "," << count << ")\n";
    cat.print();

    // add keys
    std::cout << " add_keys: ";
    printType(data2,4,g_nulls2);
    cudf::category<T>* addcat = cat.add_keys( data2, 4, g_nulls2 );
    addcat->print(" ");

    // remove unused keys
    std::cout << " remove_unused\n";
    cudf::category<T>* unucat = addcat->remove_unused_keys();
    unucat->print(" ");
    delete unucat;
    delete addcat;

    // remove keys
    std::cout << " remove_keys: ";
    printType(data2+2,2,g_nulls3);
    cudf::category<T>* rmvcat = cat.remove_keys( data2+2, 2, g_nulls3 );
    rmvcat->print(" ");
    delete rmvcat;

    // set keys
    std::cout << " set_keys: ";
    printType(data2,4,g_nulls3);
    cudf::category<T>* setcat = cat.set_keys( data2, 4, g_nulls3 );
    setcat->print(" ");
    delete setcat;

    // gather
    int gatidxs[] = { 2,3,1,0,3,3 };
    std::cout << " gather: "; printValues(gatidxs,6);
    cudf::category<T>* gatcat = cat.gather( gatidxs, 6 );
    gatcat->print(" ");
    delete gatcat;
    gatidxs[3] = 1;
    std::cout << " gather_and_remap: "; printValues(gatidxs,6);
    gatcat = cat.gather_and_remap( gatidxs, 6 );
    gatcat->print(" ");
    delete gatcat;
    std::cout << " gather_values: "; printValues(gatidxs,6);
    gatcat = cat.gather_values( gatidxs, 6 );
    gatcat->print(" ");
    delete gatcat;
    gatidxs[3] = 0;
    std::cout << " gather_type: "; printValues(gatidxs,6);
    T* gout = new T[6];
    BYTE* gatnulls = new BYTE[1];
    cat.gather_type( gatidxs, 6, gout, gatnulls );
    std::cout << " "; printType(gout,6,gatnulls);
    delete[] gout;
    delete[] gatnulls;

    // merge
    std::cout << " merge ";
    cudf::category<T> twocat( data2, 4, g_nulls3 );
    twocat.print(" ");
    cudf::category<T>* mrgcat = cat.merge(twocat);
    mrgcat->print(" ");
    delete mrgcat;
}

int main( int argc, const char** argv )
{
    testcat<int>( g_ivals, g_ivals2 );
    testnulls<int>( g_ivals, g_ivals2 );
    //testcat<float>( g_fvals, g_fvals2 );
    //testnulls<float>( g_fvals, g_fvals2 );
    //testcat<std::string>( g_svals, g_svals2 );
    //testnulls<std::string>( g_svals, g_svals2 );

    //testcat<long>( g_lvals, g_lvals2 );
    //testcat<double>( g_dvals, g_dvals2 );

    
    return 0;
}
