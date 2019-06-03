
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <typeinfo>
#include "include/category.h"

// nvcc -w -std=c++11 --expt-extended-lambda test.cpp src/category.cpp -o test

void printMap( const char* title, int* pMap, int count )
{
    if( title )
        printf("%s:\n ",title);
    for( int i=0; i < count; ++i )
        printf("%d:%d ",i,pMap[i]);
    printf("\n");
}

void printValues( const int* values, int count )
{
    for( int i=0; i < count; ++i )
        printf(" %d",values[i]);
    printf("\n");
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
    for( size_t idx=0; idx < count; ++idx )
        std::cout << " " << keys[idx];
    std::cout << "\n";
}

template<typename T>
void testcat( const T* data1, const T* data2 )
{
    std::cout << "-----------------\n";
    for( int idx=0; idx < 9; ++idx )
        std::cout << data1[idx] << " ";
    std::cout << "\n";

    cudf::category<T> cat( data1, 9 );
    const int* values = cat.values();
    int count = (int)cat.size();
    std::cout << typeid(T).name() << " cat("<< cat.keys_size() << "," << count << ")\n";
    printKeys(cat);
    printValues(values,count);

    // add keys
    std::cout << " add_keys: ";
    for( int idx=0; idx < 4; ++idx )
        std::cout << data2[idx] << " ";
    std::cout << "\n";
    cudf::category<T>* addcat = cat.add_keys( data2, 4 );
    printKeys(*addcat);
    printValues(addcat->values(),addcat->size());

    // remove unused keys
    std::cout << " remove_unused\n";
    cudf::category<T>* unucat = addcat->remove_unused_keys();
    printKeys(*unucat);
    printValues(unucat->values(),unucat->size());
    delete unucat;
    delete addcat;

    // remove keys
    std::cout << " remove_keys: ";
    for( int idx=2; idx < 4; ++idx )
        std::cout << data2[idx] << " ";
    std::cout << "\n";
    cudf::category<T>* rmvcat = cat.remove_keys( data2+2, 2 );
    printKeys(*rmvcat);
    printValues(rmvcat->values(),rmvcat->size());
    delete rmvcat;

    // set keys
    std::cout << " set_keys: ";
    for( int idx=0; idx < 4; ++idx )
        std::cout << data2[idx] << " ";
    std::cout << "\n";
    cudf::category<T>* setcat = cat.set_keys( data2, 4 );
    printKeys(*setcat);
    printValues(setcat->values(),setcat->size());
    delete setcat;
    // null keyset
    cudf::category<T>* nullcat = cat.set_keys( nullptr, 0 );
    std::cout << " null keyset size = " << nullcat->keys_size() << "\n";
    printValues(nullcat->values(),nullcat->size());
    setcat = nullcat->set_keys( data2, 4 );
    printKeys(*setcat);
    printValues(setcat->values(),setcat->size());
    delete setcat;
    delete nullcat;

    // gather
    std::cout << " gather indexes=[2,3,1,1,3,3]\n";
    int gvals[] = { 2,3,1,1,3,3 };
    cudf::category<T>* gatcat = cat.gather( gvals, 6 );
    printKeys(*gatcat);
    printValues(gatcat->values(),gatcat->size());
    delete gatcat;

    // merge
    std::cout << " merge ";
    cudf::category<T> twocat( data2, 4 );
    printKeys(twocat);
    cudf::category<T>* mrgcat = cat.merge(twocat);
    printKeys(*mrgcat);
    printValues(mrgcat->values(),mrgcat->size());
    delete mrgcat;
}


int main( int argc, const char** argv )
{
    testcat<int>( g_ivals, g_ivals2 );
    //testcat<float>( g_fvals, g_fvals2 );
    //testcat<std::string>( g_svals, g_svals2 );

    //testcat<long>( g_lvals, g_lvals2 );
    //testcat<double>( g_dvals, g_dvals2 );

    return 0;
}
