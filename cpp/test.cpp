
#include <cstdio>
#include <vector>
#include <string>
#include "include/dstring.h"
#include "include/category.h"

// nvcc -w -std=c++11 --expt-extended-lambda test.cpp -o test

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
float g_fvals[] = { 0.0,1.0,1.25,1.50,1.0,1.25,1.0,1.0,0 };
const char* g_cvals[] = { "e", "a", "d",  "b", "c", "c",  "c", "e", "a" };
std::string g_svals[] = { "e", "a", "d",  "b", "c", "c",  "c", "e", "a" };


void test_int()
{
    cudf::category<int> intcat( g_ivals, 9 );
    const int* values = intcat.values();
    int count = (int)intcat.size();
    printf("intcat(%ld,%ld)\n", intcat.keys_size(), intcat.size());
    for( int idx=0; idx < (int)intcat.keys_size(); ++idx )
        printf(" %d",intcat.keys()[idx]);
    printf("\n");
    printValues(values,count);

    // add keys
    printf(" int add_keys\n");
    int avals[] = { 2,4,3,0 };
    cudf::category<int>* addcat = intcat.add_keys( avals, 4 );
    for( int idx=0; idx < (int)addcat->keys_size(); ++idx )
        printf(" %d",addcat->keys()[idx]);
    printf("\n");
    printValues(addcat->values(),addcat->size());
    delete addcat;

    // remove keys
    printf(" int remove_keys\n");
    int rvals[] = { 4,0 };
    cudf::category<int>* rmvcat = intcat.remove_keys( rvals, 2 );
    for( int idx=0; idx < (int)rmvcat->keys_size(); ++idx )
        printf(" %d",rmvcat->keys()[idx]);
    printf("\n");
    printValues(rmvcat->values(),rmvcat->size());
    delete rmvcat;
}

void test_float()
{
    cudf::category<float> fltcat( g_fvals, 9 );
    printf("fltcat(%ld,%ld)\n", fltcat.keys_size(), fltcat.size());
    for( int idx=0; idx < (int)fltcat.keys_size(); ++idx )
        printf(" %g",fltcat.keys()[idx]);
    printf("\n");
    printValues(fltcat.values(),fltcat.size());
}

void test_string()
{
    cudf::category<std::string> strcat( g_svals, 9 );
    printf("strcat(%ld,%ld)\n", strcat.keys_size(), strcat.size());
    for( int idx=0; idx < (int)strcat.keys_size(); ++idx )
        printf(" %s",(strcat.keys()[idx]).c_str());
    printf("\n");
    printValues(strcat.values(),strcat.size());
}

void test_dstring()
{
    // create dstring objects from the g_svals
    std::vector<cudf::dstring> custrs;
    for( int idx=9; idx > 0; --idx ) // copying backwards just for kicks
        custrs.push_back( {g_svals[idx-1].c_str(), g_svals[idx-1].length()+1 } ); //+1 for terminator
    
    cudf::category<cudf::dstring> cucat( custrs.data(), custrs.size() );
    printf("cucat(%ld,%ld)\n", cucat.keys_size(), cucat.size());
    for( int idx=0; idx < (int)cucat.keys_size(); ++idx )
        printf(" %s",(cucat.keys()[idx]).data()); // need terminator for this
    printf("\n");
    printValues(cucat.values(),cucat.size());
}

int main( int argc, const char** argv )
{
    test_int();
    test_float();
    test_string();
    test_dstring();

    return 0;
}
