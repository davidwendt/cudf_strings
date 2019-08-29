
#include <cstdio>
#include <iostream>
#include <vector>
#include <random>
#include "include/category.h"
#include <sys/time.h>
#include <unistd.h>

// nvcc -w -std=c++11 --expt-extended-lambda test_scale.cpp src/category.cpp -o test_scale

double GetTime()
{
	timeval tv;
	gettimeofday( &tv, NULL );
	return (double)(tv.tv_sec*1000000+tv.tv_usec)/1000000.0;
}

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

template<typename T>
void printKeys( custr::category<T>& cat )
{
    const T* keys = cat.keys();
    int count = (int)cat.keys_size();
    for( size_t idx=0; idx < count; ++idx )
        std::cout << " " << keys[idx];
    std::cout << "\n";
}

void test_int(std::mt19937& mt)
{
    std::uniform_real_distribution<double> dist(1.0,10000.0);
    std::vector<int> data;
    for( int idx=0; idx < 1000000; ++idx )
        data.push_back((int)dist(mt));

    printf("categorizing...\n");
    double st = GetTime();
    custr::category<int> cat( data.data(), data.size() );
    double et = GetTime() - st;
    printf("%g seconds\n",et);
}

void test_string(std::mt19937& mt)
{
    std::uniform_int_distribution<int> dist(32,126);
    std::vector<std::string> data;
    for( int idx=0; idx < 1000000; ++idx )
    {
        std::string str;
        for( int jdx=0; jdx < 20; ++jdx )
        {
            char ch = (char)dist(mt);
            str.append(1,ch);
        }
        data.push_back(str);
    }

    for( int i=0; i<5; ++i )
    {
        printf("categorizing...\n");
        double st = GetTime();
        custr::category<std::string> cat( data.data(), data.size() );
        double et = GetTime() - st;
        printf("%g seconds\n",et);
    }
}

int main( int argc, const char** argv )
{
    std::random_device rd;
    std::mt19937 mt(rd());

    test_int(mt);
    //test_string(mt);

    return 0;
}
