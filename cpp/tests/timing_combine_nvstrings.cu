
#include <vector>
#include <string>
#include <sys/time.h>
#include <unistd.h>

#include <nvstrings/NVStrings.h>

// nvcc -w -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_70,code=sm_70 timing_combine_nvstrings.cu -I/usr/local/cuda/include -I/cudf/cpp/include -I/rmm/include -L/cudf/cpp/build -L/rmm/build -lrmm -lNVStrings -o timing_combine_nvstrings

std::vector<const char*> hstrs1{ "TUVWXYZ", "1234567890", "abcdéfgij", nullptr, "" };
std::vector<const char*> hstrs2{ nullptr, "ABC", "","9=4+5", "éa" };

double GetTime()
{
	timeval tv;
	gettimeofday( &tv, NULL );
	return (double)(tv.tv_sec*1000000+tv.tv_usec)/1000000.0;
}


int main(int argc, const char** argv)
{

    // create large string arrays
    std::vector<const char*> strs1, strs2;
    for( int idx=0; idx < 1000000; ++idx )
    {
        strs1.push_back(hstrs1[idx % hstrs1.size()]);
        strs2.push_back(hstrs2[idx % hstrs2.size()]);
    }
    printf("build strings\n");
    double st_c1 = GetTime();
    NVStrings* strings1 = NVStrings::create_from_array(strs1.data(), strs1.size());
    double et_c1 = GetTime();
    printf("strings1(%d): (%ld bytes)\n", strings1->size(), strings1->memsize());
    strings1->print(0,10);
    printf("%g seconds\n",(et_c1-st_c1));

    double st_c2 = GetTime();
    NVStrings* strings2 = NVStrings::create_from_array(strs2.data(), strs2.size());
    double et_c2 = GetTime();
    printf("strings2(%d): (%ld bytes)\n", strings2->size(), strings2->memsize());
    strings2->print(0,10);
    printf("%g seconds\n",(et_c2-st_c2));

    double st_cat = GetTime();
    auto results = strings1->cat(strings2,":");
    double et_cat = GetTime();
    printf("s1.cat(s2,':'):\n");
    results->print(0,10);
    printf("%g seconds\n",(et_cat-st_cat));

    return 0;
}
