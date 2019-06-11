cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)
find_package(CUDA REQUIRED)

#
set(LIB_CATEGORY_SOURCES
#src/category.cpp
src/category.cu
)

set(LIB_CATEGORY_HEADERS 
include/category.h
)

set (INCLUDE_DIR
${PYTHON_INCLUDE_DIRS}
.
)

SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\$ORIGIN")

add_definitions(-std=c++14)
add_compile_options(-fPIC)

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 --expt-extended-lambda -gencode arch=compute_70,code=sm_70 -Xcompiler -fPIC)

cuda_add_library (cucategory SHARED ${LIB_CATEGORY_SOURCES} ${LIB_CATEGORY_HEADERS})

cuda_add_library (pynicucategory SHARED pycategory.cpp)

set_target_properties(pynicucategory PROPERTIES SUFFIX ".so")
set_target_properties(pynicucategory PROPERTIES PREFIX "")

target_link_libraries(pynicucategory ${PYTHON_LIBRARIES} cucategory)

install(TARGETS cucategory LIBRARY DESTINATION lib)
