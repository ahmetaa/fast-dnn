cmake_minimum_required(VERSION 3.1)
project(dnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -mtune=native")
include_directories("/usr/lib/jvm/java-8-oracle/include"  "/usr/lib/jvm/java-8-oracle/include/linux")
set(SOURCE_FILES dnn.cpp fast_dnn.cpp fast_dnn.h handle.h)
add_executable(dnn ${SOURCE_FILES} dnn.h float_dnn.h float_dnn.cpp)