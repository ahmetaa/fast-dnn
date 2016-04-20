cmake_minimum_required(VERSION 3.1)
project(fast-dnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -std=c++11 -funroll-loops -msse4 -fPIC -DDEBUG")
include_directories("../../include")
set(SOURCE_FILES dnn.cc jni_dnn.cc)
add_executable(dnn ${SOURCE_FILES} dnn.h float_dnn.h float_dnn.cc)
