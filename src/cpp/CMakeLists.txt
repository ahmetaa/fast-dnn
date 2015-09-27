cmake_minimum_required(VERSION 3.1)
project(fast-dnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -mtune=native -DDEBUG")
include_directories("../../include")
set(SOURCE_FILES dnn.cc jni_dnn.cc)
add_executable(dnn ${SOURCE_FILES} dnn.h float_dnn.h float_dnn.cc)
