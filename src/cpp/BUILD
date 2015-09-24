# Bazel build file.

# To build library, run from workspace root:
# bazel build src/cpp:dnn
cc_library(
    name = "dnn",
    srcs = ["dnn.cc"],
    hdrs = ["dnn.h"],
    copts = [
        "-mtune=native",
        "-march=native",
        "-std=c++11",
        "-funroll-loops",
        "-msse4",
        "-fPIC",
    ]
)
