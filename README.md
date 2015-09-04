# fast-dnn
This is a fast deep neural network library designed for DNNs used in Speech Recognition systems. But it can be applied to any feed forward NN.
It improves the speed of the DNN calculations on CPU using SIMD instructions, linear quantization and batch processing. Ideas are taken from Vanhoucke et-al's [Improving the speed of neural networks on CPUs] (http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37631.pdf) paper. Library contains the C++ implementation and a Java access API

## Current Limitations
* Not yet usable.
* Initially it will only work for Linux x86-64 systems.
* Node counts must be aligned to 16 
* Hiden layers must be in equal size.
* Only accepts a special binary file for Dnn construction
* No network Training (Not planned)

## TODO
* Finish JNI api. 
* Add documentation and provide a Java api for DNN file generation
* Add Windows library
* Implement Lazy-Batch operation for last layer.
* Implement PCA in Java API for network node reduction.
* Implement exp lookup for faster soft-max operation.
* Implement ReLu activation quantization network (later)
