# fast-dnn
This is a fast deep neural network library. This implementation is specifically targeted for DNN's used in Speech recognition systems.
Library does not provide training. It improves DNN calculations on CPU using special SIMD instructions, linear quantization and batch processing. 
Ideas are taken from Vanhoucke et-al's 'Improving the speed of neural networks on CPUs' paper.


Library contains the C++ implementation and a Java access API. So far it only works for Linux x86-64 systems.

## TODO
* Finish JNI api. 
* Add Windows library
* Implement Lazy-Batch operation for last layer.
* Implement PCA in Java API for network reduction.
* Implement exp lookup for faster soft-max operation.
* Implement ReLu activation quantization network (later)