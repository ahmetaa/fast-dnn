# fast-dnn
This is a fast deep neural network library designed for DNNs used in Speech Recognition systems. But it probably can be applied to any feed forward NN. This is a runtime library, it is not designed for training DNNs.

Implementation improves the speed of the DNN calculations on CPUs using SIMD instructions, linear quantization and batch processing. Ideas are taken from Vanhoucke et-al's [Improving the speed of neural networks on CPUs] (http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37631.pdf) paper. Library contains the C++ implementation and a Java API. 

## Basic usage
First a proper DNN model file needs to be created. System uses a text DNN file and converts it to a binary form. For now, it accepts [Kaldi] (http://kaldi.sourceforge.net/dnn.html) style network files. Also, neural network input and hidden layer node sizes needs to be aligned to 4 and 16 respectively.

For converting network:

	FeedForwardNetwork network = FeedForwardNetwork.loadFromTextFile(
	  new File("dnn.txt"),
	  new File("feature_transform")
	);

First file represents the network, second represents the input transformation. Second file contains two vectors. Each input vector is transformed by adding the first vector and multiplying the second. Then padding and binartization is applied.

	network.align(4,16); // pads input to a factor of 4 and hidden layer node counts to a factor of 16
	network.saveBinary(new File("dnn.bin"));

Operations above are only required to run once and not required during runtime. Once the network is ready, it is used via *QuantizedDnn* in runtime:

	QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("dnn.bin"));
	System.out.println(dnn.inputDimension());
	float[][] input = ... input vectors as a matrix. it must match input dimension
	float[][] results = dnn.calculate(input); // output softmax result. 

## Speed
In general, this network is about a magnitude of order faster than a naive C++/Java implementation. According to my tests, it is more than 2 times faster than networks that uses BLAS (Via JBlas). BLAS uses optimization tricks and SIMD operations extensively. Once layz-batching is applied it will probably give a %30-%40 more relative speed improvement. This library allows usage of very large DNNs like the ones probably used by Google in recent years (5 2048 node hidden layers and 8000 output nodes). For small networks, speed difference may not be that important.

## Limitations
* Only tested in Ubuntu Linux x86-64 (Event then, C++ side may need to be re-compiled). 
* Works in CPU's with SSE4.2 support.
* Hiden layers must be in equal size.

## TODO
* Add Windows library
* Make it ready for multiple threaded applications.
* Fix memory leaks
* Implement Lazy-Batch operation for last layer.
* Implement PCA in Java API for network node reduction.
* Implement log and exp lookup for faster soft-max - log operations.
* Implement ReLu activation quantization network (later)
