# fast-dnn
This is a fast deep neural network library designed for DNNs used in Speech Recognition systems. But it probably can be applied to any feed forward NN.
It improves the speed of the DNN calculations on CPU using SIMD instructions, linear quantization and batch processing. Ideas are taken from Vanhoucke et-al's [Improving the speed of neural networks on CPUs] (http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37631.pdf) paper. Library contains the C++ implementation and a Java API.

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

Operations below are offline. Only one it needs to be run. For runtime:

	QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("dnn.bin"));
	System.out.println(dnn.inputDimension());
	float[][] input = ... input vectors as a matrix. it must match input dimension
	float[][] nativeResult = dnn.calculate(input); // output softmax result. 

## Limitations
* Only tested in Ubuntu Linux x86-64. 
* Works in CPU's with SSE4.2 support.
* Hiden layers must be in equal size.
* No network Training (Not planned)

## TODO
* Add Windows library
* Make it ready for multiple threaded applications.
* Fix memory leaks
* Implement Lazy-Batch operation for last layer.
* Implement PCA in Java API for network node reduction.
* Implement exp lookup for faster soft-max operation.
* Implement ReLu activation quantization network (later)
