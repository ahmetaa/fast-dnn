# fast-dnn
This is a fast deep neural network library designed for DNNs used in Speech Recognition systems.  
This is a runtime library designed to run on x86-64 CPUs. Training DNNs is not in the scope of this library. 

Implementation improves the speed of the DNN calculations on CPUs using SIMD instructions, linear quantization, batch processing and lazy output calculation. 
Ideas are taken from Vanhoucke et-al's [Improving the speed of neural networks on CPUs] (http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37631.pdf) paper. Library contains the C++ implementation and a Java API. 

## Basic usage
First a proper DNN model file needs to be created. System uses a text DNN file and converts it to a binary form. 
For now, it accepts [Kaldi] (http://kaldi.sourceforge.net/dnn.html) style network files. Also, neural network input and hidden layer node sizes needs to be aligned to 4 and 16 respectively.
A Java API is provided for these operations.

For converting network:

	FeedForwardNetwork network = FeedForwardNetwork.loadFromTextFile(
	  new File("dnn.txt"),
	  new File("feature_transform")
	);

First file represents the network, second represents the input transformation. Second file contains two vectors. Each input vector is transformed by adding the first vector and multiplying the second. Then padding and binarization is applied.

	network.align(4,16); // pads input to a factor of 4 and hidden layer node counts to a factor of 16
	network.saveBinary(new File("dnn.bin"));

Operations above are only required to run once and not required during runtime. Once the network is ready, it is used via *QuantizedDnn* in runtime:

	QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("dnn.bin"));
	float[][] input = ... input vectors as a matrix. it must match input dimension
	float[][] results = dnn.calculate(input); // output soft-max result.
	 
There is also a lazy output calculation option. This is task specific to Speech Recognition engines. Basically during recognition, for each input vector
not all output probabilities needs to be calculated. In average %30-50 of output probabilities are required. Therefore, output activation calculations
can be made lazily. However, user of the API must provide an output-length byte array that contains "1"'s as active outputs. Here is an example:

	QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("dnn.bin"));
	float[][] input = ... input vectors as a matrix. it must match input dimension
    QuantizedDnn.LazyContext context = dnn.getNewLazyContext(input.length);
    context.calculateUntilOutput(input);
    for (each input vector) {
         byte[] mask = ... // get a byte array from ASR system. Array length is equal to output size. 1 values in  
                           // the array represents active outputs to be calculated.
         float[] softMaxResult = context.calculateForOutputNodes(mask); // output for current input         
    }

If amount of outputs for each input is below %50 of the total outputs, lazy calculation may give around 5-10% speed increase. However, because of the JNI round trips, lazy calculation is not as effective as it should be.  

## How does it work?

The DNNs used in Automatic Speech Recognition (ASR) systems are usually very large. Especially server side applications use networks with sometimes more than 40 million parameters. In common ASR systems, for 1 seconds of speech, around 100 full network output activations needs to be calculated. This makes around 3-4 billion multiplication and sum operations for 1 second of speech.   

One idea is to use GPUs for this task. Indeed they work and they are very fast adn should be preferred for batch processin if possible. But they are not as ubiquitous as CPUs and they may not be so practical for real-time speech processing.
So, for some applications those DNNs needs to run fast in CPUs. Conventional calculation techniques becomes too slow for practical use, as stated in the paper, processing 1 second of speech takes around 4 seconds using
using naive floating point matrix multiplications. Using floating point SIMD instructions comes to mind, but that only brings down the number to around 1 seconds. This is still not good enough (Libraries like Eigen and Blas does a much better job though). 
  
Quantization comes to the rescue.
Instead of using 32 bit floating numbers for weights and sigmoid activations, 8 bit signed and unsigned numbers can be used.
32 bit values are linearly quantized to 8 bit numbers. This is possible because the weights are usually lie 
between -1 and 1 (Actually a Gaussian curve with small variance), and sigmoid activation values are always between 0 and 1. 
Then, using a special SIMD operation, 16 signed integers
are multiplied with 16 unsigned integers and results are summed nicely with a couple of SIMD instructions. There are some exceptions and caveats but long story short, this reduces the time required for processing 1 second 
of speech to around 0.25-0.3 seconds. Which is acceptable even for the runtime systems. For details, please refer to the paper.

## Actual Speed
In general, this network is about a magnitude of order faster than a naive C++/Java implementation. According to my tests, it is about 2 times faster than networks that uses BLAS (Via JBlas). Speed difference is lower when compared to C++ Blas, it is (around %30 faster) [https://plus.google.com/+AhmetAAkın/posts/RQwqZh9GyPg]. When using Java API, it may take a small hit because of the JNI. This library allows usage of very large DNNs (such as 7 2048 node hidden layers and 8000 output nodes). But for small networks, speed difference may not be significant.

## Limitations
* Only tested in Ubuntu Linux x86-64 (Event then, C++ side may need to be re-compiled). 
* Works in CPU's with SSE4.2 support.
* Hidden layers must be in equal size.
* Hidden layer activations are Sigmoid and output activations are SoftMax functions.

## Alternatives
To my knowledge, there is no open source implementation of the paper mentioned above. However, recently a low precision hand tuned matrix multiplication library is open sourced by Google [gemlowp] (https://github.com/google/gemmlowp). Library seems to be developed for DNNs to be used in mobile applications but it also has x86 support.

## TODO
* Add Windows library
* Implement PCA in Java API for network node reduction.
* Implement log and exp approximations for more speed.
* Implement ReLu activation quantization network (later)
