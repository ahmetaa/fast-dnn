package suskun.nn;

import java.io.File;
import java.io.IOException;

/**
 * This is an optimized feed-forward neural network implementation that uses native code.
 * Optimizations are based on
 * - Vanhoucke et al's "Improving the speed of neural networks on CPUs" [2011]
 * <p>
 * Idea is to linearly quantize the weight and activation values to 8 bit values and use special SIMD instructions
 * that uses 8 bit arguments. This not only allows more parameters to calculate in parallel with SIMD instructions
 * but also improves memory throughput greatly.
 * However, input layer weights and all bias values are not quantized.
 * Quantization is applied to each layer separately by taking the maximum weight magnitude and quantize to [-128, 127]
 * <p>
 * Batch processing. Instead of calculating one input vector at a time, multiple vectors are calculated.
 * <p>
 * Lazy processing: In the last layer, not all outputs are required to be calculated.
 * So, only required outputs are calculated. This requires communication with the call side.
 * <p>
 */
public class QuantizedDnn {

    static {
        try {
            String osName = System.getProperty("os.name");
            if (osName.contains("Windows")) {
                NativeUtils.loadLibraryFromJar("/resources/fast-dnn.dll");
            } else if (osName.contains("Linux")) {
                NativeUtils.loadLibraryFromJar("/resources/libfast-dnn.so");
            } else {
                throw new IllegalStateException("There is no library for OS = " + osName);
            }
        } catch (IOException e1) {
            e1.printStackTrace();
        }
    }

    private int inputDimension;
    private int outputDimension;
    private long nativeDnnHandle;


    // generates the dnn network in native code from binary network file.
    native long initialize(String fileName, float weightCutOffValue);

    /**
     * Creates a quantized dnn with a certain weight cut-off value.
     * weightCutOffValue is a positive number, usually between 1 and 5.
     * When linearly quantizing weights in hidden units, the values outside
     * [-weightCutOffValue, weightCutOffValue] are trimmed.
     */
    public static QuantizedDnn loadFromFile(File dnnFile, float weightCutOffValue) {
        if(weightCutOffValue<=0){
            throw new IllegalArgumentException("Weight cut off value must be positive. But it is " + weightCutOffValue);
        }
        QuantizedDnn dnn = new QuantizedDnn();
        dnn.nativeDnnHandle = dnn.initialize(dnnFile.getAbsolutePath(), weightCutOffValue);
        dnn.inputDimension = dnn.inputDimension();
        dnn.outputDimension = dnn.outputDimension();
        return dnn;
    }

    /**
     * Creates a quantized dnn with weight cut-off value = 3
     */
    public static QuantizedDnn loadFromFile(File dnnFile) {
        return loadFromFile(dnnFile, 3f);
    }

    public static class LazyContext {
        final QuantizedDnn dnn;
        long handle;
        final int inputVectorCount;
        int currentVectorIndex;

        private LazyContext(QuantizedDnn dnn, long handle, int inputVectorCount) {
            this.dnn = dnn;
            this.handle = handle;
            this.inputVectorCount = inputVectorCount;
        }

        public void calculateUntilOutput(float[][] input) {
            dnn.calculateUntilOutput(handle, toVector(input));
        }

        public float[] calculateForOutputNodes(byte[] activeNodesMask) {
            // flat array containing activations. Length = [nodeIndexes.length * bufferSize]
            float[] result = dnn.calculateLazy(handle, currentVectorIndex, activeNodesMask);
            currentVectorIndex++;
            return result;
        }

        public void delete() {
            dnn.deleteLazyContext(handle);
        }
    }

    public LazyContext getNewLazyContext(int inputVectorCount) {
        return getNewLazyContext(inputVectorCount, 8);
    }

    public LazyContext getNewLazyContext(int inputVectorCount, int batchSize) {
        long handle = getContext(nativeDnnHandle, inputVectorCount, batchSize);
        return new LazyContext(this, handle, inputVectorCount);
    }

    private native void deleteLazyContext(long dnnHandle);

    private native void delete(long dnnHandle);

    private native long getContext(long dnnHandle, int inputVectorCount, int batchSize);

    private native void calculateUntilOutput(long contextHandle, float[] input);

    private native float[] calculateLazy(long contextHandle, int inputIndex, byte[] outputMask);

    private native float[] calculate(long dnnHandle, float[] input, int inputVectorCount, int inputDimension, int batchSize);

    private native int inputDimension(long dnnHandle);

    private native int outputDimension(long dnnHandle);

    private native int layerDimension(long dnnHandle, int layerIndex);

    private native int layerCount(long dnnHandle);

    public int inputDimension() {
        return inputDimension(nativeDnnHandle);
    }

    public int outputDimension() {
        return outputDimension(nativeDnnHandle);
    }

    public void delete() {
        delete(nativeDnnHandle);
    }

    public int layerDimension(int layerIndex) {
        return layerDimension(nativeDnnHandle, layerIndex);
    }

    public int layerCount() {
        return layerCount(nativeDnnHandle);
    }

    public float[][] calculate(float[][] input) {
        return calculate(input, 10);
    }

    public float[][] calculate(float[][] input, int batchSize) {
        if (input.length == 0) {
            return new float[0][0];
        }
        if (input[0].length != inputDimension) {
            throw new IllegalArgumentException(
                    String.format("Input vector size %d must be equal with network input size %d",
                            input[0].length, inputDimension));
        }

        int dimension = input[0].length;
        float[] flattened = toVector(input);
        float[] res1d = calculate(nativeDnnHandle, flattened, input.length, dimension, batchSize);
        return toMatrix(res1d, input.length, outputDimension);
    }


    private static float[] toVector(float[][] arr2d) {
        int vecCount = arr2d.length;
        int dimension = arr2d[0].length;
        float[] res = new float[vecCount * dimension];
        for (int i = 0; i < vecCount; i++) {
            System.arraycopy(arr2d[i], 0, res, i * dimension, dimension);
        }
        return res;
    }

    private static float[][] toMatrix(float[] arr, int vectorSize, int dimension) {
        float[][] res = new float[vectorSize][dimension];
        for (int i = 0; i < vectorSize; i++) {
            System.arraycopy(arr, i * dimension, res[i], 0, dimension);
        }
        return res;
    }

}