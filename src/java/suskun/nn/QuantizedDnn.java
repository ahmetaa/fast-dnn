package suskun.nn;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

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
 * Lazy processing (Not yet implemented): In the last layer, not all outputs are required to be calculated.
 * So, only required outputs are calculated. This requires communication with the call side.
 * <p>
 */
public class QuantizedDnn {

    static {
        try {
            NativeUtils.loadLibraryFromJar("/resources/libfast-dnn.so");
        } catch (IOException e1) {
            e1.printStackTrace();
        }
    }

    int inputDimension;
    int outputDimension;

    // generates the dnn network in native code from binary network file.
    native void initialize(String fileName);

    public static QuantizedDnn loadFromFile(File dnnFile) {
        QuantizedDnn dnn = new QuantizedDnn();
        dnn.initialize(dnnFile.getAbsolutePath());
        dnn.inputDimension = dnn.inputDimension();
        dnn.outputDimension = dnn.outputDimension();
        return dnn;
    }

    public static class LazyContext {
        final QuantizedDnn dnn;
        long handle;
        final int inputVectorCount;
        int currentVectorIndex;

        public LazyContext(QuantizedDnn dnn, long handle, int inputVectorCount) {
            this.dnn = dnn;
            this.handle = handle;
            this.inputVectorCount = inputVectorCount;
        }

        public void calculateUntilOutput(float[][] input) {
            dnn.calculateUntilOutput(handle, toVector(input));
        }

        public float[] calculateForOutputNodes(byte[] activeNodesMask) {
            // flat array containing activations. Length = [nodeIndexes.length * bufferSize]
            float[] result = dnn.calculateSoftMaxForOutputs(handle, currentVectorIndex, activeNodesMask);
            currentVectorIndex++;
            return result;
        }

        public void delete() {
            dnn.deleteLazyContext(handle);
        }
    }

    public static class LazyBatchContext {
        final QuantizedDnn dnn;
        long handle;
        final int inputVectorCount;
        final int bufferSize;
        int currentBufferIndex;
        final float[][] resultBuffer;
        int currentVectorIndex;
        float[] softMaxExponential;
        int[] indexesArray;

        public LazyBatchContext(QuantizedDnn dnn, long handle, int inputVectorCount, int bufferSize) {
            this.dnn = dnn;
            this.handle = handle;
            this.inputVectorCount = inputVectorCount;
            this.bufferSize = bufferSize;
            this.resultBuffer = new float[bufferSize][dnn.outputDimension];
            this.softMaxExponential = new float[dnn.outputDimension];
            this.indexesArray = new int[dnn.outputDimension];
        }

        public void calculateUntilOutput(float[][] input) {
            dnn.calculateUntilOutput(handle, toVector(input));
        }

        public float[] calculateForOutputNodes(int inputVectorIndex, int[] nodeIndexes) {
            return dnn.calculateForOutputs(handle, inputVectorIndex, nodeIndexes);
        }

        public float[] calculateForOutputNodes(byte[] activeNodesMask) {

            // before starting, discard the previous results because new results may be written there.
            if (currentVectorIndex > 0) {
                int previousIndex = currentBufferIndex == 0 ? bufferSize - 1 : currentBufferIndex - 1;
                Arrays.fill(resultBuffer[previousIndex], 0);
            }

            final float[] current = resultBuffer[currentBufferIndex];
            int newActiveNodesCount = 0;
            for (int i = 0; i < activeNodesMask.length; i++) {
                if (activeNodesMask[i] == 0)
                    current[i] = 0;
                else if (current[i] == 0) {
                    indexesArray[newActiveNodesCount++] = i;
                }
            }
            // get the newly activated indexes.
            int[] activeIndexes = Arrays.copyOf(indexesArray, newActiveNodesCount);

            //System.out.println(newActiveNodesCount);
            if (newActiveNodesCount > 0) {
                // flat array containing activations. Length = [nodeIndexes.length * bufferSize]
                float[] activations = dnn.calculateForOutputs(handle, currentVectorIndex, activeIndexes);
                for (int i = 0; i < bufferSize; i++) {
                    float[] scores = resultBuffer[(i + currentBufferIndex) % bufferSize];
                    int k = 0;
                    for (int j = 0; j < newActiveNodesCount; j++) {
                        float activation = activations[k];
                        scores[activeIndexes[j]] = activation;
                        k++;
                    }
                }
            }
            currentBufferIndex = (currentBufferIndex + 1) % bufferSize;
            currentVectorIndex++;

            softMax(current);
            return current;
        }

        private void softMax(float[] input) {
            float total = 0;
            for (int i = 0; i < input.length; i++) {
                float v = input[i];
                // most likely more than half of the values will be zero.
                float d = v == 0 ? 1 : (float) Math.exp(v);
                //float d = v == 0 ? 1 : expApproximation(v);
                softMaxExponential[i] = d;
                total += d;
            }
            for (int i = 0; i < input.length; i++) {
                input[i] = softMaxExponential[i] / total;
            }
        }

        private float expApproximation(float input) {
            final long tmp = (long) (1512775 * input + 1072632447);
            return (float) Double.longBitsToDouble(tmp << 32);
        }
    }

    public LazyBatchContext getNewLazyBatchContext(int inputVectorCount, int batchSize) {
        long handle = getContext(inputVectorCount, batchSize);
        return new LazyBatchContext(this, handle, inputVectorCount, batchSize);
    }

    public LazyContext getNewLazyContext(int inputVectorCount) {
        long handle = getContext(inputVectorCount,8);
        return new LazyContext(this, handle, inputVectorCount);
    }

    native void deleteLazyContext(long handle);

    public native void deleteNativeDnn();

    native long getContext(int inputVectorCount, int batchSize);

    native void calculateUntilOutput(long contextHandle, float[] input);

    native float[] calculateForOutputs(long contextHandle, int inputIndex, int[] outputIndexes);

    native float[] calculateSoftMaxForOutputs(long contextHandle, int inputIndex, byte[] outputMask);

    native float[] calculate(float[] input, int inputVectorCount, int inputDimension, int batchSize);

    public native int inputDimension();

    public native int outputDimension();

    public float[][] calculate(float[][] input) {
        return calculate(input, 8);
    }

    public float[][] calculate(float[][] input, int batchSize) {
        int dimension = input[0].length;
        float[] flattened = toVector(input);
        float[] res1d = calculate(flattened, input.length, dimension, batchSize);
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
