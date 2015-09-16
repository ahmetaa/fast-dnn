package suskun.nn;

import java.io.*;
import java.util.Arrays;

public class FastNativeDnn {

    static {
        try {
            NativeUtils.loadLibraryFromJar("/resources/libfast-dnn.so");
        } catch (IOException e1) {
            e1.printStackTrace();
        }
    }

    // generates the dnn network in native code from binary network file.
    public native void initialize(String fileName);

    private native float[] calculate(float[] input, int inputVectorCount, int inputDimension, int batchSize);

    public float[][] calculate(float[][] input, int outputSize) {
        int dimension = input[0].length;
        float[] flattened = flatten(input);
        float[] res1d = calculate(flattened, input.length, dimension, 8);
        return make2d(res1d, input.length, outputSize);
    }

    private float[] flatten(float[][] arr2d) {
        int vecCount = arr2d.length;
        int dimension = arr2d[0].length;
        float[] res = new float[vecCount * dimension];
        for (int i = 0; i < vecCount; i++) {
            System.arraycopy(arr2d[i], 0, res, i * dimension, dimension);
        }
        return res;
    }

    private float[][] make2d(float[] arr, int vectorSize, int dimension) {
        float[][] res = new float[vectorSize][dimension];
        for (int i = 0; i < vectorSize; i++) {
            System.arraycopy(arr, i * dimension, res[i], 0, dimension);
        }
        return res;
    }

    public static void main(String[] args) throws IOException {
        FastNativeDnn dnn = new FastNativeDnn();
        dnn.initialize("data/dnn.aligned.model");
        float[][] input = BatchData.loadRawBinary("a",new File("data/8khz.aligned.bin")).getAsFloatMatrix();

        for (int i = 0; i < 1; i++) {
            long start = System.currentTimeMillis();
            float[][] result = dnn.calculate(input, 4046);

            for (int j = 0; j < input.length; j++) {
                float[] out = result[j];
                System.out.println(j + " " + dump(Arrays.copyOf(out, 20)));
            }

            System.out.println(System.currentTimeMillis() - start);
        }
    }

    static String dump(float[] data) {
        StringBuilder sb = new StringBuilder();
        for (float v : data) {
            sb.append(String.format("%.4f ", v));
        }
        return sb.toString();
    }

}
