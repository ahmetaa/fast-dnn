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

    public static float[][] loadInputData(File file) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            int frameCount = dis.readInt();
            int dimension = dis.readInt();
            float[][] data = new float[frameCount][];
            for (int i = 0; i < frameCount; i++) {
                data[i] = new float[dimension];
                for (int j = 0; j < dimension; j++) {
                    data[i][j] = dis.readFloat();
                }
            }
            return data;
        }
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
        dnn.initialize("/home/afsina/data/dnn-5-1024/dnn.tv.model");
        float[][] input = loadInputData(new File("/home/afsina/projects/suskun/feats"));

        for (int i = 0; i < 5; i++) {
            long start = System.currentTimeMillis();
            float[][] result = dnn.calculate(input, 4055);

            for (int j = 0; j < input.length; j++) {
                float[] out = result[j];
                System.out.println("output " + j + " = " + Arrays.toString(Arrays.copyOf(out, 30)));
            }

            System.out.println(System.currentTimeMillis() - start);
        }
    }

}
