package suskun.nn;

import java.io.IOException;

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

    private native float[] calculate(float[] input,int inputVectorCount, int inputDimension, int batchSize);

    public float[][] calculate(float[][] input) {
        int dimension = input[0].length;
        float[] flattened = new float[input.length * dimension];
        for (float[] floats : input) {


        }
       // return calculate(flattened, input.length, dimension, 8);
        return null;
    }


    public static void main(String[] args) {
        new FastNativeDnn().initialize("/home/afsina/data/dnn-5-1024/dnn.model.le");
    }

}
