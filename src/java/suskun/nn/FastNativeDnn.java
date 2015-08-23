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

    private long contextHandle;

    // generates the dnn network in native code from binary network file.
    public native void initialize(String fileName);

    public native long getContext();


    public static void main(String[] args) {
        new FastNativeDnn().initialize("/home/afsina/data/dnn-5-1024/dnn.model.le");
    }


}
