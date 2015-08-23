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

    public native void initialize(String fileName);

    public static void main(String[] args) {
        new FastNativeDnn().initialize("hello!");
    }


}
