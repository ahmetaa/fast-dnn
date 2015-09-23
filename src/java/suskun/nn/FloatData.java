package suskun.nn;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;

/**
 * A generic class for carrying a float array and an index value attached to it.
 */
class FloatData {

    private float[] data;
    public final int index;

    public static final FloatData EMPTY = new FloatData(new float[0], 0);

    public FloatData(float[] data, int index) {
        this.index = index;
        this.data = data;
    }

    public FloatData(float[] data) {
        this(data, 0);
    }

    public float[] getCopyOfData() {
        return data.clone();
    }

    /**
     * Create a copy of this and replaces the data with newData
     *
     * @param newData new Data to use in the copy.
     * @return a copy of this FloatData but with newData
     */
    public FloatData copy(float[] newData) {
        return new FloatData(newData, index);
    }

    /**
     * @param dataToAppend data to append
     * @return A new FloatData instance by appending the input data to this FrameData
     */
    public FloatData append(float[] dataToAppend) {
        float[] newData = Arrays.copyOf(data, data.length + dataToAppend.length);
        System.arraycopy(dataToAppend, 0, newData, data.length, dataToAppend.length);
        return new FloatData(newData, index);
    }

    public void replaceData(float[] data) {
        this.data = data;
    }

    public String toString() {
        return index + " " + format(10, 5, " ", data);
    }

    public String toString(int amount) {
        return index + " " + format(10, 5, " ", Arrays.copyOf(data, amount));
    }

    public int size() {
        return data.length;
    }

    public float[] getData() {
        return data;
    }

    public void serializeToBinaryStream(DataOutputStream dos) throws IOException {
        dos.writeInt(this.index);
        dos.writeInt(size());
        FeedForwardNetwork.serializeRaw(dos, this.data);
    }

    /**
     * Formats a float array as string using English Locale.
     */
    public static String format(int rightPad, int fractionDigits, String delimiter, float... input) {
        StringBuilder sb = new StringBuilder();
        String formatStr = "%." + fractionDigits + "f";
        int i = 0;
        for (float v : input) {
            String num = String.format(formatStr, v);
            sb.append(String.format(Locale.ENGLISH, "%-" + rightPad + "s", num));
            if (i++ < input.length - 1) sb.append(delimiter);
        }
        return sb.toString().trim();
    }

    public static float[] alignTo(float[] input, int alignment) {
        int dimension = input.length;

        int padded = alignedSize(dimension, alignment);
        if (padded==dimension) {
            return input;
        }
        return Arrays.copyOf(input, padded);
    }

    public static int alignedSize(int size, int alignment) {
        if (size % alignment == 0) {
            return size;
        }
        return size + alignment - (size % alignment);
    }

}
