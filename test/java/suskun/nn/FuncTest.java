package suskun.nn;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class FuncTest {
    public static void generateNN() throws IOException {
        FeedForwardNetwork network = FeedForwardNetwork.loadFromTextFile(
                new File("nnet_iter18.txt"),
                new File("final.feature_transform")
        );
        network.align(4, 16);
        network.saveBinary(new File("data/dnn.tv.model"));
    }


    public static void extendNetwork(File input, File output) throws IOException {
        FeedForwardNetwork network = FeedForwardNetwork.loadFromBinary(input);
        System.out.println(network.info());
        network.extend(2048, 8000);
        network.align(4, 16);
        System.out.println(network.info());
        network.saveBinary(output);
    }


    public static void generateAlignedInput(int vectorCount) throws IOException {
        BatchData data = BatchData.loadFromText(new File("data/16khz"));
        System.out.println("Input Vector Count   = " + data.vectorCount());
        System.out.println("Input Data Dimension = " + data.get(0).size());
        data.alignDimension(4);
        System.out.println("Aligned Input Data Dimension = " + data.get(0).size());
        data.serializeDataMatrix(new File("data/16khz.bin"), vectorCount);
    }

    public static float[][] runQuantized() throws IOException {

        //QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("data/dnn.tv.model"));
        QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("data/dnn.extended.tv.model"));
        float[][] input = BatchData.loadRawBinary("a", new File("data/16khz.bin")).getAsFloatMatrix();
        float[][] nativeResult = null;
        for (int i = 0; i < 1; i++) {
            long start = System.currentTimeMillis();
            nativeResult = dnn.calculate(input, 10);
            System.out.println(i + " Native calculated in: " + (System.currentTimeMillis() - start));
        }
        return nativeResult;
    }


    public static void diff(FeedForwardNetwork n, List<FloatData> result, float[][] nativeResult) {
        float[] dif = new float[n.outputLayer.outputDimension];

        for (int i = 0; i < nativeResult.length; i++) {
            float[] q = nativeResult[i];
            float[] r = result.get(i).getData();
            for (int j = 0; j < r.length; j++) {
                dif[j] += Math.abs(q[j] - r[j]);
            }
        }

        for (float v : dif) {
            if (v > 0.1)
                System.out.println(v);
        }
    }

    static List<FloatData> runNaive() throws IOException {
        FeedForwardNetwork n = FeedForwardNetwork.loadFromBinary(new File("data/dnn.extended.tv.model"));
        BatchData b = BatchData.loadRawBinary("a", new File("data/16khz.bin"));
        long start = System.currentTimeMillis();
        List<FloatData> result = n.calculate(b.getData());
        System.out.println("Java calculated in: " + (System.currentTimeMillis() - start));
        return result;
    }

    static void lazyTest() throws IOException {
        QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("data/dnn.tv.model"));
        float[][] input = BatchData.loadRawBinary("a", new File("data/16khz.bin")).getAsFloatMatrix();
        QuantizedDnn.LazyBatchContext context = dnn.getNewLazyBatchContext(input.length, 10);
        context.calculateUntilOutput(input);

        int[] outputIndexes = new int[50];
        for (int i = 0; i < 50; i++) {
            outputIndexes[i] = i * 2;
        }
        for (int i = 0; i < input.length; i++) {
            float[] result = context.calculateForOutputNodes(i, outputIndexes);
            System.out.println(Arrays.toString(result));
        }
    }

    static void lazyEmulation() throws IOException {
        QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("data/dnn.extended.tv.model"));
        float[][] input = BatchData.loadRawBinary("a", new File("data/16khz.bin")).getAsFloatMatrix();
        byte[][] masks = generateMasks(
                input.length,
                dnn.outputDimension,
                (int) (dnn.outputDimension * 0.45),
                (int) (dnn.outputDimension * 0.02));
        QuantizedDnn.LazyContext context = dnn.getNewLazyContext(input.length);
        long start = System.currentTimeMillis();
        context.calculateUntilOutput(input);
        float total = 0;
        for (int i = 0; i < input.length; i++) {
            byte[] mask = masks[i];
            float[] result = context.calculateForOutputNodes(mask);
            total += result[0];
        }
        System.out.println("Lazy calculated in: " + (System.currentTimeMillis() - start));
    }

    static void lazyBatchEmulation() throws IOException {
        QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("data/dnn.extended.tv.model"));
        float[][] input = BatchData.loadRawBinary("a", new File("data/16khz.bin")).getAsFloatMatrix();
        byte[][] masks = generateMasks(
                input.length,
                dnn.outputDimension,
                (int) (dnn.outputDimension * 0.4),
                (int) (dnn.outputDimension * 0.01));
        QuantizedDnn.LazyBatchContext context = dnn.getNewLazyBatchContext(input.length, 4);
        long start = System.currentTimeMillis();
        context.calculateUntilOutput(input);
        float total = 0;
        for (int i = 0; i < input.length; i++) {
            byte[] mask = masks[i];
            //System.out.println(countOfOnes(mask));
            float[] result = context.calculateForOutputNodes(mask);
            total += result[0];
            //System.out.println(Arrays.toString(result));
        }
        System.out.println("Lazy-Batch calculated in: " + (System.currentTimeMillis() - start));

    }

    static byte[][] generateMasks(int count, int dimension, int averageActiveNode, int averageNewActiveNode) {
        byte[][] res = new byte[count][dimension];
        res[0] = new byte[dimension];

        setValueRandom(averageActiveNode, res[0], (byte) 1);

        for (int i = 1; i < count; i++) {
            res[i] = res[i - 1].clone();
            setValueRandom(averageNewActiveNode, res[i], (byte) 1);
            setValueRandom(averageNewActiveNode, res[i], (byte) 0);
        }
        return res;
    }

    static int countOfOnes(byte[] values) {
        int count = 0;
        for (byte val : values) {
            if (val == 1) count++;
        }
        return count;
    }


    static private void setValueRandom(int nodeCount, byte[] b, byte val) {
        Random r = new Random();
        int cnt = 0;
        while (cnt < nodeCount) {
            int k = r.nextInt(b.length);
            if (b[k] != val) {
                b[k] = val;
                cnt++;
            }
        }
    }

    static String dump(float[] data) {
        StringBuilder sb = new StringBuilder();
        for (float v : data) {
            sb.append(String.format("%.4f ", v));
        }
        return sb.toString();
    }

    public static void main(String[] args) throws IOException {
        //generateNN();
        //extendNetwork(new File("data/dnn.tv.model"), new File("data/dnn.extended.tv.model"));
        //generateAlignedInput(100);
        runQuantized();
        //runNaive();
        //lazyTest();
        lazyEmulation();
        lazyBatchEmulation();
    }
}
