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

    public static void generateAlignedInput(int vectorCount) throws IOException {
        BatchData data = BatchData.loadFromText(new File("data/16khz"));
        data.alignDimension(4);
        data.serializeDataMatrix(new File("data/16khz.bin"), vectorCount);
    }

    public static void calculateError() throws IOException {

        QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("data/dnn.tv.model"));
        float[][] input = BatchData.loadRawBinary("a", new File("data/16khz.bin")).getAsFloatMatrix();
        long start = System.currentTimeMillis();
        float[][] nativeResult = dnn.calculate(input);
        System.out.println("Native calculated in: " + (System.currentTimeMillis() - start));

        FeedForwardNetwork n = FeedForwardNetwork.loadFromBinary(new File("data/dnn.tv.model"));
        BatchData b = BatchData.loadRawBinary("a", new File("data/16khz.bin"));
        start = System.currentTimeMillis();
        List<FloatData> result = n.calculate(b.getData());
        System.out.println("Java calculated in: " + (System.currentTimeMillis() - start));

        float[] dif = new float[n.outputLayer.outputDimension];

        for (int i = 0; i < nativeResult.length; i++) {
            float[] q = nativeResult[i];
            float[] r = result.get(i).getData();
            for (int j = 0; j < r.length; j++) {
                dif[j] += Math.abs(q[j] - r[j]);
            }
        }
/*
        for (float v : dif) {
            if (v > 0.1)
                System.out.println(v);
        }*/
    }

    static void lazyTest() throws IOException {
        QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("data/dnn.tv.model"));
        float[][] input = BatchData.loadRawBinary("a", new File("data/16khz.bin")).getAsFloatMatrix();
        QuantizedDnn.Context context = dnn.getNewContext(input.length, 8);
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

    static void lazyBatchEmulation() throws IOException {
        QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("data/dnn.tv.model"));
        float[][] input = BatchData.loadRawBinary("a", new File("data/16khz.bin")).getAsFloatMatrix();
        byte[][] masks = generateMasks(
                input.length,
                dnn.outputDimension,
                (int) (dnn.outputDimension * 0.4),
                (int) (dnn.outputDimension * 0.02));
        long start = System.currentTimeMillis();
        QuantizedDnn.Context context = dnn.getNewContext(input.length, 10);
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
            setValueRandom(averageNewActiveNode, res[i], (byte) 0);
            setValueRandom(averageNewActiveNode, res[i], (byte) 1);
        }
        return res;
    }

    static int countOfOnes(byte[] values) {
        int count = 0;
        for (byte val : values) {
            if(val==1) count++;
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
        //generateAlignedInput(100);
        calculateError();
        //lazyTest();
        lazyBatchEmulation();
    }
}
