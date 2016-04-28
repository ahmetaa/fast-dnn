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


    public static void generateAlignedInput(int vectorCount, File outFile) throws IOException {
        BatchData data = BatchData.loadFromText(new File("data/16khz"));
        System.out.println("Input Vector Count   = " + data.vectorCount());
        System.out.println("Input Data Dimension = " + data.get(0).size());
        data.alignDimension(4);
        System.out.println("Aligned Input Data Dimension = " + data.get(0).size());
        data.serializeDataMatrix(outFile, vectorCount);
    }

    public static float[][] runQuantized(
            File model,
            File inputFile,
            int iterationCount) throws IOException {

        QuantizedDnn dnn = QuantizedDnn.loadFromFile(model);
        float[][] input = BatchData.loadRawBinary("a", inputFile).getAsFloatMatrix();
        float[][] nativeResult = null;
        for (int i = 0; i < iterationCount; i++) {
            long start = System.currentTimeMillis();
            nativeResult = dnn.calculate(input, 10);
            System.out.println(i + " Native calculated in: " + (System.currentTimeMillis() - start));
        }
        System.out.println("-------------");
        dnn.delete();
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

    static List<FloatData> runNaive(
            File model,
            File inputFile,
            int iterationCount) throws IOException {
        FeedForwardNetwork n = FeedForwardNetwork.loadFromBinary(model);
        BatchData b = BatchData.loadRawBinary("a", inputFile);
        List<FloatData> result = null;
        for (int i = 0; i < iterationCount; i++) {
            long start = System.currentTimeMillis();
            result = n.calculate(b.getData());
            System.out.println("Java naive calculated in: " + (System.currentTimeMillis() - start));
        }
        System.out.println("-------------");
        return result;
    }

    static float[][] lazyEmulation(
            File model,
            File inputFile,
            int iterationCount,
            float averageOutputNodeRatio) throws IOException {
        QuantizedDnn dnn = QuantizedDnn.loadFromFile(model);
        float[][] input = BatchData.loadRawBinary("a", inputFile).getAsFloatMatrix();
        byte[][] masks = generateMasks(
                input.length,
                dnn.outputDimension(),
                (int) (dnn.outputDimension() * averageOutputNodeRatio),
                (int) (dnn.outputDimension() * 0.03));
        float[][] result = new float[input.length][];
        for (int j = 0; j < iterationCount; j++) {
            long start = System.currentTimeMillis();
            QuantizedDnn.LazyContext context = dnn.getNewLazyContext(input.length);
            context.calculateUntilOutput(input);
            for (int i = 0; i < input.length; i++) {
                byte[] mask = masks[i];
                result[i] = context.calculateForOutputNodes(mask);
            }
            context.delete();
            System.out.println("Lazy calculated in: " + (System.currentTimeMillis() - start));
        }
        dnn.delete();
        System.out.println("-------------");
        return result;
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

    public static void main(String[] args) throws Exception {
        //generateNN();
        //extendNetwork(new File("data/dnn.tv.model"), new File("data/dnn.extended.tv.model"));
        //generateAlignedInput(1000, new File("data/16khz-10s.bin"));
        File model = new File("data/dnn.extended.tv.model");
        File input = new File("data/16khz.bin");
        int iterationCount = 5;

        //runNaive(model, input, 1);
        runQuantized(model, input, iterationCount);
        lazyEmulation(model, input, iterationCount, 0.40f);
    }
}
