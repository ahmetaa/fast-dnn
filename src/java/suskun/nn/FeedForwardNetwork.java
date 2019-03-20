package suskun.nn;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Used for neural network file plumbing and verification.
 */
public class FeedForwardNetwork {
    // layers contain all weight and bias values.
    private List<Layer> layers;

    // for convenience we have separate Layer object references for output and first hidden layer.
    public final Layer outputLayer;
    public final Layer firstLayer;

    // transpose of shift and scale vectors.
    // Shift matrix is added to input matrix and result is multiplied with scale matrix.
    private float[] shiftVector;
    private float[] scaleVector;

    public FeedForwardNetwork(List<Layer> layers, float[] shiftVector, float[] scaleVector) {
        this.layers = layers;
        this.outputLayer = layers.get(layers.size() - 1);
        this.firstLayer = layers.get(0);
        this.shiftVector = shiftVector;
        this.scaleVector = scaleVector;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public int layerDimension(int layerIndex) {
        return getLayer(layerIndex).outputDimension;
    }

    public Layer getLayer(int layerIndex) {
        if (layerIndex < 0 || layerIndex >= layers.size())
            throw new IllegalArgumentException("Illegal layer index " + layerIndex);
        return layers.get(layerIndex);
    }

    public void align(int inputAlignment, int hiddenLayerAlignment) {
        shiftVector = FloatData.alignTo(shiftVector, inputAlignment);
        scaleVector = FloatData.alignTo(scaleVector, inputAlignment);
        firstLayer.align(inputAlignment, hiddenLayerAlignment);
        for (int i = 1; i < layers.size() - 1; i++) {
            layers.get(i).align(hiddenLayerAlignment, hiddenLayerAlignment);
        }
        outputLayer.align(hiddenLayerAlignment, 1);
    }

    public void extend(int hiddenLayerNodecount, int outputcount) {
        firstLayer.extend(firstLayer.inputDimension, hiddenLayerNodecount);
        for (int i = 1; i < layers.size() - 1; i++) {
            layers.get(i).extend(hiddenLayerNodecount, hiddenLayerNodecount);
        }
        outputLayer.align(hiddenLayerNodecount, outputcount);
    }

    public String info() {
        StringBuilder builder = new StringBuilder();
        int i = 0;
        for (Layer layer : layers) {
            builder.append(String.format("Layer %d neuron count = %d\n", i, layer.inputDimension));
            i++;
        }
        builder.append(String.format("Output count         = %d\n", outputLayer.outputDimension));
        return builder.toString();
    }

    public static Pattern FEATURE_LINES_PATTERN = Pattern.compile("(?:\\[)(.+?)(?:\\])", Pattern.DOTALL | Pattern.MULTILINE);

    /**
     * Generate a JBlasDnn instance from text file
     *
     * @throws IOException
     */
    public static FeedForwardNetwork loadFromTextFile(File networkFile, File transformationFile) throws IOException {
        List<Layer> layers = loadLayersFromTextFile(networkFile);

        List<String> lines = Files.readAllLines(transformationFile.toPath(), StandardCharsets.UTF_8);
        String wholeThing = String.join(" ", lines);
        List<String> featureBlocks = new ArrayList<>();
        Matcher m = FEATURE_LINES_PATTERN.matcher(wholeThing);
        while (m.find()) {
            featureBlocks.add(m.group(1).trim());
        }

        // if there is <Splice> block, omit it.
        if (featureBlocks.size() == 3) {
            featureBlocks = new ArrayList<>(featureBlocks.subList(1, featureBlocks.size()));
        }

        if (featureBlocks.size() != 2) {
            throw new IllegalStateException("Unexpected feature transformation vector size : "
                    + featureBlocks.size());
        }
        float[] shiftVector = fromString(featureBlocks.get(0));
        float[] scaleVector = fromString(featureBlocks.get(1));

        int inputDimension = layers.get(0).inputDimension;
        if (shiftVector.length != inputDimension) {
            throw new IllegalStateException("Shift transformation vector size " + shiftVector.length +
                    " is not same as input dimension " + inputDimension);
        }
        if (scaleVector.length != inputDimension) {
            throw new IllegalStateException("Scale transformation vector size " + scaleVector.length +
                    " is not same as input dimension " + inputDimension);
        }
        return new FeedForwardNetwork(layers, shiftVector, scaleVector);
    }

    public void shiftAndScale(List<FloatData> data) {
        for (FloatData floatData : data) {
            float[] d = floatData.getData();
            for (int i = 0; i < d.length; i++) {
                d[i] = (d[i] + shiftVector[i]) * scaleVector[i];
            }
        }
    }

    /**
     * Calculates layer activations for multiple vectors.
     */
    public List<FloatData> calculate(List<FloatData> inputVectors) {
        shiftAndScale(inputVectors);
        List<FloatData> input = inputVectors;

        for (Layer layer : layers) {
            List<FloatData> activations = layer.activations(input);
            if (layer == outputLayer) {
                softMax(activations);
                return activations;
            } else {
                sigmoid(activations);
            }
            input = activations;
        }
        throw new IllegalStateException("Output layer cannot be reached!");
    }

    public static float[] fromString(String str) {
        String[] tokens = str.split("[ ]+");
        float[] result = new float[tokens.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = Float.parseFloat(tokens[i]);
        }
        return result;
    }

    private static List<Layer> loadLayersFromTextFile(File networkFile) throws IOException {
        List<Layer> layers = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(networkFile.toPath(), StandardCharsets.UTF_8)) {

            String line;
            int nodeCount = -1;
            int inputCount = -1;

            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.length() == 0)
                    continue;

                if (line.startsWith("<AffineTransform>")) {
                    String layerCountInfo = line.substring(line.indexOf('>') + 1).trim();
                    String[] split = layerCountInfo.split("[ ]+");
                    nodeCount = Integer.parseInt(split[0]);
                    inputCount = Integer.parseInt(split[1]);
                }
                
                if (nodeCount == -1 || line.startsWith("<") || line.trim().equals("[") || line.trim().equals("]")) {
                    continue;
                }

                float[][] weights = new float[nodeCount][inputCount];
                float[] bias = new float[nodeCount];

                // load weights. we read one more line for the bias vector.
                for (int i = 0; i < nodeCount + 1; i++) {
                    String l = i == 0 ? line : reader.readLine();
                    String[] weightStrings = l.replaceAll("\\[|\\]", "").trim().split("[ ]+");
                    int weightAmount = i < nodeCount ? inputCount : nodeCount;
                    float[] weightLine = new float[weightAmount];
                    for (int j = 0; j < weightLine.length; j++) {
                        weightLine[j] = Float.parseFloat(weightStrings[j]);
                    }
                    if (i < nodeCount) {
                        weights[i] = weightLine;
                    } else {
                        bias = weightLine;
                    }
                }
                Layer l = new Layer(weights, bias);
                layers.add(l);
            }
        }
        return layers;
    }

    public static FeedForwardNetwork loadFromBinary(File file) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            int layerCount = dis.readInt();
            List<Layer> layers = new ArrayList<>(layerCount);
            for (int i = 0; i < layerCount; i++) {
                Layer l = Layer.loadFromStream(dis);
                /*if(i==0)
                    l.dumpWeightHistogram();*/
                layers.add(l);
            }
            int inputSize = layers.get(0).inputDimension;
            float[] shiftVector = deserializeRaw(dis, inputSize);
            float[] scaleVector = deserializeRaw(dis, inputSize);
            return new FeedForwardNetwork(layers, shiftVector, scaleVector);
        }
    }

    public void saveBinary(File file) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
            dos.writeInt(layers.size());
            for (Layer layer : layers) {
                layer.saveToStream(dos);
            }
            serializeRaw(dos, shiftVector);
            serializeRaw(dos, scaleVector);
        }
    }

    public static float[] deserializeRaw(DataInputStream dis, int amount) throws IOException {
        float[] result = new float[amount];
        for (int i = 0; i < amount; i++) {
            result[i] = dis.readFloat();
        }
        return result;
    }

    public static void serializeRaw(DataOutputStream dos, float[] data) throws IOException {
        for (float v : data) {
            dos.writeFloat(v);
        }
    }

    public static class Layer {
        public float[][] weights;
        public float[] bias;
        public int inputDimension;
        public int outputDimension;

        public Layer(float[][] weights, float[] bias) {
            this.weights = weights;
            this.bias = bias;
            this.inputDimension = weights[0].length;
            this.outputDimension = weights.length;
        }

        public void align(int inputAlignment, int outputAlignment) {
            // align bias
            bias = FloatData.alignTo(bias, outputAlignment);
            int paddedOut = FloatData.alignedSize(weights.length, outputAlignment);
            int paddedIn = FloatData.alignedSize(weights[0].length, inputAlignment);

            float[][] aligned = new float[paddedOut][paddedIn];
            for (int i = 0; i < paddedOut; i++) {
                if (i < weights.length) {
                    aligned[i] = Arrays.copyOf(weights[i], paddedIn);
                } else {
                    aligned[i] = new float[paddedIn];
                }
            }
            this.weights = aligned;
            this.inputDimension = weights[0].length;
            this.outputDimension = weights.length;
        }

        void extend(int inputNodeCount, int outputNodeCount) {
            // extend bias
            bias = extend(this.bias, outputNodeCount);

            float[][] newWeights = new float[outputNodeCount][];

            for (int i = 0; i < outputDimension; i++) {
                newWeights[i] = extend(weights[i], inputNodeCount);
            }
            for(int i = outputDimension; i<outputNodeCount; i++) {
                newWeights[i] = newWeights[i%outputDimension].clone();
            }
            this.weights = newWeights;
            this.inputDimension = weights[0].length;
            this.outputDimension = weights.length;
        }

        // extends an array by copying input array circularly
        static float[] extend(float[] input, int size) {
            float[] result = new float[size];
            for (int i = 0; i < result.length; i++) {
                result[i] = input[i % input.length];
            }
            return result;
        }


        public static Layer loadFromStream(DataInputStream dis) throws IOException {
            int inputDimension = dis.readInt();
            int outputDimension = dis.readInt();
            float[][] weights = new float[outputDimension][inputDimension];
            float[] bias = new float[outputDimension];
            for (int i = 0; i < outputDimension + 1; i++) {
                // for nodes, there are input amount of weights. For bias, node amount.
                int dim = i < outputDimension ? inputDimension : outputDimension;
                float[] weightLine = new float[dim];
                for (int j = 0; j < weightLine.length; j++) {
                    weightLine[j] = dis.readFloat();
                }
                if (i < outputDimension) {
                    weights[i] = weightLine;
                } else {
                    bias = weightLine;
                }
            }
            return new Layer(weights, bias);
        }

        public void saveToStream(DataOutputStream dos) throws IOException {
            dos.writeInt(inputDimension);
            dos.writeInt(outputDimension);
            for (int i = 0; i < outputDimension + 1; i++) {
                float[] weightLine = i < outputDimension ? weights[i] : bias;
                for (float w : weightLine) {
                    dos.writeFloat(w);
                }
            }
        }

        /**
         * Saves only a portion of the layer. Used for debugging purposes.
         */
        public void saveToStream(DataOutputStream dos, int inputSize, int outputSize) throws IOException {
            dos.writeInt(inputSize);
            dos.writeInt(outputSize);
            for (int i = 0; i < outputSize + 1; i++) {
                float[] weightLine = i < outputSize ? weights[i] : bias;
                int amount = i == outputSize ? outputSize : inputSize;
                for (int j = 0; j < amount; j++) {
                    dos.writeFloat(weightLine[j]);
                }
            }
        }

        /**
         * Calculates layer activations.
         */
        public float[] activations(float[] inputVector) {
            float[] result = new float[outputDimension];
            int i = 0;
            for (float[] nodeWeights : weights) {
                float sum = 0;
                for (int j = 0; j < nodeWeights.length; j++) {
                    sum += inputVector[j] * nodeWeights[j];
                }
                result[i] = sum + bias[i];
                i++;
            }
            return result;
        }

        /**
         * Calculates layer activations for multiple vectors.
         */
        public List<FloatData> activations(List<FloatData> inputVectors) {
            List<FloatData> result = new ArrayList<>();
            for (FloatData inputVector : inputVectors) {
                result.add(inputVector.copy(activations(inputVector.getData())));
            }
            return result;
        }
    }

    public static void sigmoid(List<FloatData> inputVectors) {
        for (FloatData inputVector : inputVectors) {
            sigmoid(inputVector.getData());
        }
    }

    public static void softMax(List<FloatData> inputVectors) {
        for (FloatData inputVector : inputVectors) {
            softMax(inputVector.getData());
        }
    }

    public static void sigmoid(float[] f) {
        for (int i = 0; i < f.length; i++) {
            f[i] = (float) (1f / (1 + Math.exp(-f[i])));
        }
    }

    public static void softMax(float[] f) {
        float total = 0;
        float[] expArray = new float[f.length];
        for (int i = 0; i < expArray.length; i++) {
            expArray[i] = (float) Math.exp(f[i]);
            total += expArray[i];
        }
        for (int i = 0; i < f.length; i++) {
            f[i] = expArray[i] / total;
        }
    }
}
