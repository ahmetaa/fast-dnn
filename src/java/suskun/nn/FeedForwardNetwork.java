package suskun.nn;


import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

public class FeedForwardNetwork {
    // layers contain all weight and bias values.
    private List<Layer> layers;

    // for convenience we have separate Layer object references for output and first hidden layer.
    private Layer outputLayer;
    private Layer firstLayer;

    // transpose of shift and scale vectors.
    // Shift matrix is added to input matrix and result is multiplied with scale matrix.
    private float[] shiftVector;
    private float[] scaleVector;

    public Layer getLayer(int layerIndex) {
        if (layerIndex < 0 || layerIndex >= layers.size())
            throw new IllegalArgumentException("Illegal layer index " + layerIndex);
        return layers.get(layerIndex);
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

/*    *//**
     * Generate a JBlasDnn instance from text file
     *
     * @throws IOException
     *//*
    public static FeedForwardNetwork loadFromTextFile(File networkFile, File transformationFile) throws IOException {
        List<Layer> layers = loadLayersFromTextFile(networkFile);

        String wholeThing = new SimpleTextReader(transformationFile, "UTF-8").asString();
        List<String> featureBlocks = Regexps.firstGroupMatches(FEATURE_LINES_PATTERN, wholeThing);

        if (featureBlocks.size() != 2) {
            throw new IllegalStateException("Feature transformation file should have two vectors in it. But it has "
                    + featureBlocks.size());
        }

        float[] shiftVector = FloatArrays.fromString(featureBlocks.get(0), " ");
        float[] scaleVector = FloatArrays.fromString(featureBlocks.get(1), " ");

        int inputDimension = layers.get(0).inputDimension;
        if (shiftVector.length != inputDimension) {
            throw new IllegalStateException("Shift transformation vector size " + shiftVector.length +
                    " is not same as input dimension " + inputDimension);
        }
        if (scaleVector.length != inputDimension) {
            throw new IllegalStateException("Scale transformation vector size " + scaleVector.length +
                    " is not same as input dimension " + inputDimension);
        }
        return new JBlasDnn(layers, shiftVector, scaleVector);
    }

    private static List<Layer> loadLayersFromTextFile(File networkFile) throws IOException {
        List<Layer> layers = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(networkFile.toPath(), Charsets.UTF_8)) {

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

                if (nodeCount == -1 || line.startsWith("<"))
                    continue;

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
                        weights.putRow(i, new FloatMatrix(weightLine));
                    } else {
                        bias.putColumn(0, new FloatMatrix(weightLine));
                    }
                }
                Layer l = new Layer(weights, bias);
                layers.add(l);
            }
        }
        return layers;
    }   */
}
