package suskun.nn;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

class BatchData implements Comparable<BatchData> {
    public final String id; // id of the input
    private final List<FloatData> data; // contains data chunks

    public BatchData(String id, List<FloatData> data) {
        this.id = id;
        this.data = new ArrayList<>(data);
    }

    public BatchData(String id, FloatData[] data) {
        this.id = id;
        this.data = Arrays.asList(data);
    }

    public BatchData(String id) {
        this.id = id;
        data = new ArrayList<>();
    }

    public List<FloatData> getData() {
        return data;
    }

    public FloatData get(int index) {
        return data.get(index);
    }

    public void add(FloatData... arr) {
        Collections.addAll(data, arr);
    }

    public void add(Iterable<FloatData> frames) {
        for (FloatData fData : frames) {
            data.add(fData);
        }
    }

    public float[][] getAsFloatMatrix() {
        float[][] matrix = new float[data.size()][];
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = data.get(i).getCopyOfData();
        }
        return matrix;
    }

    public int vectorCount() {
        return data.size();
    }

    @Override
    public int compareTo(BatchData o) {
        return o.id.compareTo(id);
    }

    public String asTextRepresentation() {
        StringBuilder sb = new StringBuilder(1000);
        sb.append(id).append(" [\n");
        int i = 0;
        for (FloatData feature : data) {
            sb.append(format(3, feature.getData()));
            if (i < data.size() - 1)
                sb.append("\n");
            else
                sb.append("]");
            i++;
        }
        return sb.toString();
    }

    public static BatchData loadRawBinary(String id, File binaryFile) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(binaryFile)))) {
            int featureAmount = dis.readInt();
            int dimension = dis.readInt();
            List<FloatData> vectors = new ArrayList<>();
            for (int i = 0; i < featureAmount; i++) {
                float[] data = FeedForwardNetwork.deserializeRaw(dis, dimension);
                vectors.add(new FloatData(data, i));
            }
            return new BatchData(id, vectors);
        }
    }

    public BatchData alignDimension(int alignment) {
        for (FloatData floatData : data) {
            floatData.replaceData(FloatData.alignTo(floatData.getData(), alignment));
        }
        return this;
    }


    public void serializeDataMatrix(File file, int featureAmount) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
            serializeDataMatrix(dos, true, featureAmount);
        }
    }

    public void serializeDataMatrix(DataOutputStream dos, boolean bigEndian, int featureAmount) throws IOException {
        if (data.size() == 0) {
            throw new IllegalStateException("There is no data to serialize.");
        }

        if (featureAmount < 0)
            featureAmount = data.size();

        featureAmount = featureAmount < data.size() ? featureAmount : data.size();

        int dimension = data.get(0).size();

        if (bigEndian) {
            dos.writeInt(featureAmount);
            dos.writeInt(dimension);
        } else {
            dos.writeInt(Integer.reverseBytes(featureAmount));
            dos.writeInt(Integer.reverseBytes(dimension));
        }
        int k = 0;
        for (FloatData floatData : data) {
            if (bigEndian) {
                FeedForwardNetwork.serializeRaw(dos, floatData.getData());
            } else {
                for (float v : floatData.getData()) {
                    dos.writeInt(Integer.reverseBytes(Float.floatToIntBits(v)));
                }
            }
            if (k == featureAmount)
                break;
            k++;
        }
    }

    public static Pattern FEATURE_LINES_PATTERN = Pattern.compile("(?:\\[)(.+?)(?:\\])", Pattern.DOTALL);
    static Pattern ID_PATTERN = Pattern.compile("(.+?)(?:\\[.+?\\])", Pattern.DOTALL);
    static Pattern NEW_LINE_PATTERN = Pattern.compile("\\r|\\n", Pattern.DOTALL);

    /**
     * Loads features from Kaldi text file. Format is
     * utterance-id [
     * 0.0 1.1 .....
     * 2.3 4.5 .....
     * 5.6 7.8 ..... ]
     * utterance-id [
     * ....
     * ]
     */
    public static List<BatchData> loadMultipleFromText(File featureFile) throws IOException {

        List<String> lines = Files.readAllLines(featureFile.toPath(), StandardCharsets.UTF_8);
        String wholeThing = String.join("\n", lines);

        List<String> featureBlocks = firstGroupMatches(FEATURE_LINES_PATTERN, wholeThing);
        List<String> idLines = firstGroupMatches(ID_PATTERN, wholeThing);
        List<BatchData> result = new ArrayList<>();

        for (int i = 0; i < idLines.size(); i++) {
            String id = idLines.get(i);
            String block = featureBlocks.get(i);

            List<String> featureLines = NEW_LINE_PATTERN.splitAsStream(block).collect(Collectors.toList());
            List<FloatData> data = new ArrayList<>();

            int seq = 0;
            for (String featureLine : featureLines) {
                data.add(new FloatData(fromString(featureLine, " "), seq++));
            }

            result.add(new BatchData(id, data));
        }
        return result;
    }

    public static BatchData loadFromText(File featureFile) throws IOException {
        return loadMultipleFromText(featureFile).get(0);
    }


    public void saveAsTextFile(File file) throws IOException {
        try (PrintWriter pw = new PrintWriter(file, "UTF-8")) {
            pw.println(asTextRepresentation());
        }
    }

    public static List<String> firstGroupMatches(Pattern p, String s) {
        List<String> matches = new ArrayList<>();
        Matcher m = p.matcher(s);
        while (m.find()) {
            matches.add(m.group(1).trim());
        }
        return matches;
    }

    /**
     * Formats a float array as string using English Locale.
     */
    public static String format(int fractionDigits, float... input) {
        return format(fractionDigits, " ", input);
    }

    /**
     * Formats a float array as string using English Locale.
     */
    public static String format(int fractionDigits, String delimiter, float... input) {
        StringBuilder sb = new StringBuilder();
        String formatStr = "%." + fractionDigits + "f";
        int i = 0;
        for (float v : input) {
            sb.append(String.format(Locale.ENGLISH, formatStr, v));
            if (i++ < input.length - 1) sb.append(delimiter);
        }
        return sb.toString();
    }

    public static float[] fromString(String str, String delimiter) {
        String[] tokens = str.split("[" + delimiter + "]");
        float[] result = new float[tokens.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = Float.parseFloat(tokens[i]);
        }
        return result;
    }

}

