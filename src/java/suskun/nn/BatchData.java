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

    /**
     * Saves data to stream with this format.
     * <pre>
     *     [id] UTF
     *     [frame count] int32 BE
     *       [frame id] int32 BE
     *       [data] float32 BE
     *       ...
     *       [data]
     *       [frame id] int32 BE
     *       [data]
     *       ...
     *       [data]
     * </pre>
     */
    public void saveBinaryToStream(DataOutputStream dos) throws IOException {
        dos.writeUTF(id);
        dos.write(data.size());
        for (FloatData floatData : data) {
            floatData.serializeToBinaryStream(dos);
        }
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

    public void serializeDataMatrix(DataOutputStream dos, boolean bigEndian, int featureAmount) throws IOException {
        if (data.size() == 0) {
            throw new IllegalStateException("There is no data to serialize.");
        }

        if (featureAmount < 0)
            featureAmount = data.size();

        featureAmount = featureAmount < data.size() ? featureAmount : data.size();

        if (bigEndian) {
            dos.writeInt(featureAmount);
            dos.writeInt(data.get(0).size());
        } else {
            dos.writeInt(Integer.reverseBytes(featureAmount));
            dos.writeInt(Integer.reverseBytes(data.get(0).size()));
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
    public static List<BatchData> loadInputFromText(File featureFile) throws IOException {

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
    public static String format(float... input) {
        return format(10, 3, " ", input);
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

    public static float[] fromString(String str, String delimiter) {
        String[] tokens = str.split("[" + delimiter + "]");
        float[] result = new float[tokens.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = Float.parseFloat(tokens[i]);
        }
        return result;
    }

}

