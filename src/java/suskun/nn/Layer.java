package suskun.nn;

public class Layer {
    public final float[][] weights;
    public final float[] bias;
    public final int inputDimension;
    public final int outputDimension;

    public Layer(float[][] weights, float[] bias, int inputDimension, int outputDimension) {
        this.weights = weights;
        this.bias = bias;
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
    }
}
