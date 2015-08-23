//
// Created by afsina on 5/24/15.
//

#ifndef DNN_FLOAT_DNN_H
#define DNN_FLOAT_DNN_H

#include <vector>
#include <string>
#include <assert.h>
#include <iostream>

using  namespace std;

namespace dnn {

    template<typename T>
    void print_container(const T &c, int amount) {
        cout << "[";
        bool isFirst = true;
        for (int i = 0; i < amount; ++i) {
            if (isFirst) isFirst = false;
            else cout << ", ";
            cout << c[i];
        }
        cout << "]" << endl;
    }

    template<typename T>
    void print_int(const T &c, int amount) {
        cout << "[";
        bool isFirst = true;
        for (int i = 0; i < amount; ++i) {
            if (isFirst) isFirst = false;
            else cout << ", ";
            cout << (int) c[i];
        }
        cout << "]" << endl;
    }

    /*
     * This class actually holds a float32 matrix with [dimension] columns and [frameCount] rows.
     * This will hold the input data of the DNN.
     */
    class BatchData {
    public:
        float *features;
        int dimension;
        int frameCount;

        BatchData(std::string fileName, int batchSize);
        BatchData(float* input, int frameCount, int dimension, int batchSize);
    };

    /* A simple class for loading binary data from a file. It can load little endian int32 and float32 values
     * This class contains an offset pointer so it is stateful.
     */
    class BinaryLoader {

    public:
        char *content;
        int offset = 0;
        int length;

        BinaryLoader(std::string fileName);

        // loads a 32 bit little endian integer.
        int load_int() {
            assert(offset < length);
            int val = *(reinterpret_cast<int *> (content + offset));
            offset += 4;
            return val;
        }

        // loads a 32 bit little endian float.
        float load_float() {
            assert(offset < length);
            float val = *(reinterpret_cast<float *> (content + offset));
            offset += 4;
            return val;
        }

        // loads an array of 32 bit float array. However, it pads zeroes if paddedSize is larger than amount.
        float *loadFloatArray(int amount, int paddedSize) {
            float *values = new float[paddedSize];
            for (int i = 0; i < paddedSize; ++i) {
                values[i] = i < amount ? load_float() : 0;
            }
            return values;
        }
    };

// Layer for FloatDnn
    class FloatLayer {

    public:
        float **weights;
        float *bias;
        int inputDim;
        int nodeCount;

        FloatLayer() { };

        FloatLayer(float **weights, float *bias, int inputDim, int nodeCount) : weights(weights), bias(bias),
                                                                              inputDim(inputDim),
                                                                                nodeCount(nodeCount) { }

        ~FloatLayer() {
        }

    };

// DNN with 32 bit floating numbers.
    class FloatDnn {

    public:
        FloatLayer *inputLayer;
        std::vector<FloatLayer> layers;
        float *shift;
        float *scale;

        FloatDnn(std::string fileName);

        long outputSize() {
            return this->layers[this->layers.size() - 1].nodeCount;
        }

        int inputDimension() {
            return inputLayer->inputDim;
        }

        int layerCount() {
            return (int) layers.size();
        }

        ~FloatDnn() {
        }
    };

}

#endif //DNN_FLOAT_DNN_H
