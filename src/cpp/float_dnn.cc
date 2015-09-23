//
// Created by afsina on 5/24/15.
//
#include <iostream>
#include <vector>
#include "dnn.h"
#include <chrono>

namespace dnn {

    int paddedSize(int num, int div);

    FloatDnn::FloatDnn(std::string fileName) {

        BinaryLoader *loader  = new BinaryLoader(fileName, false);

        int layerCount = loader->load_int();

#ifdef DEBUG
        cout << "Layer count = " << layerCount << endl;
#endif

        this->layers = std::vector<FloatLayer*>((unsigned long) layerCount);
        int actualInputDimension = 0;

        for (int j = 0; j < layerCount; j++) {

            int inputDimension = loader->load_int();
            if (j == 0) {
                actualInputDimension = inputDimension;
            }
            // make sure input is a factor of 4 for the first layer
            int paddedInputDim = j == 0 ? dnn::paddedSize(inputDimension, 4) : inputDimension;
            int outputDimension = loader->load_int();

#ifdef DEBUG
            cout << "Layer " << j << " input dimension = " << inputDimension << endl;
            if (j == 0) {
                cout << "Layer " << j << " padded input dimension = " << paddedInputDim << endl;
            }
            cout << "Layer " << j << " output dimension = " << outputDimension << endl;
#endif
            //load weights
            float **weights = new float *[outputDimension];
            for (int oo = 0; oo < outputDimension; oo++) {
                weights[oo] = new float[paddedInputDim];
                for (int ii = 0; ii < paddedInputDim; ii++) {
                    float d = ii < inputDimension ? loader->load_float() : 0;
                    weights[oo][ii] = d;
                }
            }

            // load bias values.
            float *bias = new float[outputDimension];
            for (int ii = 0; ii < outputDimension; ii++) {
                bias[ii] = loader->load_float();
            }

            dnn::FloatLayer *layer = new dnn::FloatLayer(weights, bias, paddedInputDim, outputDimension);
            layers[j] = layer;
        }

        // set input layer reference.
        this->inputLayer = layers[0];

        // load shift vector. This is added to input vector.
        this->shift = loader->loadFloatArray(actualInputDimension, this->inputLayer->inputDim);


        // load scale vector. This is multiplied to input vector.
        this->scale = loader->loadFloatArray(actualInputDimension, this->inputLayer->inputDim);

        delete loader;
    }

    int paddedSize(int num, int div) {
        int dif = div - num % div;
        if (dif == div)
            return num;
        else
            return num + dif;
    }

    BinaryLoader::BinaryLoader(std::string fileName, bool littleEndian) {
        FILE *pFile = fopen(fileName.c_str(), "rb");

        // obtain file size:
        fseek(pFile, 0, SEEK_END);
        long lSize = ftell(pFile);
        rewind(pFile);

        // allocate memory to contain the whole file:
        this->content = new char[lSize];
        this->length = (int) lSize;
        this->littleEndian = littleEndian;

        // copy the file into the buffer:
        size_t result = fread(this->content, 1, (size_t) lSize, pFile);
        if ((long) result != lSize) {
            fputs("Reading error", stderr);
            exit(3);
        }
        // terminate
        fclose(pFile);
    }

    BatchData::BatchData(std::string fileName) {

        BinaryLoader *loader = new BinaryLoader(fileName, false);

        int frameCount = loader->load_int();
        this->dimension = loader->load_int();
        this->vectorCount = frameCount;

        this->data = new float[this->dimension * frameCount]();

        int t = 0;
        for (int j = 0; j < frameCount; ++j) {
            for (int k = 0; k < this->dimension; ++k) {
                float d = loader->load_float();
                this->data[t] = d;
                t++;
            }
        }

        delete loader;
    }

    BatchData::BatchData(float *input, int vectorCount, int dimension) {
        this->data = input;
        this->vectorCount = vectorCount;
        this->dimension = dimension;
    }
}
