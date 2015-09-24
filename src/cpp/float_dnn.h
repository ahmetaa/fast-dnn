//
// Created by afsina on 5/24/15.
//

#ifndef DNN_FLOAT_DNN_H
#define DNN_FLOAT_DNN_H

#include <vector>
#include <string>
#include <assert.h>
#include <iostream>

using namespace std;

namespace dnn {

template <typename T>
void print_container(const T &c, int amount) {
  cout << "[";
  bool isFirst = true;
  for (int i = 0; i < amount; ++i) {
    if (isFirst)
      isFirst = false;
    else
      cout << ", ";
    cout << c[i];
  }
  cout << "]" << endl;
}

template <typename T>
void print_int(const T &c, int amount) {
  cout << "[";
  bool isFirst = true;
  for (int i = 0; i < amount; ++i) {
    if (isFirst)
      isFirst = false;
    else
      cout << ", ";
    cout << (int)c[i];
  }
  cout << "]" << endl;
}

/*
 * This class actually holds a float32 matrix with [dimension] columns and
 * [frameCount] rows.
 * This will hold the input data of the DNN.
 */
class BatchData {
 public:
  float *data;
  int dimension;
  int vectorCount;

  BatchData(std::string fileName);

  BatchData(float *input, int vectorCount, int dimension);
};

/* A simple class for loading binary data from a file. It can load little/big
 * endian int32 and float32 values
 * This class contains an offset pointer so it is stateful.
 */
class BinaryLoader {
 public:
  char *content;
  int offset = 0;
  int length;
  bool littleEndian;

  BinaryLoader(std::string fileName, bool littleEndian);

  // loads a 32 bit integer.
  int load_int() {
    assert(offset < length);
    int val = *(reinterpret_cast<int *>(content + offset));
    offset += 4;
    return littleEndian ? val : toBigEndian(val);
  }

  // loads a 32 bit  float.
  float load_float() {
    int val = load_int();
    return *(reinterpret_cast<float *>(&val));
  }

  // loads an array of 32 bit float array. However, it pads zeroes if paddedSize
  // is larger than amount.
  float *loadFloatArray(int amount, int paddedSize) {
    float *values = new float[paddedSize];
    for (int i = 0; i < paddedSize; ++i) {
      values[i] = i < amount ? load_float() : 0;
    }
    return values;
  }

  ~BinaryLoader() { delete[] content; }

 private:
  // convert value to big endian representation
  int toBigEndian(int num) {
    return ((num >> 24) & 0xff) | ((num << 8) & 0xff0000) |
           ((num >> 8) & 0xff00) | ((num << 24) & 0xff000000);
  }
};

// Layer for FloatDnn
class FloatLayer {
 public:
  float **weights;
  float *bias;
  int inputDim;
  int nodeCount;

  FloatLayer(){};

  FloatLayer(float **weights, float *bias, int inputDim, int nodeCount)
      : weights(weights),
        bias(bias),
        inputDim(inputDim),
        nodeCount(nodeCount) {}

  ~FloatLayer() {
    delete[] weights;
    delete bias;
  }
};

// DNN with 32 bit floating numbers.
class FloatDnn {
 public:
  FloatLayer *inputLayer;
  std::vector<FloatLayer *> layers;
  float *shift;
  float *scale;

  FloatDnn(std::string fileName);

  long outputSize() const {
    return this->layers[this->layers.size() - 1]->nodeCount;
  }

  int inputDimension() const { return inputLayer->inputDim; }

  int layerCount() const { return (int)layers.size(); }

  ~FloatDnn() {
    for (FloatLayer *layer : layers) {
      delete layer;
    }
    delete shift;
    delete scale;
  }
};
}

#endif  // DNN_FLOAT_DNN_H
