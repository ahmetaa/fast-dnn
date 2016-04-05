//
// Created by afsina on 5/24/15.
//

#ifndef DNN_FLOAT_DNN_H
#define DNN_FLOAT_DNN_H

#include <vector>
#include <string>
#include <assert.h>
#include <iostream>
#include <string.h>

using namespace std;

namespace dnn {

template<typename T>
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

template<typename T>
void print_int(const T &c, int amount) {
  cout << "[";
  bool isFirst = true;
  for (int i = 0; i < amount; ++i) {
    if (isFirst)
      isFirst = false;
    else
      cout << ", ";
    cout << (int) c[i];
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
  size_t dimension;
  size_t vectorCount;

  BatchData(std::string fileName);

  BatchData(float *input, size_t vectorCount, size_t dimension);

  ~BatchData() { delete data; }
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
  char *fourBytes;
  char *eightBytes;

  BinaryLoader(std::string fileName, bool littleEndian);

  // loads a 32 bit integer.
  int load_int() {
    char *bytes = loadFourBytes(sizeof(int));
    return *(reinterpret_cast<int *>(bytes));
  }

  // loads to an int
  size_t load_size_t() {
    char *bytes = loadFourBytes(sizeof(size_t));
    return *(reinterpret_cast<size_t *> (bytes));
  }

  // loads 4 bytes and casts to size_t
  float load_float() {
    char *bytes = loadFourBytes(sizeof(float));
    return *(reinterpret_cast<float *>(bytes));
  }

  // loads an array of 32 bit float array. However, it pads zeroes if paddedSize
  // is larger than amount.
  float *loadFloatArray(size_t amount, size_t paddedSize) {
    float *values = new float[paddedSize];
    for (size_t i = 0; i < paddedSize; ++i) {
      values[i] = i < amount ? load_float() : 0;
    }
    return values;
  }

  ~BinaryLoader() {
    delete[] content;
    delete[] fourBytes;
    delete[] eightBytes;
  }

 private:

  char *loadFourBytes(int size) {
    assert(offset < length);
    assert(size >= 4 || size == 8);

    char *bytes = size == 4 ? fourBytes : eightBytes;
    std::fill(bytes, bytes + size, 0);

    for (size_t i = 0; i < 4; ++i) {
      char c = content[offset + i];
      if (littleEndian) {
        bytes[i] = content[offset + i];
      } else {
        bytes[3 - i] = c;
      }
    }
    offset = offset + 4;
    return bytes;
  }

};

// Layer for FloatDnn
class FloatLayer {
 public:
  float **weights;
  float *bias;
  size_t inputDim;
  size_t nodeCount;

  FloatLayer() { };

  FloatLayer(float **weights, float *bias, size_t inputDim, size_t nodeCount)
      : weights(weights),
        bias(bias),
        inputDim(inputDim),
        nodeCount(nodeCount) { }

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

  size_t outputSize() const {
    return this->layers[this->layers.size() - 1]->nodeCount;
  }

  size_t inputDimension() const { return inputLayer->inputDim; }

  size_t layerCount() const { return layers.size(); }

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
