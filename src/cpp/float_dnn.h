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

 private:

  float *data_;
  size_t dimension_;
  size_t vector_count_;

 public:

  BatchData(std::string fileName);

  BatchData(float *input, size_t vectorCount, size_t dimension);

  size_t dimension() const { return dimension_; }
  size_t vector_count() const { return vector_count_; }

  float *data() const { return data_; }

  ~BatchData() { delete data_; }
};

/* A simple class for loading binary data from a file. It can load little/big
 * endian int32 and float32 values
 * This class contains an offset pointer so it is stateful.
 */
class BinaryLoader {
 private:
  char *content_;
  int offset_ = 0;
  int length_;
  bool little_endian_;
  char *four_bytes_;
  char *eight_bytes_;

 public:

  BinaryLoader(std::string fileName, bool littleEndian);

  // loads 4 byte and returns the value as int
  int load_int() {
    char *bytes = loadFourBytes(sizeof(int));
    return *(reinterpret_cast<int *>(bytes));
  }

  // loads 4 byte and returns the value as size_t
  size_t load_size_t() {
    char *bytes = loadFourBytes(sizeof(size_t));
    return *(reinterpret_cast<size_t *> (bytes));
  }

  // loads 4 byte and returns the value as float
  float load_float() {
    char *bytes = loadFourBytes(sizeof(float));
    return *(reinterpret_cast<float *>(bytes));
  }

  // loads an array of 32 bit float array. However, it pads zeroes if paddedSize
  // is larger than amount.
  float *load_float_array(size_t amount, size_t padded_size) {
    float *values = new float[padded_size];
    for (size_t i = 0; i < padded_size; ++i) {
      values[i] = i < amount ? load_float() : 0;
    }
    return values;
  }

  ~BinaryLoader() {
    delete[] content_;
    delete[] four_bytes_;
    delete[] eight_bytes_;
  }

 private:

  // loads 4 byte content, converts to little endian representation if necessary.
  // resulting byte array is [size] bytes. [size] can only be 4 or 8 bytes.
  char *loadFourBytes(int size) {
    assert(offset_ < length_);
    assert(size >= 4 || size == 8);

    char *bytes = size == 4 ? four_bytes_ : eight_bytes_;
    std::fill(bytes, bytes + size, 0);

    for (size_t i = 0; i < 4; ++i) {
      char c = content_[offset_ + i];
      if (little_endian_) {
        bytes[i] = content_[offset_ + i];
      } else {
        bytes[3 - i] = c;
      }
    }
    offset_ = offset_ + 4;
    return bytes;
  }

};

class LayerBase {

 protected:
  float *bias_;
  size_t input_dimension_;
  size_t node_count_;

  LayerBase() { };


 public:
  LayerBase(float *bias_, size_t input_dimension_, size_t node_count_)
      : bias_(bias_), input_dimension_(input_dimension_), node_count_(node_count_) { }

 public:
  size_t input_dimension() const { return input_dimension_; }

  size_t node_count() const { return node_count_; }

  float *bias() const { return bias_; }

  virtual ~LayerBase() {
    delete bias_;
  }

};

// Layer for FloatDnn
class FloatLayer: public LayerBase {
 private:
  float **weights_;

 public:
  FloatLayer() { };

  FloatLayer(float **weights, float *bias, size_t inputDim, size_t nodeCount)
      : weights_(weights), LayerBase(bias, inputDim, nodeCount) { }

  float **weights() const { return weights_; }

  ~FloatLayer() {
    delete[] weights_;
  }
};

// DNN with 32 bit floating numbers. This is only used for constructing a QuantizedDnn.
class FloatDnn {

 private:
  FloatLayer *input_layer_;
  std::vector<FloatLayer *> layers_;
  float *shift_;
  float *scale_;

 public:
  FloatDnn(std::string fileName);

  size_t output_size() const {
    return this->layers_[this->layers_.size() - 1]->node_count();
  }

  size_t input_dimension() const { return input_layer_->input_dimension(); }

  size_t layer_count() const { return layers_.size(); }

  FloatLayer *input_layer() const { return input_layer_; }

  std::vector<FloatLayer *> layers() const { return layers_; }

  float *shift() const { return shift_; }
  float *scale() const { return scale_; }

  ~FloatDnn() {
    for (FloatLayer *layer : layers_) {
      delete layer;
    }
    delete shift_;
    delete scale_;
  }
};
}

#endif  // DNN_FLOAT_DNN_H
