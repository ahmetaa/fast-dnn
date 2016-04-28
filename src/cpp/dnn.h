#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include "float_dnn.h"

#ifndef DNN_DNN_H
#define DNN_DNN_H

namespace dnn {

inline void *aligned_malloc(size_t align, size_t size) {
  void *result;
#ifdef _MSC_VER
  result = _aligned_malloc(size, align);
#else
  if (posix_memalign(&result, align, size)) result = 0;
#endif
  return result;
}

inline void aligned_free(void *ptr) {
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

template<typename T>
inline T *AlignedAlloc(size_t count) {
  return reinterpret_cast<T *> (aligned_malloc(16, sizeof(T) * count));
}

const float SIGMOID_QUANTIZATION_MULTIPLIER = 255.0f;
const unsigned char SIGMOID_QUANTIZATION_MULTIPLIER_UCHAR = 255;

const int SIGMOID_LOOKUP_SIZE = 1280; // arbitrary 64*28
const int SIGMOID_HALF_LOOKUP_SIZE = SIGMOID_LOOKUP_SIZE / 2;

class QuantizedSigmoid {

 public:
  QuantizedSigmoid();
  ~QuantizedSigmoid() { delete[] lookup_; }

  unsigned char get(float input) {
    int k = static_cast<int> (round(input * 100));
    if (k <= -SIGMOID_HALF_LOOKUP_SIZE) return 0;
    if (k >= SIGMOID_HALF_LOOKUP_SIZE) {
      return dnn::SIGMOID_QUANTIZATION_MULTIPLIER_UCHAR;
    }
    return lookup_[k + dnn::SIGMOID_HALF_LOOKUP_SIZE];
  }

 private :
  unsigned char *lookup_;

};

class SoftMax {

 public:
  SoftMax(size_t size) : size_(size) {
    exp_array_ = new float[size];
  }

  void apply(float *input);

  ~SoftMax() { delete[] exp_array_; }

 private:
  float *exp_array_;
  size_t size_;

};

// Layer for SIMD Float Dnn. Used in input layer because we do not quantize input values.
class FloatSimdLayer: public LayerBase {

 public:
  FloatSimdLayer() { };

  FloatSimdLayer(const FloatLayer *float_layer);

  float *weights() const { return weights_; }

  ~FloatSimdLayer() {
    aligned_free(weights_);
  }

 private :
  float *weights_;
};

// Layer for Quantized DNN
class QuantizedSimdLayer: public LayerBase {

 public:

  QuantizedSimdLayer(const FloatLayer &floatLayer, float cutoff);

  float multiplier() const { return multiplier_; }

  char *weights() const { return weights_; }

  ~QuantizedSimdLayer() {
    aligned_free(weights_);
  }

 private:
  char *weights_;
  float multiplier_;

};

// DNN with quantized SIMD layers. Only the input layer is not quantized.
class QuantizedDnn {

 public:

  QuantizedDnn(const FloatDnn &floatDnn, float cutoff);

  size_t output_dimension() const { return output_layer_->node_count(); }

  size_t input_dimension() const { return input_layer_->input_dimension(); }

  size_t layer_count() const { return (size_t) layers_.size(); }

  QuantizedSimdLayer *output_layer() const { return output_layer_; };

  FloatSimdLayer *input_layer() const { return input_layer_; };

  std::vector<QuantizedSimdLayer *> layers() const { return layers_; };

  void ApplyShiftAndScale(const BatchData &input);

  ~QuantizedDnn() {
    delete input_layer_;
    for (QuantizedSimdLayer *layer : layers_) {
      delete layer;
    }
    aligned_free(shift_);
    aligned_free(scale_);
  }

 private:

  FloatSimdLayer *input_layer_;
  std::vector<QuantizedSimdLayer *> layers_;
  QuantizedSimdLayer *output_layer_;
  float *shift_;
  float *scale_;
};

class CalculationContext {

 public:

  CalculationContext(
      QuantizedDnn *dnn,
      size_t input_count,
      size_t batch_size);

  void CalculateUntilLastHiddenLayer(const BatchData &input);

  void QuantizedLayerActivations(
      const QuantizedSimdLayer &layer,
      size_t batch_start_index,
      float *linear_activations);

  void InputActivations(const BatchData &input, size_t batch_index);

  void AddBias(const float *bias);

  void QuantizedSigmoid(size_t batch_index);

  size_t input_count() const { return input_count_; }

  QuantizedDnn *dnn() const { return dnn_; }

  BatchData *CalculateOutput();

  BatchData *Calculate(const BatchData &input);

  float *LazyOutputActivations(size_t inputIndex, const char *outputNodes);

  ~CalculationContext() {
    aligned_free(quantized_activations_);
    aligned_free(activations_);
    aligned_free(single_output_);
    delete soft_max_;
  }

 private:
  QuantizedDnn *dnn_;

  size_t input_count_;

  // represents the amount of input vectors that outputs will be calculated in
  // one pass.
  size_t batch_size_;

  // hidden layer node counts
  size_t hidden_node_count_;

  // quantized inputs. This is used in all layers except input layer. This is
  // actually a two dimensional matrix.
  unsigned char *quantized_activations_;

  // represents the buffer amount of float activations as the result of weight
  // input matrix multiplication and addition of bias.
  // this is actually a flattened two dimensional array.
  float *activations_;

  // this is used for lazy calculation
  float *single_output_;

  SoftMax *soft_max_;
};
}
#endif  // DNN_DNN_H
