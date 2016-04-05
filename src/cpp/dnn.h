//
// Created by afsina on 4/26/15.
//
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <vector>
#include <string>
#include <assert.h>
#include <math.h>
#include "float_dnn.h"

#ifndef DNN_DNN_H
#define DNN_DNN_H

using namespace std;

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

static const float SIGMOID_QUANTIZATION_MULTIPLIER = 255.0f;
static const unsigned char SIGMOID_QUANTIZATION_MULTIPLIER_UCHAR = 255;

static const int SIGMOID_LOOKUP_SIZE = 1280; // arbitrary 64*28
static const int SIGMOID_HALF_LOOKUP_SIZE = SIGMOID_LOOKUP_SIZE/2;

class QuantizedSigmoid {
 public:
  unsigned char *lookup;

  QuantizedSigmoid();

  inline unsigned char get(float input) {
    int k = (int) roundf(input * 100);
    if (k <= -SIGMOID_HALF_LOOKUP_SIZE) return 0;
    if (k >= SIGMOID_HALF_LOOKUP_SIZE)
      return dnn::SIGMOID_QUANTIZATION_MULTIPLIER_UCHAR;
    return lookup[k + dnn::SIGMOID_HALF_LOOKUP_SIZE];
  }
};

class SoftMax {
 private:
  float *expArray;
  size_t size;

 public:
  SoftMax(size_t size) {
    expArray = new float[size];
    this->size = size;
  }

  void apply(float *input);

  ~SoftMax() { delete[] expArray; }
};

// Layer for SIMD Float Dnn
class FloatSimdLayer {
 public:
  __m128 *weights;
  float *bias;
  size_t inputDimension;
  size_t nodeCount;

  FloatSimdLayer(){};

  FloatSimdLayer(const FloatLayer *floatLayer);

  void validate();

  ~FloatSimdLayer() {
    aligned_free(weights);
    delete bias;
  }
};

// Layer for Quantized DNN
class QuantizedSimdLayer {
 public:
  __m128i *weights;
  float *bias;

  size_t inputDim;
  size_t nodeCount;
  float multiplier;

  QuantizedSimdLayer(const FloatLayer &floatLayer, float cutoff);

  ~QuantizedSimdLayer() {
    aligned_free(weights);
    delete bias;
  }

};

// DNN with quantized SIMD layers. Only the input layer is not quantized.
class QuantizedDnn {
 public:
  FloatSimdLayer *inputLayer;
  std::vector<QuantizedSimdLayer *> layers;
  QuantizedSimdLayer *outputLayer;
  __m128 *shift;
  __m128 *scale;

  QuantizedDnn(const FloatDnn &floatDnn, float cutoff);

  size_t outputDimension() { return this->outputLayer->nodeCount; }

  size_t inputDimension() { return inputLayer->inputDimension; }

  size_t layerCount() { return (size_t)layers.size(); }

  void applyShiftAndScale(const BatchData &input);

  ~QuantizedDnn() {
    delete inputLayer;
    for (QuantizedSimdLayer *layer : layers) {
      delete layer;
    }
    aligned_free(shift);
    aligned_free(scale);
  }
};

class CalculationContext {
 public:
  QuantizedDnn *dnn;
  // BatchData *input;

  size_t inputCount;

  // represents the amount of input vectors that outputs will be calculated in
  // one pass.
  size_t batchSize;

  // hidden layer node counts
  size_t hiddenNodeCount;

  // quantized inputs. This is used in all layers except input layer. This is
  // actually a two dimensional matrix.
  unsigned char *quantizedActivations;

  // represents the buffer amount of float activations as the result of weight
  // input matrix multiplication and addition of bias.
  // this is actually a flattened two dimensional array.
  float *activations;

  float *singleOutput;

  SoftMax *softMax;


  CalculationContext(QuantizedDnn *dnn, size_t inputCount, size_t batchSize);

  void lastHiddenLayerActivations(const BatchData &input);

  void quantizedLayerActivations(const QuantizedSimdLayer *layer, size_t batchStartIndex,
                                 float *sequentialActivations);

  void inputActivations(const BatchData &input, size_t batchIndex);

  void addBias(const float *bias);

  void quantizedSigmoid(size_t batchIndex);

  // void convertSequentialActivations();

  BatchData *calculateOutput();

  void test(const BatchData &input);

  float *calculate(const BatchData &input);

  float *lazyOutputActivations(size_t inputIndex, const char *outputNodes);

  ~CalculationContext() {
    delete quantizedActivations;
    delete activations;
    delete softMax;
    delete singleOutput;
  }
};
}
#endif  // DNN_DNN_H
