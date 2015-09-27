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

static const float SIGMOID_QUANTIZATION_MULTIPLIER = 200.0f;

static const unsigned char SIGMOID_QUANTIZATION_MULTIPLIER_UCHAR = 200;

static const int SIGMOID_HALF_LOOKUP_SIZE = 544;

class QuantizedSigmoid {
 public:
  unsigned char *lookup;

  QuantizedSigmoid();

  inline unsigned char get(float input) {
    int k = (int)(input * 100);
    if (k <= -SIGMOID_HALF_LOOKUP_SIZE) return 0;
    if (k >= SIGMOID_HALF_LOOKUP_SIZE)
      return dnn::SIGMOID_QUANTIZATION_MULTIPLIER_UCHAR;
    return lookup[k + dnn::SIGMOID_HALF_LOOKUP_SIZE];
  }
};

class SoftMax {
 private:
  float *expArray;
  int size;

 public:
  SoftMax(int size) {
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
  int inputDimension;
  int nodeCount;

  FloatSimdLayer(){};

  FloatSimdLayer(FloatLayer *floatLayer);

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

  int inputDim;
  int nodeCount;
  float multiplier;

  QuantizedSimdLayer(){};

  QuantizedSimdLayer(const FloatLayer &floatLayer);

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

  QuantizedDnn(const FloatDnn &floatDnn);

  int outputDimension() { return this->outputLayer->nodeCount; }

  int inputDimension() { return inputLayer->inputDimension; }

  int layerCount() { return (int)layers.size(); }

  void applyShiftAndScale(BatchData *input);

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

  int inputCount;

  // represents the amount of input vectors that outputs will be calculated in
  // one pass.
  int batchSize;

  // hidden layer node counts
  int hiddenNodeCount;

  // quantized inputs. This is used in all layers except input layer. This is
  // actually a two dimensional matrix.
  unsigned char *quantizedActivations;

  // represents the buffer amount of float activations as the result of weight
  // input matrix multiplication and addition of bias.
  // this is actually a flattened two dimensional array.
  float *activations;

  SoftMax *softmax;

  CalculationContext(QuantizedDnn *dnn, int inputCount, int batchSize);

  void lastHiddenLayerActivations(BatchData *input);

  void quantizedLayerActivations(QuantizedSimdLayer *layer, int batchStartIndex,
                                 float *sequentialActivations);

  void inputActivations(BatchData *input, int batchIndex);

  void addBias(float *bias);

  void quantizedSigmoid(int batchIndex);

  // void convertSequentialActivations();

  BatchData *calculateOutput();

  void test(BatchData *input);

  float *calculate(BatchData *input);

  float *lazyBatchOutputActivations(int batchStartIndex, int *outputNodes,
                                    int outputCount);

  float *lazyOutputActivations(int batchStartIndex, char *outputNodes);

  ~CalculationContext() {
    delete quantizedActivations;
    delete activations;
    delete softmax;
  }
};
}
#endif  // DNN_DNN_H
