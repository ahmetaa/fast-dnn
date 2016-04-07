#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <vector>
#include <math.h>
#include "dnn.h"
#include <chrono>

using namespace std;

int main() {
  std::string fName = "/home/kodlab/projects/fast-dnn/data/dnn.tv.model";
  const dnn::FloatDnn floatDnn(fName);
  std::string featureName = "/home/kodlab/projects/fast-dnn/data/16khz.bin";
  dnn::BatchData batchData(featureName);

  dnn::QuantizedDnn qDnn(floatDnn, 3);

  dnn::CalculationContext context(&qDnn, batchData.vector_count(), 8);

  context.Test(batchData);

  return 0;
}

namespace dnn {

static QuantizedSigmoid *qSigmoid = new QuantizedSigmoid();

__m128 *getSimdFloat(const float *values, size_t dim);

static inline float __horizontalSumFloat32(__m128 x);

static inline int __horizontalSumInt32(__m128i x);

static float inline quantizedNodeSum(const size_t vectorSize,
                                     const unsigned char *quantizedInput,
                                     const __m128i *weights);

const float WEIGHT_MULTIPLIER = 127;


QuantizedSigmoid::QuantizedSigmoid() {
  int size = SIGMOID_LOOKUP_SIZE;
  this->lookup_ = new unsigned char[SIGMOID_LOOKUP_SIZE];

  // when a sigmoid is quantized with 255, numbers below around -5.4 becomes 0,
  // numbers over around +5.4 becomes 254
  for (int i = -SIGMOID_HALF_LOOKUP_SIZE; i < SIGMOID_HALF_LOOKUP_SIZE; ++i) {
    float k = i / 100.0f;
    float sigmoid = 1.0f / (1 + expf(-k));
    unsigned char q =
        (unsigned char) roundf(sigmoid * dnn::SIGMOID_QUANTIZATION_MULTIPLIER);
    lookup_[i + size / 2] = q;
  }
}

inline unsigned char sigmoid(float i) {
  float k = 1.0f / (1.0f + expf(-i));
  return (unsigned char) (roundf(k * SIGMOID_QUANTIZATION_MULTIPLIER));
}

inline __m128 *SIMD_alloc(size_t simdBlockCount) {
  return (__m128 *) aligned_malloc(16, sizeof(__m128) * simdBlockCount);
}

inline __m128i *SIMD_i_alloc(size_t simdBlockCount) {
  return (__m128i *) aligned_malloc(16, sizeof(__m128i) * simdBlockCount);
}

inline char *byte_alloc(size_t count) {
  return (char *) aligned_malloc(16, sizeof(char) * count);
}

inline float *float_alloc(size_t count) {
  return (float *) aligned_malloc(16, sizeof(float) * count);
}

FloatSimdLayer::FloatSimdLayer(const FloatLayer *float_layer) {
  this->node_count_ = float_layer->node_count();
  this->input_dimension_ = float_layer->input_dimension();

  const size_t simdVectorDim = this->input_dimension_ / 4;

  this->weights_ = dnn::SIMD_alloc(this->node_count_ * simdVectorDim);

  __m128 *w = this->weights_;

  for (size_t i = 0; i < this->node_count_; ++i) {
    for (size_t j = 0; j < simdVectorDim; ++j) {
      float *p = &float_layer->weights()[i][j * 4];
      w[j] = _mm_load_ps(p);
    }
    w += simdVectorDim;
  }

  // we do not use simd for bias.
  this->bias_ = new float[LayerBase::node_count_];
  // copy the bias values.
  std::copy(float_layer->bias(), float_layer->bias() + float_layer->node_count(),
            this->bias_);
}

__m128 *getSimdFloat(const float *values, size_t dim) {
  size_t k = dim / 4;
  __m128 *result = dnn::SIMD_alloc(k);

  for (size_t i = 0; i < k; ++i) {
    result[i] = _mm_load_ps(&values[i * 4]);
  }
  return result;
}

float absMax(float *floats, size_t size, float trimMin, float trimMax) {
  float max = -numeric_limits<float>::max();
  for (size_t i = 0; i < size; i++) {
    float f = floats[i];
    if (f < trimMin) f = trimMin;
    if (f > trimMax) f = trimMax;
    float fAbs = (float) fabs(f);
    if (fAbs > max) {
      max = fAbs;
    }
  }
  return max;
}

void dump(__m128 data) {
  float temp[4] __attribute((aligned(4 * 4)));
  _mm_store_ps(&temp[0], data);
  print_container(&temp, 4);
}

void CalculationContext::Test(const BatchData &input) {
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::milliseconds milliseconds;

  Clock::time_point t0 = Clock::now();
  this->LastHiddenLayerActivations(input);
  this->CalculateOutput();
  Clock::time_point t1 = Clock::now();
  milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);

  std::cout << "Elapsed:" << ms.count() << std::endl;
}

float *CalculationContext::Calculate(const BatchData &input) {
  this->LastHiddenLayerActivations(input);
  BatchData *output = CalculateOutput();
  return output->data();
}

static inline float __horizontalSumFloat32(__m128 x) {
  x = _mm_hadd_ps(x, x);
  x = _mm_hadd_ps(x, x);
  return _mm_cvtss_f32(x);
}

void QuantizedDnn::ApplyShiftAndScale(const BatchData &batchInput) {
  // apply shift and scale with SIMD
  const size_t size = batchInput.dimension() / 4;

  float *input = batchInput.data();

  for (size_t i = 0; i < batchInput.vector_count(); ++i) {
    for (size_t k = 0; k < size; ++k) {
      __m128 val = _mm_load_ps(&input[k * 4]);
      val = _mm_add_ps(val, this->shift_[k]);
      val = _mm_mul_ps(val, this->scale_[k]);
      _mm_store_ps(&input[k * 4], val);
    }
    input += batchInput.dimension();
  }
}

CalculationContext::CalculationContext(QuantizedDnn *dnn, size_t input_count,
                                       size_t batch_size) {
  this->dnn_ = dnn;
  this->batch_size_ = batch_size;
  this->hidden_node_count_ = this->dnn_->layers()[1]->node_count();
  this->input_count_ = input_count;

  // allocate for float activations. Only batch amount.
  this->activations_ = dnn::float_alloc(this->hidden_node_count_ * batch_size);

  // allocate for quantized unsigned char input values.
  this->quantized_activations_ =
      (unsigned char *) dnn::byte_alloc(this->hidden_node_count_ * input_count);

  this->soft_max_ = new SoftMax(dnn->output_dimension());

  this->single_output_ = dnn::float_alloc(dnn->output_dimension());

}

void CalculationContext::InputActivations(const BatchData &inputData,
                                          size_t batch_index) {
  const size_t dimension = this->dnn_->input_dimension();
  const size_t vectorInputSize = dimension / 4;

  // for each node.
  const __m128 *w = this->dnn_->input_layer()->weights();

  for (size_t i = 0; i < this->hidden_node_count_; ++i) {
    const float *input = &inputData.data()[batch_index * dimension];

    // for inputs in the batch.
    for (size_t j = 0; j < this->batch_size_; ++j) {
      if (j + batch_index >= inputData.vector_count()) break;
      __m128 sum = _mm_setzero_ps();

      for (size_t k = 0; k < vectorInputSize; ++k) {
        const __m128 input128 = _mm_load_ps(&input[k * 4]);
        const __m128 mul = _mm_mul_ps(input128, w[k]);
        sum = _mm_add_ps(sum, mul);
      }
      this->activations_[j * this->hidden_node_count_ + i] =
          dnn::__horizontalSumFloat32(sum);
      // advance to next input vector.
      input += dimension;
    }
    w += vectorInputSize;
  }
}

void CalculationContext::AddBias(const float *bias) {

  float *ac = this->activations_;
  for (size_t k = 0; k < this->batch_size_; k++) {
    for (size_t i = 0; i < this->hidden_node_count_; ++i) {
      // for inputs in the batch.
      ac[i] += bias[i];
    }

    // advance to the next activations.
    ac += this->hidden_node_count_;
  }
}

void CalculationContext::QuantizedSigmoid(size_t batch_index) {
  // start of the quantized activations.
  unsigned char *qStart =
      &this->quantized_activations_[batch_index * this->hidden_node_count_];

  // batch float activations pointer.
  float *currentActivations = this->activations_;

  // for all activations calculated from the input batch,
  for (size_t k = 0; k < this->batch_size_; k++) {
    if (k + batch_index >= this->input_count_) break;
    // calculate quantized sigmoid. And write the result
    for (size_t i = 0; i < this->hidden_node_count_; ++i) {
      qStart[i] = dnn::qSigmoid->get(currentActivations[i]);
      //qStart[i] = dnn::sigmoid(currentActivations[i]);
    }
    // advance the float and quantized activations.
    qStart += hidden_node_count_;
    currentActivations += hidden_node_count_;
  }
}

void CalculationContext::QuantizedLayerActivations(const QuantizedSimdLayer *layer,
                                                   size_t batch_start_index,
                                                   float *sequential_activations) {
  size_t vectorSize = layer->input_dimension() / 16;

  // get quantized weight array for the node i.
  const __m128i *w = layer->weights();

  const size_t nodeCount = layer->node_count();
  float dequantizationCoefficient =
      layer->multiplier() * dnn::SIGMOID_QUANTIZATION_MULTIPLIER;

  // for each node
  for (size_t i = 0; i < nodeCount; ++i) {
    unsigned char *input =
        &this->quantized_activations_[batch_start_index * layer->input_dimension()];

    // for inputs in the batch.
    for (size_t k = 0; k < this->batch_size_; k++) {
      if (k + batch_start_index >= this->input_count_) break;

      float sum = dnn::quantizedNodeSum(vectorSize, input, w);

      const size_t i1 = k * nodeCount + i;
      sequential_activations[i1] = sum / dequantizationCoefficient;
      input += layer->input_dimension();
    }
    w += vectorSize;
  }
}

static float inline quantizedNodeSum(const size_t vectorSize,
                                     const unsigned char *quantizedInput,
                                     const __m128i *weights) {
  // set sum to 0
  __m128i sum = _mm_setzero_si128();

  // loop for input_dimension/16 times. (Because we quantized to 1 byte)
  for (size_t j = 0; j < vectorSize; ++j) {
    // load quantized unsigned char input values.
    const __m128i
        inputVec = _mm_load_si128((__m128i *) &quantizedInput[j * 16]);
    // c = saturate(i[0]*w[0]+i[1]*w[1]), saturate(i[2]*w[2]+i[3]*w[3]),...,
    // saturate(i[14]*w[14]+i[15]*w[15])
    // c contains eight 16 bit value.
    const __m128i c = _mm_maddubs_epi16(inputVec, weights[j]);
    // unpack 4 lowest 16 bit values to 32 bits.
    const __m128i lo = _mm_cvtepi16_epi32(c);
    // unpack 4 highest 16 bit values to 32 bits.
    const __m128i hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
    // add them to sum.
    sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);
  }
  return dnn::__horizontalSumInt32(sum);
}

/* Calculates activations for a set of output nodes against a single
* input vector.
* Output node set is usually a small amount for speech recognition
* applications.
*/
float *CalculationContext::LazyOutputActivations(size_t inputIndex,
                                                 const char *outputNodes) {
  // we do this only for output.
  QuantizedSimdLayer *layer = this->dnn_->output_layer();

  const size_t vectorSize = layer->input_dimension() / 16;

  const float dequantizationCoefficient =
      layer->multiplier() * dnn::SIGMOID_QUANTIZATION_MULTIPLIER;

  float *result = this->single_output_;
  const __m128i *weights = layer->weights();
  const float *bias = layer->bias();

  // for each node
  for (size_t i = 0; i < layer->node_count(); ++i) {
    // skip if no calculation is needed for the output.
    if (outputNodes[i] == 0) {
      result[i] = 0;
      continue;
    }

    __m128i w = weights[i * vectorSize];

    // input batch start.
    unsigned char *input =
        &this->quantized_activations_[inputIndex * layer->input_dimension()];

    if (inputIndex >= this->input_count_) break;

    float sum = dnn::quantizedNodeSum(vectorSize, input, &w);

    // we set the result after dequantization and adding bias.
    result[i] = sum / dequantizationCoefficient + layer->bias()[i];
  }

  this->soft_max_->apply(result);

  return result;
}

static inline int __horizontalSumInt32(__m128i x) {
  x = _mm_hadd_epi32(x, x);
  x = _mm_hadd_epi32(x, x);
  return _mm_extract_epi32(x, 0);
}

void CalculationContext::LastHiddenLayerActivations(const BatchData &input) {
  this->dnn_->ApplyShiftAndScale(input);

  const size_t frameCount = input.vector_count();

  // calculate input layer in batches.
  for (size_t i = 0; i < frameCount; i += batch_size_) {
    InputActivations(input, i);
    AddBias(this->dnn_->input_layer()->bias());
    QuantizedSigmoid(i);
  }

  // calculate hidden layer activations, except the output.
  for (size_t j = 0; j < this->dnn_->layer_count() - 1; ++j) {
    const QuantizedSimdLayer *layer = this->dnn_->layers()[j];

    for (size_t i = 0; i < frameCount; i += batch_size_) {
      QuantizedLayerActivations(layer, i, this->activations_);
      AddBias(layer->bias());
      QuantizedSigmoid(i);
    }
  }
}

BatchData *CalculationContext::CalculateOutput() {
  // allocate for output.
  const size_t outSize = this->dnn_->output_dimension();
  float *outputs = dnn::float_alloc(this->input_count_ * outSize);

  // calculate in batches.
  for (size_t i = 0; i < this->input_count_; i += batch_size_) {
    QuantizedLayerActivations(this->dnn_->output_layer(), i, &outputs[i * outSize]);
  }

  // add bias values and calculate softMax for the output vectors.
  const float *biasArr = this->dnn_->output_layer()->bias();

  // add bias and apply soft max.
  SoftMax *softMax = new SoftMax(outSize);

  for (size_t i = 0; i < this->input_count_; i++) {
    float *out = &outputs[i * outSize];
    for (size_t j = 0; j < outSize; ++j) {
      // for inputs in the batch.
      out[j] += biasArr[j];
    }
    softMax->apply(&outputs[i * outSize]);
#ifdef DEBUG
    if (i < 30) {
      dnn::print_container(&outputs[i * outSize], 16);
    }
#endif
  }

  delete softMax;

  BatchData *result = new BatchData(outputs, this->input_count_, outSize);

  return result;
}

void FloatSimdLayer::Validate() {
  const size_t vectorInputSize = this->input_dimension_ / 4;

  __m128 *w = weights_;
  for (size_t i = 0; i < this->node_count_; ++i) {
    for (size_t j = 0; j < vectorInputSize; ++j) {
      float result[4];
      _mm_store_ps(result, w[j]);
      for (size_t k = 0; k < 4; k++) {
        if (result[k] < -20 || result[k] > 20) {
          cout << result[k] << endl;
        }
      }
    }
    w += vectorInputSize;
  }
}

QuantizedSimdLayer::QuantizedSimdLayer(const FloatLayer &floatLayer, float cutoff) {
  this->node_count_ = floatLayer.node_count();
  this->input_dimension_ = floatLayer.input_dimension();
  float maxWeight = cutoff;
  float minWeight = -cutoff;

  // find maximum absolute value in the layer
  float max = -numeric_limits<float>::max();
  for (size_t i = 0; i < floatLayer.node_count(); ++i) {
    float nodeMax = dnn::absMax(floatLayer.weights()[i], floatLayer.input_dimension(),
                                minWeight, maxWeight);
    if (nodeMax > max) {
      max = nodeMax;
    }
  }

  // find linear quantization multiplier
  this->multiplier_ = roundf(dnn::WEIGHT_MULTIPLIER / max);

  const size_t inputSimdVectorSize = floatLayer.input_dimension() / 16;

  // allocate SIMD registers for `char` valued weights. Total amount is
  // nodecount*input dim.
  this->weights_ = dnn::SIMD_i_alloc(this->node_count_ * inputSimdVectorSize);

  __m128i *w = this->weights_;
  // for each node
  for (size_t i = 0; i < this->node_count_; i++) {
    char *quantizedWeights;
    // align allocated memory for quantized Weights.
    quantizedWeights = dnn::byte_alloc(floatLayer.input_dimension());

    // 8 bit weight quantization
    for (size_t k = 0; k < floatLayer.input_dimension(); ++k) {
      float f = floatLayer.weights()[i][k];
      if (f < minWeight) {
        f = minWeight;
      }
      if (minWeight > maxWeight) {
        f = maxWeight;
      }
      quantizedWeights[k] = static_cast<char>(roundf(f * multiplier_));
    }

    // transfer char values and load to SIMD.
    for (size_t k = 0; k < inputSimdVectorSize; ++k) {
      w[k] = _mm_load_si128((const __m128i *) &quantizedWeights[k * 16]);
    }
    w += inputSimdVectorSize;
  }

  this->bias_ = new float[floatLayer.node_count()];
  // copy the bias values. We do not apply quantization.
  std::copy(floatLayer.bias(), floatLayer.bias() + floatLayer.node_count(),
            this->bias_);
}

QuantizedDnn::QuantizedDnn(const FloatDnn &floatDnn, float cutoff) {
  this->input_layer_ = new FloatSimdLayer(floatDnn.input_layer());
  this->layers_ = std::vector<QuantizedSimdLayer *>();
  this->layers_.reserve((unsigned long) (floatDnn.layer_count() - 1));

  for (size_t i = 1; i < floatDnn.layer_count(); i++) {
    dnn::QuantizedSimdLayer *layer =
        new dnn::QuantizedSimdLayer(*floatDnn.layers()[i], cutoff);
    this->layers_.push_back(layer);
  }

  this->output_layer_ = this->layers_[layers_.size() - 1];
  this->shift_ = dnn::getSimdFloat(floatDnn.shift(), floatDnn.input_dimension());
  this->scale_ = dnn::getSimdFloat(floatDnn.scale(), floatDnn.input_dimension());
}

void SoftMax::apply(float *input) {
  float total = 0;
  for (size_t i = 0; i < this->size_; ++i) {
    float d = expf(input[i]);
    this->exp_array_[i] = d;
    total += d;
  }
  for (size_t i = 0; i < this->size_; ++i) {
    input[i] = this->exp_array_[i] / total;
  }
}
}
