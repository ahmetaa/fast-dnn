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

    string fName = "/home/afsina/data/dnn-5-1024/dnn.model";
    //string fName = "/home/afsina/data/dnn-5-1024/dnn.model.small";
    dnn::FloatDnn floatDnn(fName);
    //string featureName = "/home/afsina/projects/suskun/feats-test";
    string featureName = "/home/afsina/projects/suskun/feats";
    dnn::BatchData batchData(featureName, 8);

    dnn::QuantizedDnn qDnn(&floatDnn);

    dnn::CalculationContext context(&qDnn, &batchData, 8);

    context.test();

    return 0;
}

namespace dnn {

    static QuantizedSigmoid *qSigmoid = new QuantizedSigmoid();

    __m128 *getSimdFloat(float *values, int dim);

    static inline float __horizontalSumFloat32(__m128 x);

    static inline int __horizontalSumInt32(__m128i x);

    const float WEIGHT_MULTIPLIER = 127;

    const float MAX_WEIGHT_THRESHOLD = 3;

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

    QuantizedSigmoid::QuantizedSigmoid() {

        int size = 1088; // table lookup size 1088=64*17 is arbitrary but can be divided to 64
        this->lookup = new unsigned char[1088];

        // when a sigmoid is quantized with 255, numbers below around -5.4 becomes 0, numbers over around +5.4 becomes 254
        for (int i = -SIGMOID_HALF_LOOKUP_SIZE; i < SIGMOID_HALF_LOOKUP_SIZE; ++i) {
            float k = i / 100.0f;
            float sigmoid = 1.0f / (1 + expf(-k));
            unsigned char q = (unsigned char) (sigmoid * dnn::SIGMOID_QUANTIZATION_MULTIPLIER);
            lookup[i + size / 2] = q;
        }
    }

    __m128 *SIMD_alloc(int simdBlockCount) {
        return (__m128 *) aligned_malloc(16, sizeof(__m128) * simdBlockCount);
    }


    __m128i *SIMD_i_alloc(int simdBlockCount) {
        return (__m128i *) aligned_malloc(16, sizeof(__m128i) * simdBlockCount);
    }

    inline char *byte_alloc(int count) {
        return (char *) aligned_malloc(16, sizeof(char) * count);
    }

    inline float *float_alloc(int count) {
        return (float *) aligned_malloc(16, sizeof(float) * count);
    }

    FloatSimdLayer::FloatSimdLayer(FloatLayer *floatLayer) {

        this->nodeCount = floatLayer->nodeCount;
        this->inputDim = floatLayer->inputDim;

        const int simdVectorDim = this->inputDim / 4;

        this->weights = dnn::SIMD_alloc(this->nodeCount * simdVectorDim);

        __m128 *w = this->weights;

        for (int i = 0; i < this->nodeCount; ++i) {
            for (int j = 0; j < simdVectorDim; ++j) {
                float *p = &floatLayer->weights[i][j * 4];
                w[j] = _mm_load_ps(p);
            }
            w += simdVectorDim;
        }

        // we do not use simd for bias.
        this->bias = new float[nodeCount];
        // copy the bias values.
        std::copy(floatLayer->bias, floatLayer->bias + floatLayer->nodeCount, this->bias);

    }

    __m128 *getSimdFloat(float *values, int dim) {

        int k = dim / 4;
        __m128 *result = dnn::SIMD_alloc(k);

        for (int i = 0; i < k; ++i) {
            result[i] = _mm_load_ps(&values[i * 4]);
        }
        return result;
    }

    float absMax(float *floats, int size, float trimMin, float trimMax) {
        float max = -numeric_limits<float>::max();
        for (int i = 0; i < size; i++) {
            float f = floats[i];
            if (f < trimMin)
                f = trimMin;
            if (f > trimMax)
                f = trimMax;
            float fAbs = (float) fabs(f);
            if (fAbs > max) {
                max = fAbs;
            }
        }
        return max;
    }

    void dump(__m128 data) {
        float temp[4] __attribute((aligned(4*4)));
        _mm_store_ps(&temp[0], data);
        print_container(&temp, 4);
    }

    void CalculationContext::test() {

        typedef std::chrono::high_resolution_clock Clock;
        typedef std::chrono::milliseconds milliseconds;

        Clock::time_point t0 = Clock::now();
        this->lastHiddenLayerActivations();
        this->calculateOutput();
        Clock::time_point t1 = Clock::now();
        milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);

        std::cout << "Elapsed:" << ms.count() << std::endl;
    }

    float *CalculationContext::calculate() {
        //cout << "Hidden Layers " << endl;
        this->lastHiddenLayerActivations();
        //cout << "Output Layer " << endl;
        BatchData *output = calculateOutput();
        //cout << "Output Calculated " << endl;
        return output->data;
    }

    static inline float __horizontalSumFloat32(__m128 x) {
        x = _mm_hadd_ps(x, x);
        x = _mm_hadd_ps(x, x);
        return _mm_cvtss_f32(x);
    }

    void QuantizedDnn::applyShiftAndScale(BatchData *batchInput) {

        // apply shift and scale with SIMD
        const int size = batchInput->dimension / 4;

        float *input = batchInput->data;

        for (int i = 0; i < batchInput->vectorCount; ++i) {

            for (int k = 0; k < size; ++k) {
                __m128 val = _mm_load_ps(&input[k * 4]);
                val = _mm_add_ps(val, this->shift[k]);
                val = _mm_mul_ps(val, this->scale[k]);
                _mm_store_ps(&input[k * 4], val);
            }
            input += batchInput->dimension;
        }
    }

    CalculationContext::CalculationContext(QuantizedDnn *dnn, BatchData *input, int batchSize) {
        this->dnn = dnn;
        this->input = input;
        this->batchSize = batchSize;
        this->hiddenNodeCount = this->dnn->layers[1].nodeCount;

        // allocate for float activations. Only batch amount.
        this->activations = dnn::float_alloc(this->hiddenNodeCount * batchSize);

        // allocate for quantized unsigned char input values.
        this->quantizedActivations = (unsigned char *) dnn::byte_alloc(this->hiddenNodeCount * input->vectorCount);
    }

    void CalculationContext::inputActivations(int batchIndex) {

        const int dimension = this->input->dimension;
        const int vectorInputSize = dimension / 4;

        // for each node.
        const __m128 *w = this->dnn->inputLayer->weights;

        for (int i = 0; i < this->hiddenNodeCount; ++i) {

            float *input = &this->input->data[batchIndex * dimension];

            // for inputs in the batch.
            for (int j = 0; j < this->batchSize; ++j) {

                __m128 sum = _mm_setzero_ps();

                for (int k = 0; k < vectorInputSize; ++k) {
                    const __m128 input128 = _mm_load_ps(&input[k * 4]);
                    const __m128 mul = _mm_mul_ps(input128, w[k]);
                    sum = _mm_add_ps(sum, mul);
                }
                this->activations[j * this->hiddenNodeCount + i] = dnn::__horizontalSumFloat32(sum);
                // advance to next input vector.
                input += dimension;
            }
            w += vectorInputSize;
        }
    }

    void CalculationContext::addBias(float *bias) {

        // add bias values to output batch
        float *biasArr = bias;

        float *ac = this->activations;
        for (int k = 0; k < this->batchSize; k++) {

            for (int i = 0; i < this->hiddenNodeCount; ++i) {
                // for inputs in the batch.
                ac[i] += biasArr[i];
            }

            // advance to the next activations.
            ac += this->hiddenNodeCount;
        }

    }

    void CalculationContext::quantizedSigmoid(int batchIndex) {

        // start of the quantized activations.
        unsigned char *qStart = &this->quantizedActivations[batchIndex * this->hiddenNodeCount];

        // batch float activations pointer.
        float *currentActivations = this->activations;

        // for all activations calculated from the input batch,
        for (int k = 0; k < this->batchSize; k++) {
            // calculate quantized sigmoid. And write the result
            for (int i = 0; i < this->hiddenNodeCount; ++i) {
                // for inputs in the batch.
                qStart[i] = dnn::qSigmoid->get(currentActivations[i]);
            }
            // advance the float and quantized activations.
            qStart += hiddenNodeCount;
            currentActivations += hiddenNodeCount;
        }
    }

    void CalculationContext::quantizedLayerActivations(
            QuantizedSimdLayer *layer,
            int batchStartIndex,
            float *activations) {

        const int vectorSize = layer->inputDim / 16;

        // get quantized weight array for the node i.
        __m128i *w = layer->weights;

        const int nodeCount = layer->nodeCount;
        const float dequantizationCoefficient = layer->multiplier * dnn::SIGMOID_QUANTIZATION_MULTIPLIER;

        // for each node
        for (int i = 0; i < nodeCount; ++i) {

            const unsigned char *input = &this->quantizedActivations[batchStartIndex * layer->inputDim];

            // for inputs in the batch.
            for (int k = 0; k < this->batchSize; k++) {

                // set sum to 0
                __m128i sum = _mm_setzero_si128();

                // loop for input_dimension/16 times. (Because we quantized to 1 byte)
                for (int j = 0; j < vectorSize; ++j) {
                    // load quantized unsigned char input values.
                    const __m128i inputVec = _mm_load_si128((__m128i *) &input[j * 16]);
                    // c = saturate(i[0]*w[0]+i[1]*w[1]), saturate(i[2]*w[2]+i[3]*w[3]),..., saturate(i[14]*w[14]+i[15]*w[15])
                    // c contains eight 16 bit value.
                    const __m128i c = _mm_maddubs_epi16(inputVec, w[j]);
                    // unpack 4 lowest 16 bit values to 32 bits.
                    const __m128i lo = _mm_cvtepi16_epi32(c);
                    // unpack 4 highest 16 bit values to 32 bits.
                    const __m128i hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
                    // add them to sum.
                    sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);
                }
                // here we sum all 32 bit integers to a single 32 bit value. then We calculate the float value of the activation.

                int i1 = k * nodeCount + i;
                activations[i1] =
                        ((float) dnn::__horizontalSumInt32(sum)) / dequantizationCoefficient;
                input += layer->inputDim;

            }
            w += vectorSize;
        }
    }

    static inline int __horizontalSumInt32(__m128i x) {
        x = _mm_hadd_epi32(x, x);
        x = _mm_hadd_epi32(x, x);
        return _mm_extract_epi32(x, 0);
    }

    void CalculationContext::lastHiddenLayerActivations() {

        this->dnn->applyShiftAndScale(this->input);

        const int frameCount = input->vectorCount;

        // calculate input layer in batches.
        for (int i = 0; i < frameCount; i += batchSize) {
            inputActivations(i);
            addBias(this->dnn->inputLayer->bias);
            quantizedSigmoid(i);
        }

        // calculate hidden layer activations, except the output.
        for (int j = 0; j < this->dnn->layerCount() - 1; ++j) {
            QuantizedSimdLayer *layer = &this->dnn->layers[j];

            for (int i = 0; i < frameCount; i += batchSize) {
                quantizedLayerActivations(layer, i, this->activations);
                addBias(layer->bias);
                quantizedSigmoid(i);
            }
        }
    }

    BatchData *CalculationContext::calculateOutput() {

        // allocate for output.
        const int outSize = this->dnn->outputSize();
        float *outputs = dnn::float_alloc(this->input->vectorCount * outSize);

        // calculate in batches.
        for (int i = 0; i < this->input->vectorCount; i += batchSize) {
            quantizedLayerActivations(this->dnn->outputLayer, i, &outputs[i * outSize]);
        }

        // add bias values and calculate softmax for the output vectors.
        const float *biasArr = this->dnn->outputLayer->bias;

        // add bias and apply soft max.
        SoftMax *softMax = new SoftMax(outSize);

        for (int i = 0; i < this->input->vectorCount; i++) {
            float *out = &outputs[i * outSize];
            for (int j = 0; j < outSize; ++j) {
                // for inputs in the batch.
                out[j] += biasArr[j];
            }
            softMax->apply(&outputs[i * outSize]);
        }
        delete softMax;

        BatchData *result = new BatchData(
                outputs,
                this->input->vectorCount,
                outSize
        );

        return result;

    }


    void FloatSimdLayer::validate() {
        const int vectorInputSize = this->inputDim / 4;

        __m128 *w = weights;
        for (int i = 0; i < this->nodeCount; ++i) {
            for (int j = 0; j < vectorInputSize; ++j) {
                float result[4];
                _mm_store_ps(result, w[j]);
                for (int k = 0; k < 4; k++) {
                    if (result[k] < -20 || result[k] > 20) {
                        cout << result[k] << endl;
                    }
                }
            }
            w += vectorInputSize;
        }
    }

    QuantizedSimdLayer::QuantizedSimdLayer(FloatLayer *floatLayer) {

        this->nodeCount = floatLayer->nodeCount;
        this->inputDim = floatLayer->inputDim;
        float maxWeight = MAX_WEIGHT_THRESHOLD;
        float minWeight = -MAX_WEIGHT_THRESHOLD;

        // find maximum absolute value in the layer
        float max = -numeric_limits<float>::max();
        for (int i = 0; i < floatLayer->inputDim; ++i) {
            float nodeMax = dnn::absMax(floatLayer->weights[i], floatLayer->inputDim, minWeight, maxWeight);
            if (nodeMax > max) {
                max = nodeMax;
            }
        }

        // find linear quantization multiplier
        this->multiplier = dnn::WEIGHT_MULTIPLIER / max;

        const int inputSimdVectorSize = floatLayer->inputDim / 16;

        //allocate SIMD registers for `char` valued weights. Total amount is nodecount*input dim.
        this->weights = dnn::SIMD_i_alloc(this->nodeCount * inputSimdVectorSize);


        __m128i *w = this->weights;
        // for each node
        for (int i = 0; i < this->nodeCount; i++) {

            char *quantizedWeights;
            // align allocated memory for quantized Weights.
            quantizedWeights = dnn::byte_alloc(floatLayer->inputDim);

            // 8 bit weight quantization
            for (int k = 0; k < floatLayer->inputDim; ++k) {
                float f = floatLayer->weights[i][k];
                if (f < minWeight) {
                    f = minWeight;
                }
                if (minWeight > maxWeight) {
                    f = maxWeight;
                }
                quantizedWeights[k] = static_cast<char> (f * multiplier);
            }

            // transfer char values and load to SIMD.
            for (int k = 0; k < inputSimdVectorSize; ++k) {
                w[k] = _mm_load_si128((const __m128i *) &quantizedWeights[k * 16]);
            }
            w += inputSimdVectorSize;
        }

        this->bias = new float[floatLayer->nodeCount];
        // copy the bias values. We do not apply quantization.
        std::copy(floatLayer->bias, floatLayer->bias + floatLayer->nodeCount, this->bias);

    }

    QuantizedDnn::QuantizedDnn(FloatDnn *floatDnn) {
        this->inputLayer = new FloatSimdLayer(floatDnn->inputLayer);
        this->layers = std::vector<QuantizedSimdLayer>();
        this->layers.reserve((unsigned long) (floatDnn->layerCount() - 1)); // we only put hidden layers

        for (int i = 1; i < floatDnn->layerCount(); i++) {
            QuantizedSimdLayer layer(&floatDnn->layers[i]);
            this->layers.push_back(layer);
        }

        this->outputLayer = &this->layers[layers.size() - 1];

        this->shift = dnn::getSimdFloat(floatDnn->shift, floatDnn->inputDimension());
        this->scale = dnn::getSimdFloat(floatDnn->scale, floatDnn->inputDimension());

    }

    void SoftMax::apply(float *input) {
        float total = 0;
        for (int i = 0; i < this->size; ++i) {
            float d = expf(input[i]);
            this->expArray[i] = d;
            total += d;
        }
        for (int i = 0; i < this->size; ++i) {
            input[i] = this->expArray[i] / total;
        }
    }
}

