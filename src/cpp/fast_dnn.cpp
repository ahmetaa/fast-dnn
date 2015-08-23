#include "float_dnn.h"
#include "dnn.h"
#include "handle.h"
#include "suskun_nn_FastNativeDnn.h"

dnn::FloatDnn *floatDnn;
dnn::QuantizedDnn *quantizedDnn;

void Java_suskun_nn_FastNativeDnn_initialize
(JNIEnv *env, jobject obj, jstring str) {
    const char *kstr = env->GetStringUTFChars(str, 0);
    cout<<kstr<<" huzzah!";
}

void *initialize(std::string fileName) {
    floatDnn = new dnn::FloatDnn(fileName);
    quantizedDnn = new dnn::QuantizedDnn(floatDnn);
}

dnn::CalculationContext getContext( float* input, int frameCount, int dimension, int batchAmount ) {
    dnn::BatchData batchData(input, frameCount, dimension, batchAmount);
    dnn::CalculationContext context(quantizedDnn, &batchData, batchAmount);
    return context;
}

float* calculate(dnn::CalculationContext context, float* input) {
    context.calculate(input);
    return nullptr;
}