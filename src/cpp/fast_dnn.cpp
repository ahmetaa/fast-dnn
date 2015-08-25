#include "float_dnn.h"
#include "dnn.h"
#include "handle.h"
#include "suskun_nn_FastNativeDnn.h"

dnn::FloatDnn *floatDnn;
dnn::QuantizedDnn *quantizedDnn;

void Java_suskun_nn_FastNativeDnn_initialize
        (JNIEnv *env, jobject obj, jstring str) {
    const char *kstr = env->GetStringUTFChars(str, 0);
    floatDnn = new dnn::FloatDnn(std::string(kstr));
    quantizedDnn = new dnn::QuantizedDnn(floatDnn);
}

dnn::CalculationContext getContext(float *input, int frameCount, int dimension, int batchAmount) {
    dnn::BatchData batchData(input, frameCount, dimension, batchAmount);
    dnn::CalculationContext context(quantizedDnn, &batchData, batchAmount);
    return context;
}

jfloatArray Java_suskun_nn_FastNativeDnn_calculate
        (JNIEnv *env,
         jobject obj,
         jfloatArray jInputFlattened,
         jint inputVectorCount,
         jint inputDimension,
         jint batchSize) {
    dnn::BatchData batchData(env->GetFloatArrayElements(jInputFlattened, false),
                             (int) inputVectorCount,
                             (int) inputDimension,
                             (int) batchSize);
    dnn::CalculationContext context(quantizedDnn, &batchData, batchSize);
    float *output = context.calculate();

    int len = batchData.vectorCount * quantizedDnn->outputSize();
    jfloatArray result = env->NewFloatArray(len);
    env->SetFloatArrayRegion(result, 0, len, output);
    return result;
}
