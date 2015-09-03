#include "float_dnn.h"
#include "dnn.h"
#include "handle.h"
#include "suskun_nn_FastNativeDnn.h"

dnn::FloatDnn *floatDnn;
dnn::QuantizedDnn *quantizedDnn;

JNIEXPORT void JNICALL Java_suskun_nn_FastNativeDnn_initialize
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

JNIEXPORT jfloatArray JNICALL Java_suskun_nn_FastNativeDnn_calculate
        (JNIEnv *env,
         jobject obj,
         jfloatArray jInputFlattened,
         jint inputVectorCount,
         jint inputDimension,
         jint batchSize) {
    dnn::BatchData batchData(env->GetFloatArrayElements(jInputFlattened, 0),
                             (int) inputVectorCount,
                             (int) inputDimension,
                             (int) batchSize);
    //cout << "Batch data created." << endl;
    dnn::CalculationContext context(quantizedDnn, &batchData, batchSize);
    float *output = context.calculate();
    //cout << "done calculation" << endl;
    int len = batchData.vectorCount * quantizedDnn->outputSize();
    //cout << "length is " << len << endl;
    jfloatArray result = env->NewFloatArray(len);
    env->SetFloatArrayRegion(result, 0, len, output);
    return result;
}
