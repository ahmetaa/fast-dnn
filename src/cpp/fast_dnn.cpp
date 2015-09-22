#include "float_dnn.h"
#include "dnn.h"
#include "suskun_nn_QuantizedDnn.h"

dnn::FloatDnn *floatDnn;
dnn::QuantizedDnn *quantizedDnn;

JNIEXPORT void JNICALL Java_suskun_nn_QuantizedDnn_initialize
        (JNIEnv *env, jobject obj, jstring str) {
    const char *kstr = env->GetStringUTFChars(str, 0);
    floatDnn = new dnn::FloatDnn(std::string(kstr));
    quantizedDnn = new dnn::QuantizedDnn(*floatDnn);
}

// TODO: not implemented yet.
JNIEXPORT jlong JNICALL Java_suskun_nn_QuantizedDnn_getContext
        (JNIEnv *env,
         jobject obj,
         jint inputVectorCount,
         jint inputDimension,
         jint batchSize) {
}

// TODO: not implemented yet.
JNIEXPORT void JNICALL Java_suskun_nn_QuantizedDnn_calculateUntilOutput
        (JNIEnv *env, jobject obj, jlong handle, jfloatArray input) {

}

JNIEXPORT jint JNICALL Java_suskun_nn_QuantizedDnn_inputDimension
        (JNIEnv *env, jobject obj) {
    return (jint) quantizedDnn->inputDimension();
}

JNIEXPORT jint JNICALL Java_suskun_nn_QuantizedDnn_outputDimension
        (JNIEnv *env, jobject obj) {
    return (jint) quantizedDnn->outputDimension();
}

JNIEXPORT jfloatArray JNICALL Java_suskun_nn_QuantizedDnn_calculate
        (JNIEnv *env,
         jobject obj,
         jfloatArray jInputFlattened,
         jint inputVectorCount,
         jint inputDimension,
         jint batchSize) {
    dnn::BatchData batchData(env->GetFloatArrayElements(jInputFlattened, 0),
                             (int) inputVectorCount,
                             (int) inputDimension);
    dnn::CalculationContext context(quantizedDnn, inputVectorCount, batchSize);
    float *output = context.calculate(&batchData);
    int len = batchData.vectorCount * quantizedDnn->outputDimension();
    jfloatArray result = env->NewFloatArray(len);
    env->SetFloatArrayRegion(result, 0, len, output);
    return result;
}
