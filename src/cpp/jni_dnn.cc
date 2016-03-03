#include "float_dnn.h"
#include "dnn.h"
#include "suskun_nn_QuantizedDnn.h"

static inline dnn::QuantizedDnn *getDnn(jlong handle) {
  return reinterpret_cast<dnn::QuantizedDnn *>(handle);
}

JNIEXPORT jlong JNICALL Java_suskun_nn_QuantizedDnn_initialize(JNIEnv *env,
                                                              jobject obj,
                                                              jstring str) {
  // creates a global Quantized
  const char *chars = env->GetStringUTFChars(str, 0);
  const std::string modelPath(chars);
  const dnn::FloatDnn floatDnn(modelPath);
  dnn::QuantizedDnn *quantizedDnn = new dnn::QuantizedDnn(floatDnn);
  env->ReleaseStringUTFChars(str, chars);
  return reinterpret_cast<jlong>(quantizedDnn);
}

JNIEXPORT jint JNICALL Java_suskun_nn_QuantizedDnn_inputDimension(
        JNIEnv *env, jobject obj, jlong handle) {
  return (jint)(getDnn(handle)->inputDimension());
}

JNIEXPORT jint JNICALL
Java_suskun_nn_QuantizedDnn_outputDimension(JNIEnv *env, jobject obj, jlong handle) {
  return (jint)(getDnn(handle)->outputDimension());
}

JNIEXPORT jfloatArray JNICALL Java_suskun_nn_QuantizedDnn_calculate(
    JNIEnv *env, jobject obj, jlong handle, jfloatArray jInputFlattened,
    jint inputVectorCount, jint inputDimension, jint batchSize) {
  dnn::BatchData batchData(env->GetFloatArrayElements(jInputFlattened, 0),
                           (int)inputVectorCount, (int)inputDimension);
  dnn::QuantizedDnn *quantizedDnn = getDnn(handle);
  dnn::CalculationContext context(quantizedDnn, inputVectorCount, batchSize);
  float *output = context.calculate(&batchData);
  int len = batchData.vectorCount * quantizedDnn->outputDimension();
  jfloatArray result = env->NewFloatArray(len);
  env->SetFloatArrayRegion(result, 0, len, output);
  delete output;
  return result;
}

JNIEXPORT jlong JNICALL Java_suskun_nn_QuantizedDnn_getContext(
    JNIEnv *env, jobject obj, jlong handle, jint inputVectorCount, jint batchSize) {
  // create a new context and return it to the Java side for making subsequent
  // calls on this context.
  dnn::CalculationContext *context =
      new dnn::CalculationContext(getDnn(handle), inputVectorCount, batchSize);
  return reinterpret_cast<jlong>(context);
}

JNIEXPORT void JNICALL Java_suskun_nn_QuantizedDnn_calculateUntilOutput(
    JNIEnv *env, jobject obj, jlong handle, jfloatArray input) {
  // get the native context pointer from Java side.
  dnn::CalculationContext *context =
      reinterpret_cast<dnn::CalculationContext *>(handle);

  // generate the data for calculation. TODO: BatchData may not be required
  dnn::BatchData batchData(env->GetFloatArrayElements(input, 0),
                           context->inputCount, context->dnn->inputDimension());
  context->lastHiddenLayerActivations(&batchData);
}

JNIEXPORT jfloatArray JNICALL
Java_suskun_nn_QuantizedDnn_calculateLazy(
    JNIEnv *env, jobject obj, jlong handle, jint inputIndex,
    jbyteArray outputMask) {
  dnn::CalculationContext *context =
      reinterpret_cast<dnn::CalculationContext *>(handle);

  float *res = context->lazyOutputActivations(
      (int)inputIndex, (char *)env->GetByteArrayElements(outputMask, 0));
  jsize len = env->GetArrayLength(outputMask);
  jfloatArray result = env->NewFloatArray(len);
  env->SetFloatArrayRegion(result, 0, len, res);
  return result;
}

JNIEXPORT void JNICALL Java_suskun_nn_QuantizedDnn_deleteLazyContext(
    JNIEnv *env, jobject obj, jlong handle) {
  dnn::CalculationContext *context =
      reinterpret_cast<dnn::CalculationContext *>(handle);
  delete context;
}

JNIEXPORT void JNICALL Java_suskun_nn_QuantizedDnn_delete(
    JNIEnv *env, jobject obj, jlong handle) {
  delete getDnn(handle);
}

JNIEXPORT jint JNICALL Java_suskun_nn_QuantizedDnn_layerDimension
  (JNIEnv *env, jobject obj, jlong handle, jint layerIndex) {
  dnn::QuantizedDnn *quantizedDnn = getDnn(handle);
  int k = layerIndex;
  if(k<0 || k>quantizedDnn->layerCount())
      return (jint) -1;
  if(k==0) {
      return (jint) (quantizedDnn->inputLayer->nodeCount);
  }
  return (jint)(quantizedDnn->layers[layerIndex]->nodeCount);
}

JNIEXPORT jint JNICALL Java_suskun_nn_QuantizedDnn_layerCount
        (JNIEnv *env, jobject obj, jlong handle) {
  dnn::QuantizedDnn *quantizedDnn = getDnn(handle);
  return (jint)(quantizedDnn->layerCount() + 1);
}