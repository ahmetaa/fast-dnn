#include "float_dnn.h"
#include "dnn.h"
#include "suskun_nn_QuantizedDnn.h"

extern "C" {

JNIEXPORT jlong JNICALL Java_suskun_nn_QuantizedDnn_initialize(
    JNIEnv *env,
    jobject obj,
    jstring str,
    jfloat cutoff) {
  const char *chars = env->GetStringUTFChars(str, 0);
  const std::string modelPath(chars);
  const dnn::FloatDnn floatDnn(modelPath);
  dnn::QuantizedDnn *quantizedDnn = new dnn::QuantizedDnn(floatDnn, cutoff);
  env->ReleaseStringUTFChars(str, chars);
  return reinterpret_cast<jlong>(quantizedDnn);
}

JNIEXPORT jint JNICALL Java_suskun_nn_QuantizedDnn_inputDimension(
    JNIEnv *env,
    jobject obj,
    jlong handle) {
  return static_cast<jint> (reinterpret_cast<dnn::QuantizedDnn *>(handle)->input_dimension());
}

JNIEXPORT jint JNICALL
Java_suskun_nn_QuantizedDnn_outputDimension(
    JNIEnv *env,
    jobject obj,
    jlong handle) {
  return static_cast<jint> (reinterpret_cast<dnn::QuantizedDnn *>(handle)->output_dimension());
}

JNIEXPORT jfloatArray JNICALL Java_suskun_nn_QuantizedDnn_calculate(
    JNIEnv *env,
    jobject obj,
    jlong handle,
    jfloatArray jInputFlattened,
    jint inputVectorCount,
    jint inputDimension,
    jint batchSize) {

  jfloat *elements = env->GetFloatArrayElements(jInputFlattened, NULL);
  dnn::BatchData batchData(elements,
                           static_cast<size_t>(inputVectorCount),
                           static_cast<size_t>(inputDimension), false);
  dnn::QuantizedDnn *quantizedDnn = reinterpret_cast<dnn::QuantizedDnn *>(handle);
  dnn::CalculationContext context(quantizedDnn,
                                  static_cast<size_t> (inputVectorCount),
                                  static_cast<size_t> (batchSize));
  dnn::BatchData *output = context.Calculate(batchData);

  size_t len = batchData.vector_count() * quantizedDnn->output_dimension();
  jfloatArray result = env->NewFloatArray(static_cast<jsize> (len));
  env->SetFloatArrayRegion(result, 0, static_cast<jsize>(len), output->data());
  delete output;
  env->ReleaseFloatArrayElements(jInputFlattened, elements, JNI_ABORT);
  batchData.setData(nullptr);

  return result;
}

JNIEXPORT jlong JNICALL Java_suskun_nn_QuantizedDnn_getContext(
    JNIEnv *env,
    jobject obj,
    jlong handle,
    jint inputVectorCount,
    jint batchSize) {
  // create a new context and return it to the Java side for making subsequent
  // calls on this context.
  dnn::CalculationContext *context =
      new dnn::CalculationContext(reinterpret_cast<dnn::QuantizedDnn *>(handle),
                                  static_cast<size_t> (inputVectorCount),
                                  static_cast<size_t> (batchSize));
  return reinterpret_cast<jlong>(context);
}

JNIEXPORT void JNICALL Java_suskun_nn_QuantizedDnn_calculateUntilOutput(
    JNIEnv *env,
    jobject obj,
    jlong handle,
    jfloatArray input) {
  // get the native context pointer from Java side.
  dnn::CalculationContext *context = reinterpret_cast<dnn::CalculationContext *>(handle);

  // generate the data for calculation.
  jfloat *elements = env->GetFloatArrayElements(input, NULL);
  dnn::BatchData batchData(elements,
                           context->input_count(),
                           context->dnn()->input_dimension(), false);
  context->CalculateUntilLastHiddenLayer(batchData);
  env->ReleaseFloatArrayElements(input, elements, JNI_ABORT);
  batchData.setData(nullptr);
}

JNIEXPORT jfloatArray JNICALL
Java_suskun_nn_QuantizedDnn_calculateLazy(
    JNIEnv *env,
    jobject obj,
    jlong handle,
    jint inputIndex,
    jbyteArray outputMask) {
  dnn::CalculationContext *context = reinterpret_cast<dnn::CalculationContext *>(handle);

  jbyte *mask = env->GetByteArrayElements(outputMask, NULL);

  float *res = context->LazyOutputActivations(
      static_cast<size_t> (inputIndex),
      reinterpret_cast<char *> (mask));

  jsize len = env->GetArrayLength(outputMask);
  jfloatArray result = env->NewFloatArray(len);
  env->SetFloatArrayRegion(result, 0, len, res);
  env->ReleaseByteArrayElements(outputMask, mask, JNI_ABORT);
  return result;
}

JNIEXPORT void JNICALL Java_suskun_nn_QuantizedDnn_deleteLazyContext(
    JNIEnv *env,
    jobject obj,
    jlong handle) {
  dnn::CalculationContext *context =
      reinterpret_cast<dnn::CalculationContext *>(handle);
  delete context;
}

JNIEXPORT void JNICALL Java_suskun_nn_QuantizedDnn_delete(
    JNIEnv *env,
    jobject obj,
    jlong handle) {
  delete reinterpret_cast<dnn::QuantizedDnn *>(handle);
}

JNIEXPORT jint JNICALL Java_suskun_nn_QuantizedDnn_layerDimension(
    JNIEnv *env,
    jobject obj,
    jlong handle,
    jint layerIndex) {
  dnn::QuantizedDnn *quantizedDnn = reinterpret_cast<dnn::QuantizedDnn *>(handle);
  size_t k = static_cast<size_t> (layerIndex);
  if (k < 0 || k > quantizedDnn->layer_count())
    return static_cast<jint> (-1);
  if (k == 0) {
    return static_cast<jint> (quantizedDnn->input_layer()->node_count());
  }
  return static_cast<jint> (quantizedDnn->layers()[layerIndex]->node_count());
}

JNIEXPORT jint JNICALL Java_suskun_nn_QuantizedDnn_layerCount(
    JNIEnv *env,
    jobject obj,
    jlong handle) {
  dnn::QuantizedDnn *quantizedDnn = reinterpret_cast<dnn::QuantizedDnn *>(handle);
  return static_cast<jint> (quantizedDnn->layer_count() + 1);
}
}