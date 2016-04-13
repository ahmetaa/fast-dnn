#include "float_dnn.h"
#include "dnn.h"
#include "suskun_nn_QuantizedDnn.h"

static inline dnn::QuantizedDnn *getDnn(jlong handle) {
  return reinterpret_cast<dnn::QuantizedDnn *>(handle);
}

JNIEXPORT jlong JNICALL Java_suskun_nn_QuantizedDnn_initialize
    (JNIEnv *env, jobject obj, jstring str, jfloat cutoff) {

  const char *chars = env->GetStringUTFChars(str, 0);
  const std::string modelPath(chars);
  const dnn::FloatDnn floatDnn(modelPath);
  dnn::QuantizedDnn *quantizedDnn = new dnn::QuantizedDnn(floatDnn, cutoff);
  env->ReleaseStringUTFChars(str, chars);
  return reinterpret_cast<jlong>(quantizedDnn);
}

JNIEXPORT jint JNICALL Java_suskun_nn_QuantizedDnn_inputDimension(
    JNIEnv *env, jobject obj, jlong handle) {
  return static_cast<jint> (getDnn(handle)->input_dimension());
}

JNIEXPORT jint JNICALL
Java_suskun_nn_QuantizedDnn_outputDimension(JNIEnv *env, jobject obj, jlong handle) {
  return static_cast<jint> (getDnn(handle)->output_dimension());
}

JNIEXPORT jfloatArray JNICALL Java_suskun_nn_QuantizedDnn_calculate(
    JNIEnv *env, jobject obj, jlong handle, jfloatArray jInputFlattened,
    jint inputVectorCount, jint inputDimension, jint batchSize) {

  jfloat *elements = env->GetFloatArrayElements(jInputFlattened, 0);
  const dnn::BatchData batchData(elements,
                                 static_cast<size_t>(inputVectorCount),
                                 static_cast<size_t>(inputDimension));
  dnn::QuantizedDnn *quantizedDnn = getDnn(handle);
  dnn::CalculationContext context(quantizedDnn,
                                  static_cast<size_t> (inputVectorCount),
                                  static_cast<size_t> (batchSize));
  dnn::BatchData *output = context.Calculate(batchData);

  size_t len = batchData.vector_count() * quantizedDnn->output_dimension();
  jfloatArray result = env->NewFloatArray((jsize) len);
  env->SetFloatArrayRegion(result, 0, (jsize) len, output->data());
  delete output;
  return result;
}

JNIEXPORT jlong JNICALL Java_suskun_nn_QuantizedDnn_getContext(
    JNIEnv *env, jobject obj, jlong handle, jint inputVectorCount, jint batchSize) {
  // create a new context and return it to the Java side for making subsequent
  // calls on this context.
  dnn::CalculationContext *context =
      new dnn::CalculationContext(getDnn(handle), (size_t) inputVectorCount, (size_t) batchSize);
  return reinterpret_cast<jlong>(context);
}

JNIEXPORT void JNICALL Java_suskun_nn_QuantizedDnn_calculateUntilOutput(
    JNIEnv *env, jobject obj, jlong handle, jfloatArray input) {
  // get the native context pointer from Java side.
  dnn::CalculationContext *context =
      reinterpret_cast<dnn::CalculationContext *>(handle);

  // generate the data for calculation. TODO: BatchData may not be required
  const dnn::BatchData batchData(env->GetFloatArrayElements(input, 0),
                                 context->input_count(),
                                 context->dnn()->input_dimension());
  context->CalculateUntilLastHiddenLayer(batchData);
}

JNIEXPORT jfloatArray JNICALL
Java_suskun_nn_QuantizedDnn_calculateLazy(
    JNIEnv *env, jobject obj, jlong handle, jint inputIndex,
    jbyteArray outputMask) {
  dnn::CalculationContext *context =
      reinterpret_cast<dnn::CalculationContext *>(handle);

  float *res = context->LazyOutputActivations(
      static_cast<size_t> (inputIndex),
      reinterpret_cast<char *> (env->GetByteArrayElements(outputMask, 0)));
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
  size_t k = static_cast<size_t> (layerIndex);
  if (k < 0 || k > quantizedDnn->layer_count())
    return static_cast<jint> (-1);
  if (k == 0) {
    return static_cast<jint> (quantizedDnn->input_layer()->node_count());
  }
  return static_cast<jint> (quantizedDnn->layers()[layerIndex]->node_count());
}

JNIEXPORT jint JNICALL Java_suskun_nn_QuantizedDnn_layerCount
    (JNIEnv *env, jobject obj, jlong handle) {
  dnn::QuantizedDnn *quantizedDnn = getDnn(handle);
  return static_cast<jint> (quantizedDnn->layer_count() + 1);
}