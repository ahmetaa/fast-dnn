#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS
#endif

#include <iostream>
#include <vector>
#include "dnn.h"
#include <chrono>
#include <fstream>

using namespace std;

namespace dnn {

size_t paddedSize(size_t num, size_t div);

FloatDnn::FloatDnn(std::string fileName) {
  BinaryLoader *loader = new BinaryLoader(fileName, false);

  size_t layerCount = loader->load_size_t();

  this->layers_ = std::vector<FloatLayer *>(layerCount);
  size_t actualInputDimension = 0;

  for (size_t j = 0; j < layerCount; j++) {
    size_t inputDimension = loader->load_size_t();
    if (j == 0) {
      actualInputDimension = inputDimension;
    }
    // make sure input is a factor of 4 for the first layer
    size_t paddedInputDim =
        j == 0 ? dnn::paddedSize(inputDimension, 4) : inputDimension;
    size_t outputDimension = loader->load_size_t();

    // load weights
    float **weights = new float *[outputDimension];
    for (size_t oo = 0; oo < outputDimension; oo++) {
      weights[oo] = new float[paddedInputDim];
      for (size_t ii = 0; ii < paddedInputDim; ii++) {
        float d = ii < inputDimension ? loader->load_float() : 0;
        weights[oo][ii] = d;
      }
    }

    // load bias values.
    float *bias = new float[outputDimension];
    for (size_t ii = 0; ii < outputDimension; ii++) {
      bias[ii] = loader->load_float();
    }

    dnn::FloatLayer *layer =
        new dnn::FloatLayer(weights, bias, paddedInputDim, outputDimension);
    layers_[j] = layer;
  }

  // set input layer reference.
  this->input_layer_ = layers_[0];

  // load shift vector. This is added to input vector.
  this->shift_ =
      loader->load_float_array(actualInputDimension, this->input_layer_->input_dimension());

  // load scale vector. This is multiplied to input vector.
  this->scale_ =
      loader->load_float_array(actualInputDimension, this->input_layer_->input_dimension());

  delete loader;
}

void FloatDnn::PrintTopology() {
  std::cout << input_layer()->input_dimension() << "-" << layers_.size() - 2 << "x"
      << layers_[0]->node_count() << "-" << layers_[layers_.size() - 1]->node_count() << std::endl;
}


size_t paddedSize(size_t num, size_t div) {
  size_t dif = div - num % div;
  if (dif == div)
    return num;
  else
    return num + dif;
}

BatchData::BatchData(const std::string fileName) {
  BinaryLoader *loader = new BinaryLoader(fileName, false);

  size_t frameCount = loader->load_size_t();
  this->dimension_ = loader->load_size_t();
  this->vector_count_ = frameCount;

  this->data_ = new float[this->dimension_ * frameCount]();

  int t = 0;
  for (size_t j = 0; j < frameCount; ++j) {
    for (size_t k = 0; k < this->dimension_; ++k) {
      float d = loader->load_float();
      this->data_[t] = d;
      t++;
    }
  }

  delete loader;
}

BatchData::BatchData(float *input, size_t vectorCount, size_t dimension) {
  this->data_ = input;
  this->vector_count_ = vectorCount;
  this->dimension_ = dimension;
}

void BatchData::dump() {
  float *p = data_;
  for (size_t i = 0; i < vector_count_; ++i) {
    for (size_t j = 0; j < dimension_; ++j) {
      printf("%f", *p);
      if (j < dimension_ - 1) {
        cout << " ";
      }
      p++;
    }
    cout << endl;
  }
}

void BatchData::dumpToFile(std::string fileName, bool binary) {
  ofstream os;
  if (binary) {
    os.open(fileName, ios::binary | ios::out);
  } else {
    os.open(fileName);
  }

  if (!os.is_open()) {
    cout << "Cannot open file " << fileName << endl;
    return;
  }
  if (binary) {
    const unsigned int v = static_cast<unsigned int>(vector_count_);
    const unsigned int d = static_cast<unsigned int>(dimension_);
    os.write(reinterpret_cast<const char *>(&v), sizeof(int));
    os.write(reinterpret_cast<const char *>(&d), sizeof(int));
  }
  float *p = data_;
  for (size_t i = 0; i < vector_count_; ++i) {
    for (size_t j = 0; j < dimension_; ++j) {
      if (binary) {
        os.write(reinterpret_cast<const char *>(p), sizeof(float));
      } else {
        os << *p;
      }
      if (!binary && (j < dimension_ - 1)) {
        os << " ";
      }
      p++;
    }
    if (!binary) {
      os << endl;
    }
  }
  os.close();
}

BinaryLoader::BinaryLoader(const std::string fileName, bool littleEndian) {

  this->four_bytes_ = new char[4];
  this->eight_bytes_ = new char[8];

  FILE *pFile = fopen(fileName.c_str(), "rb");

  // obtain file size:
  fseek(pFile, 0, SEEK_END);
  long lSize = ftell(pFile);
  rewind(pFile);

  // allocate memory to contain the whole file:
  this->content_ = new char[lSize];
  this->length_ = static_cast<int> (lSize);
  this->little_endian_ = littleEndian;

  // copy the file into the buffer:
  size_t result = fread(this->content_, 1, (size_t) lSize, pFile);
  if (static_cast<long> (result) != lSize) {
    fputs("Reading error", stderr);
    exit(3);
  }
  // terminate
  fclose(pFile);
}

// loads 4 byte content, converts to little endian representation if necessary.
// resulting byte array is [size] bytes. [size] can only be 4 or 8 bytes.
char *BinaryLoader::loadFourBytes(int size) {
  assert(offset_ < length_);
  assert(size >= 4 || size == 8);

  char *bytes = size == 4 ? four_bytes_ : eight_bytes_;
  std::fill(bytes, bytes + size, 0);

  for (size_t i = 0; i < 4; ++i) {
    char c = content_[offset_ + i];
    if (little_endian_) {
      bytes[i] = content_[offset_ + i];
    } else {
      bytes[3 - i] = c;
    }
  }
  offset_ = offset_ + 4;
  return bytes;
}
}

