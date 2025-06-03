#include <Arduino.h>
#include <TinyMLShield.h>

#include "model.h"               // Your TFLite model header
#include "image_provider.h"      // Image capture interface

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

namespace {
  constexpr int kTensorArenaSize = 64 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  tflite::ErrorReporter* error_reporter = nullptr;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model
  const tflite::Model* model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version mismatch!");
    while (1);
  }

  // Register operators (add ops used by your model)
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
  resolver.AddSoftmax();
  resolver.AddReshape();

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded and ready.");
}

void loop() {
  // Get image into input tensor
  TfLiteStatus status = GetImage(error_reporter, input->dims->data[1], input->dims->data[2], input->dims->data[3], input->data.int8);
  if (status != kTfLiteOk) {
    error_reporter->Report("Failed to get image");
    delay(1000);
    return;
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    delay(1000);
    return;
  }

  int output_len = output->bytes;
  int8_t* out_data = output->data.int8;

  Serial.print("INFERENCE:");
  for (int i = 0; i < output_len; i++) {
    Serial.print(out_data[i]);
    if (i < output_len - 1) Serial.print(',');
  }
  Serial.println();

  delay(1000);  // Wait before next frame
}
