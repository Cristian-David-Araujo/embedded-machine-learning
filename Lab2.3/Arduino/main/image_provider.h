#include "image_provider.h"

#ifndef ARDUINO_EXCLUDE_CODE

#include "Arduino.h"
#include <TinyMLShield.h>

// Capture from the OV7670 camera at 160x120, convert to 128x128 grayscale int8
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter,
                      int image_width, int image_height,
                      int channels, int8_t* image_data) {
  static bool g_is_camera_initialized = false;

  const int input_width = 160;
  const int input_height = 120;
  const int input_size = input_width * input_height;

  static uint8_t raw_image[input_size];

  // Initialize camera if not done yet
  if (!g_is_camera_initialized) {
    if (!Camera.begin(QQVGA, GRAYSCALE, 5, OV7670)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize camera!");
      return kTfLiteError;
    }
    g_is_camera_initialized = true;
  }

  // Capture frame from camera
  Camera.readFrame(raw_image);

  // Downscale 160x120 to 128x128 using simple nearest neighbor sampling
  for (int y = 0; y < image_height; y++) {
    int src_y = (y * input_height) / image_height;
    for (int x = 0; x < image_width; x++) {
      int src_x = (x * input_width) / image_width;
      int src_index = src_y * input_width + src_x;
      int dst_index = y * image_width + x;
      image_data[dst_index] = static_cast<int8_t>(raw_image[src_index] - 128);
    }
  }

  return kTfLiteOk;
}

#endif  // ARDUINO_EXCLUDE_CODE
