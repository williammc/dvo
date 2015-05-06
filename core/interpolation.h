#pragma once
#include <opencv2/opencv.hpp>

#include <dvo/core/datatypes.h>

namespace dvo {
namespace core {

struct Interpolation {
  static IntensityType none(const cv::Mat &img, float x, float y);
  static IntensityType bilinear(const cv::Mat &img, float x, float y);
  static IntensityType bilinearWithDepthBuffer(const cv::Mat &intensity,
                                               const cv::Mat &depth,
                                               const float &x, const float &y,
                                               const float &z);
};

} // namespace core
} // namespace dvo