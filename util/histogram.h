#pragma once
#include <opencv2/opencv.hpp>

namespace dvo {
namespace util {

/// Computes the number of bins for a histogram with values 
/// in the range of [min max] and binWidth values in each bin.
int getNumberOfBins(float min, float max, float binWidth);

/// Computes the one dimensional histogram of the given data.
/// The values have to be in the range [min max].
void compute1DHistogram(const cv::Mat& data, cv::Mat& histogram,
 float min, float max, float binWidth);

float computeMedianFromHistogram(const cv::Mat& histogram,
  float min, float max);

float computeEntropyFromHistogram(const cv::Mat& histogram);

int countElementsInHistogram(const cv::Mat& histogram);

}  // namespace util
}  // namespace dvo