#pragma once
#include <Eigen/Core>
#include <opencv2/core/core.hpp>

namespace dvo {

inline void generate_warped_images(
  cv::Mat intensity, cv::Mat depth,
  const Eigen::Matrix3d& projection, const Eigen::Matrix4d& ref2gen,
  cv::Mat& warped_intensity, cv::Mat& warped_depth) {
  warped_intensity.create(intensity.rows, intensity.cols, CV_32FC1);
  warped_intensity.setTo(cv::Scalar(0));
  warped_depth.create(intensity.rows, intensity.cols, CV_32FC1);
  warped_depth.setTo(cv::Scalar(0));

  Eigen::Matrix3d inv_proj = projection.inverse();
  auto get_point = [&](int x, int y, float d) {
    Eigen::Vector3d v = inv_proj * Eigen::Vector3d(x, y, 1.0) * d;
    return Eigen::Vector4d(v[0], v[1], v[2], 1.0);
  };

  auto in_image = [&](int x, int y) {
    return (x >= 0 && x < intensity.cols && y >= 0 && y < intensity.rows);
  };

  int i = 0;
  for (int r = 0; r < intensity.rows; ++r) {
    for (int c = 0; c < intensity.cols; ++c, ++i) {
      float d = depth.at<float>(r, c);
      if (d == 0.0f) continue;
      Eigen::Vector4d v4 = ref2gen * get_point(c, r, d);
      Eigen::Vector3d v3 = projection * Eigen::Vector3d(v4[0], v4[1], v4[2]);
      const Eigen::Vector2d v2(v3[0] / v3[2], v3[1] / v3[2]);
      if (in_image(v2[0], v2[1])) {
        const cv::Point pn(v2[0], v2[1]);
        warped_intensity.at<float>(pn) = intensity.at<float>(r, c);
        warped_depth.at<float>(pn) = v3[2];
      }
    }
  }
}
}  // namespace dvo