#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "dvo/dense_tracking.h"
#include "dvo/test/helpers.h"

int main () {
  int camera_width = 640;
  int camera_height = 480;
  Eigen::Matrix3d projection;
  projection << 5.2921508098293293e+02, 0.0f, 3.2894272028759258e+02,
                0.0f, 5.2556393630057437e+02, 2.6748068171871557e+02,
                0.0f, 0.0f, 1.0f;

  const std::string data_path = std::string(DVO_ROOT) + "/../data/rgbd_sequence";
  const std::string color_fn = std::string(DVO_ROOT) + 
                         "/../data/rgbd_sequence/kinect_recorder_000000-color.png"; 
  const std::string depth_fn = std::string(DVO_ROOT) + 
                         "/../data/rgbd_sequence/kinect_recorder_000000-depth.png"; 
  cv::Mat color = cv::imread(color_fn);
  cv::Mat gray_img;
  cv::cvtColor(color, gray_img, CV_BGR2GRAY);
  cv::Mat intensity;
  gray_img.convertTo(intensity, CV_32F);
  cv::Mat depth = cv::imread(depth_fn, -1);
  depth.convertTo(depth, CV_32F);
  depth = depth * 0.001;  // to meter unit

  Eigen::Matrix4d ref2gen;
  ref2gen.setIdentity();
  ref2gen(0, 3) = 0.1;
  ref2gen(1, 3) = 0.01;

  // generate 'warped' view for testing
  cv::Mat warped_intensity, warped_depth;
  dvo::generate_warped_images(intensity, depth, projection, ref2gen,
                         warped_intensity, warped_depth);

  dvo::DenseTracker::Config tracker_cfg = dvo::DenseTracker::getDefaultConfig();
  std::shared_ptr<dvo::DenseTracker> tracker(new dvo::DenseTracker(tracker_cfg));

  dvo::core::RgbdCameraPyramidPtr camera(new dvo::core::RgbdCameraPyramid(camera_width, camera_height, projection));
  camera->build(tracker_cfg.getNumLevels());

  auto reference = camera->create(intensity, depth);
  auto current = camera->create(warped_intensity, warped_depth);

  Eigen::Affine3d ref2gen_est;
  ref2gen_est.setIdentity();
  bool success = tracker->match(*reference, *current, ref2gen_est);

  Eigen::Matrix<double, 3, 4> gt_pose = ref2gen.matrix().block<3, 4>(0, 0);
  auto printf_diff = [&]() {
    std::cout << "Groundtruth pose:\n" << gt_pose << std::endl;
    std::cout << "Align() pose:\n" << ref2gen_est.matrix() << std::endl;
    std::cout << "Rotation Diff to groudtruth:"
      << (gt_pose.block<3, 3>(0, 0) - ref2gen_est.matrix().block<3, 3>(0, 0)).norm()
      << std::endl;
    const Eigen::Vector3d gt = gt_pose.block(0, 3, 3, 1);
    const Eigen::Vector3d et = ref2gen_est.matrix().block(0, 3, 3, 1);
    std::cout << "Gt translate:" << gt.transpose()
      << " estimated translate:" << et.transpose() << std::endl;
    std::cout << "Translation Diff to groudtruth:" << (gt - et).norm()
              << std::endl;
  };

  printf_diff();
  return 0;
}


