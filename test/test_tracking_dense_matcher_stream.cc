#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "dvo/dense_tracking.h"
#include "helpers.h"

using Matrix34f = Eigen::Matrix < float, 3, 4 > ;
using Vec4f = Eigen::Matrix < float, 4, 1 >;
using Vec3f = Eigen::Matrix < float, 3, 1 >;

// Draws desirable target in world coordinate to current color image
void draw_target(cv::Mat& rgb_img, const Matrix34f& mvp) {
  const Vec4f point_target(0, 0, 1, 1);
  Vec3f point_cam = mvp * point_target;
  point_cam /= point_cam[2];
  Vec3f pointx_cam = mvp * (point_target + Vec4f(0.1, 0, 0, 0));
  pointx_cam /= pointx_cam[2];
  Vec3f pointy_cam = mvp * (point_target + Vec4f(0, 0.1, 0, 0));
  pointy_cam /= pointy_cam[2];
  Vec3f pointz_cam = mvp * (point_target + Vec4f(0, 0, 0.1, 0));
  pointz_cam /= pointz_cam[2];
  cv::line(rgb_img, cv::Point(point_cam[0], point_cam[1]),
           cv::Point(pointx_cam[0], pointx_cam[1]), cv::Scalar(0, 0, 255), 3);
  cv::line(rgb_img, cv::Point(point_cam[0], point_cam[1]),
           cv::Point(pointy_cam[0], pointy_cam[1]), cv::Scalar(0, 255, 0), 3);
  cv::line(rgb_img, cv::Point(point_cam[0], point_cam[1]),
           cv::Point(pointz_cam[0], pointz_cam[1]), cv::Scalar(255, 0, 0), 3);
}

int main () {
  int camera_width = 640;
  int camera_height = 480;

  Eigen::Matrix3d projection;
  projection << 5.2921508098293293e+02, 0.0f, 3.2894272028759258e+02,
                0.0f, 5.2556393630057437e+02, 2.6748068171871557e+02,
                0.0f, 0.0f, 1.0f;
#if 1
  const std::string path_prefix = std::string(DVO_ROOT) +
    "/data/rgbd_sequence";
  std::string filename = path_prefix + "/kinect_recorder.txt";
#elif 0
  const std::string path_prefix = std::string(DVO_ROOT) +
    "/data/rgbd_sequence_long";
  std::string filename = path_prefix + "/kinect_recorder.txt";
#else
  const std::string path_prefix = "d:/data/structural_modeling/groundtruth/living_room_traj2_loop";
  std::string filename = path_prefix + "/scene.txt";
#endif
  std::ifstream ifs(filename);
  if (ifs.fail()) {
    printf("Fail to open file: %s\n", filename.c_str());
    return -1;
  }

  std::vector<std::string> color_fns, depth_fns;
  while (!ifs.eof()) {
    std::string tag;
    double timestamp;
    std::string cfn, dfn;
    bool aligned;
    ifs >> tag >> timestamp >> cfn >> dfn >> aligned;
    color_fns.push_back(cfn);
    depth_fns.push_back(dfn);
    printf("Read line color image:%s, depth image:%s\n", cfn.c_str(), dfn.c_str());
  }
  
  const std::string color_fn = path_prefix + "/" + color_fns[0];
  const std::string depth_fn = path_prefix + "/" + depth_fns[0];
  cv::Mat color = cv::imread(color_fn);
  cv::Mat gray_img;
  cv::cvtColor(color, gray_img, CV_BGR2GRAY);
  cv::Mat intensity;
  gray_img.convertTo(intensity, CV_32F);
  cv::Mat depth = cv::imread(depth_fn, -1);
  depth.convertTo(depth, CV_32F);
  depth = depth * 0.001;  // to meter unit

  dvo::DenseTracker::Config tracker_cfg = dvo::DenseTracker::getDefaultConfig();
  std::shared_ptr<dvo::DenseTracker> tracker(new dvo::DenseTracker(tracker_cfg));

  dvo::core::RgbdCameraPyramidPtr camera(new dvo::core::RgbdCameraPyramid(camera_width, camera_height, projection));
  camera->build(tracker_cfg.getNumLevels());

  auto reference = camera->create(intensity, depth);

  bool stop = false;
  int index = 0;    

  Eigen::Affine3d ref2cur;
  ref2cur.setIdentity();

  while (!stop) {
    index = (index == color_fns.size() - 1) ? 0 : index + 1;  // loopback
    //if (index == 0) {
    //  ref2cur.setIdentity();
    //}
    //index = 0;
    const std::string color_fn = path_prefix + "/" + color_fns[index];
    const std::string depth_fn = path_prefix + "/" + depth_fns[index];
    cv::Mat color = cv::imread(color_fn);
    cv::Mat gray_img;
    cv::cvtColor(color, gray_img, CV_BGR2GRAY);
    cv::Mat intensity;
    gray_img.convertTo(intensity, CV_32F);
    cv::Mat depth = cv::imread(depth_fn, -1);
    depth.convertTo(depth, CV_32FC1);
    depth = depth * 0.001;  // to meter unit

    auto current = camera->create(intensity, depth);

    current->Unload();
    // current->Load();

    dvo::DenseTracker::Result res;
    bool success = tracker->match(*reference, *current, res);
    ref2cur = res.Transformation.inverse();

    std::cout << "TrackResult:" << dvo::DenseTracker::Result::ToString(res.Status) << std::endl;
    std::cout << "Tracked pose:\n" << ref2cur.matrix() << std::endl;
    
    Matrix34f proj = (projection * ref2cur.matrix().block<3, 4>(0, 0)).cast<float>();
    cv::Mat bgr_img = color.clone();
    draw_target(bgr_img, proj);

    cv::imshow("Dense Matcher", bgr_img);
    auto const c = cv::waitKey(10);
    switch (c) {
    case 27:
    stop = true;
    break;
    case 'r':
    ref2cur.setIdentity();
    break;
    }
  }

  return 0;
}