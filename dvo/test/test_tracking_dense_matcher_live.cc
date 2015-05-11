#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "sense/tracking_events.h"
#include "sense/openni2_capture.h"
#include "sense/util.h"

#include "dvo/dense_tracking.h"
#include "dvo/test/helpers.h"

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
  sense::OpenNI2Capture oni;

  sense::SenseSubscriber sense_rec;
  sense_rec.Register(oni);

  int camera_width = 640;
  int camera_height = 480;

  Eigen::Matrix3d projection;
  projection << 5.2921508098293293e+02, 0.0f, 3.2894272028759258e+02,
                0.0f, 5.2556393630057437e+02, 2.6748068171871557e+02,
                0.0f, 0.0f, 1.0f;

  dvo::DenseTracker::Config tracker_cfg = dvo::DenseTracker::getDefaultConfig();
  std::shared_ptr<dvo::DenseTracker> tracker(new dvo::DenseTracker(tracker_cfg));

  dvo::core::RgbdCameraPyramidPtr camera(new dvo::core::RgbdCameraPyramid(camera_width, camera_height, projection));
  camera->build(tracker_cfg.getNumLevels());
  
  dvo::core::RgbdImagePyramidPtr reference, current;

  bool stop = false;
  bool pausing = false;
  int index = 0;    

  Eigen::Affine3d ref2cur;
  ref2cur.setIdentity();

  while (!stop) {
    sense::EventPtr evt;
    while (!sense_rec.empty())
      evt = sense_rec.pop();

    auto cdimgevt = std::dynamic_pointer_cast<sense::ColorAndDepthEvent>(evt);
    if (!pausing && cdimgevt) {printf("000 \n");
      if (cdimgevt->frame().empty())
        continue;
      cv::Mat gray_img;
      cv::cvtColor(cdimgevt->frame(), gray_img, CV_BGR2GRAY);
      cv::Mat intensity;
      gray_img.convertTo(intensity, CV_32F);
      cv::Mat depth = cdimgevt->depth_frame();
      depth.convertTo(depth, CV_32F);
      depth = depth * 0.001;  // to meter unit

      current = camera->create(intensity, depth);

      if (!reference) {
        reference = current;
        continue;
      }
printf("111 \n");
      dvo::DenseTracker::Result res;
      bool success = tracker->match(*reference, *current, res);
      ref2cur = res.Transformation.inverse();

      std::cout << "TrackResult:" << dvo::DenseTracker::Result::ToString(res.Status) << std::endl;
      std::cout << "Tracked pose:\n" << ref2cur.matrix() << std::endl;
      
      Matrix34f proj = (projection * ref2cur.matrix().block<3, 4>(0, 0)).cast<float>();
      cv::Mat bgr;
      cv::cvtColor(cdimgevt->frame(), bgr, CV_RGB2BGR);
      draw_target(bgr, proj);


      static cv::Mat hist_img;
      sense::GenerateHistogramImage(cdimgevt->depth_frame(), hist_img);

      static cv::Mat depth_color_img;
      sense::CombineMatrixes(hist_img, bgr, depth_color_img);
      cv::imshow("Dense Matcher Live", depth_color_img);
    }
    
    auto const c = cv::waitKey(20);

    switch (c) {
    case 27:
    stop = true;
    break;
    case 'r':
    {
      ref2cur.setIdentity();
      reference = current;
      break;
    }
    case 'p':
    pausing = !pausing;
    break;
    }
  }
}