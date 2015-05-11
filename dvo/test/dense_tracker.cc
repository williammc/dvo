#include <mutex>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "dvo/core/intrinsic_matrix.h"
#include "dvo/dense_tracking.h"

class RgbdDenseTracker {
  public:
    RgbdDenseTracker () :
//      viewer ("PCL OpenNI Viewer"),
      tracker_cfg(dvo::DenseTracker::getDefaultConfig()),
      frames_since_last_success(0),
      use_dense_tracking_estimate(false),
      camera_width(640),
      camera_height(480) {
      printf("CameraDenseTracker::ctor(...)");
      from_baselink_to_asus.setIdentity();

      latest_absolute_transform.setIdentity();
      accumulated_transform.setIdentity();

      gray_img.create(camera_height, camera_width, CV_32FC1);
      depth_img.create(camera_height, camera_width, CV_32FC1);
    }

    // call backs ==============================================================
    void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud) {
//      if (!viewer.wasStopped())
//        viewer.showCloud (cloud);

      convert_xyzrgb_to_images(cloud, gray_img, depth_img);
      printf("In cloud_cb_\n");
      update();
    }

//    void rgb_cb_(const  std::shared_ptr<openni_wrapper::Image>& ni_image) {
//      cv::Mat rgb_img(ni_image->getHeight(), ni_image->getWidth(), CV_8UC3);
//      ni_image->fillRGB(ni_image->getWidth(), ni_image->getHeight(), rgb_img.data);
//      cv::Mat temp;
//      cv::cvtColor(rgb_img, temp, CV_RGB2GRAY);
//      temp.convertTo(gray_img, CV_32F);
//      cv::imshow("Gray Image", gray_img);
//    }

//    void depth_cb_(const  std::shared_ptr<openni_wrapper::DepthImage>& ni_image) {
//      ni_image->fillDepthImage(ni_image->getWidth(),
//                               ni_image->getHeight(),
//                               (float*) depth_img.data);
//      update();
//    }

    void run () {
      reset();
      pcl::Grabber* interface;

//      if (0)
//        interface = new pcl::OpenNIGrabber();
//      else
        interface = new pcl::PCDGrabber<pcl::PointXYZRGB>(
                      "C:/Users/Thanh/dev/gitlab_syntactic_modeling/dvo/data/frame-20131120T183943.651999.pcd",
                      25,
                      true);

      // boost::function<void (const  boost::shared_ptr<openni_wrapper::Image>&)> pc_f = boost::bind (&RgbdDenseTracker::cloud_cb_, this, _1);

//      boost::function<void (const  boost::shared_ptr<openni_wrapper::Image>&)> rgb_f =
//          boost::bind (&RgbdDenseTracker::rgb_cb_, this, _1);

//      boost::function<void (const  boost::shared_ptr<openni_wrapper::DepthImage>&)> depth_f =
//          boost::bind (&RgbdDenseTracker::depth_cb_, this, _1);

      // interface->registerCallback (pc_f);
//      interface->registerCallback (rgb_f);
//      interface->registerCallback (depth_f);

      interface->start ();

      //       while (!viewer.wasStopped())
      //       {
      //         std::this_thread::sleep (boost::posix_time::seconds (1));
      //       }
      for (;;) {
        if (cv::waitKey( 30 ) >= 0)
          break;
      }

      interface->stop ();
    }

    void convert_xyzrgb_to_images(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
                                  cv::Mat& intensity, cv::Mat& depth) {
      const pcl::PointXYZRGB* point = &cloud->front();
      for (size_t y = 0; y < camera_height; ++y) {
        for (size_t x = 0; x < camera_width; ++x, ++point) {
          intensity.at<float>(y, x) = 0.3*(point->data[3] +
                                      point->data[4] +
                                      point->data[5]);
          depth.at<float>(y, x) = point->data[2];
        }
      }
    }

    void update() {
      printf("aaa 000\n");
      if (gray_img.empty() || depth_img.empty())
        return;
      reference.swap(current);
      current = camera->create(gray_img, depth_img);

      printf("aaa 111\n");
      static Eigen::Affine3d first;

      if (!reference) {
        accumulated_transform = latest_absolute_transform * from_baselink_to_asus;
        first = accumulated_transform;
        return;
      }

      Eigen::Affine3d transform;

      bool success = tracker->match(*reference, *current, transform);

      printf("aaa 222\n");
      if (success) {
        frames_since_last_success = 0;
        accumulated_transform = accumulated_transform * transform;

        printf("Tracking OK\n");

        Eigen::Matrix<double, 6, 6> covariance;

        //tracker->getCovarianceEstimate(covariance);

        //std::cerr << covariance << std::endl << std::endl;

      } else {
        frames_since_last_success++;
        reference.swap(current);
        printf("Tracking fail\n");
      }

      //      std::cout << "Tracked pose:\n" << accumulated_transform << std::endl;
      //      publishTransform(h, accumulated_transform * from_baselink_to_asus.inverse(), "base_link_estimate");
      //  publishTransform(rgb_image_msg->header, first_transform.inverse() * accumulated_transform, "asus_estimate");

      //      if(use_dense_tracking_estimate_)
      //      {
      //        publishPose(h, accumulated_transform * from_baselink_to_asus.inverse(), "baselink_estimate");
      //      }

    }

    void reset() {
      dvo::core::IntrinsicMatrix intrinsics = dvo::core::IntrinsicMatrix::create(
                                                5.2921508098293293e+02,  // fx
                                                5.2556393630057437e+02,  // fy
                                                3.2894272028759258e+02,  // cx
                                                2.6748068171871557e+02);  // cy

      camera.reset(new dvo::core::RgbdCameraPyramid(camera_width, camera_height, intrinsics));
      camera->build(tracker_cfg.getNumLevels());

      tracker.reset(new dvo::DenseTracker(tracker_cfg));

      static dvo::core::RgbdImagePyramid* const __null__ = 0;

      reference.reset(__null__);
      current.reset(__null__);
    }

//    pcl::visualization::CloudViewer viewer;
    size_t camera_width;
    size_t camera_height;
    cv::Mat gray_img;
    cv::Mat depth_img;

    std::shared_ptr<dvo::DenseTracker> tracker;
    dvo::DenseTracker::Config tracker_cfg;
    dvo::core::RgbdCameraPyramidPtr camera;
    dvo::core::RgbdImagePyramidPtr current, reference;

    Eigen::Affine3d accumulated_transform, from_baselink_to_asus, latest_absolute_transform;

    size_t frames_since_last_success;

    bool use_dense_tracking_estimate;
    std::mutex tracker_mutex;
};

int main () {
  RgbdDenseTracker v;
  v.run ();
  return 0;
}


