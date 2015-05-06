#pragma once
#include <memory>

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "dvo/core/datatypes.h"
#include "dvo/util/id_register.h"

namespace dvo {
namespace core {

/// Converts the given raw depth image (type CV_16U) to a CV_32F image rescaling
/// every pixel with the given scale
/// and replacing 0 with NaNs.
DVO_API void convertRawDepthImage(const cv::Mat &input, cv::Mat &output,
                                  float scale);

DVO_API void convertRawDepthImageSse(const cv::Mat &input, cv::Mat &output,
                                     float scale);

DVO_API void SaveImages(const cv::Mat& intensity, const cv::Mat& depth, const cv::Mat& rgb,
                       const std::string& prefix, std::string &inten_fn,
                       std::string &dep_fn, std::string &rgb_fn);

DVO_API void LoadImages(std::string inten_fn, std::string dep_fn,
                       std::string rgb_fn, cv::Mat &intensity, cv::Mat &depth,
                       cv::Mat &rgb);

using Vector8f = Eigen::Matrix<float, 8, 1>;

struct EIGEN_ALIGN16 PointWithIntensityAndDepth {
  typedef EIGEN_ALIGN16 union {
    float data[4];
    struct {
      float x, y, z;
    };
  } Point;

  typedef EIGEN_ALIGN16 union {
    float data[8];
    struct {
      float i, z, idx, idy, zdx, zdy, time_interpolation;
    };
  } IntensityAndDepth;

  typedef std::vector<PointWithIntensityAndDepth,
                      Eigen::aligned_allocator<PointWithIntensityAndDepth>>
      VectorType;

  Point point;
  IntensityAndDepth intensity_and_depth;

  Eigen::Vector4f::AlignedMapType getPointVec4f() {
    return Eigen::Vector4f::AlignedMapType(point.data);
  }

  Eigen::Vector2f::AlignedMapType getIntensityAndDepthVec2f() {
    return Eigen::Vector2f::AlignedMapType(intensity_and_depth.data);
  }

  Eigen::Vector2f::MapType getIntensityDerivativeVec2f() {
    return Eigen::Vector2f::MapType(intensity_and_depth.data + 2);
  }

  Eigen::Vector2f::MapType getDepthDerivativeVec2f() {
    return Eigen::Vector2f::MapType(intensity_and_depth.data + 4);
  }

  Vector8f::AlignedMapType getIntensityAndDepthWithDerivativesVec8f() {
    return Vector8f::AlignedMapType(intensity_and_depth.data);
  }
};

using PointCloud = Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor>;

class RgbdImage;
using RgbdImagePtr = std::shared_ptr<RgbdImage>;

class RgbdImagePyramid;
using RgbdImagePyramidPtr = std::shared_ptr<RgbdImagePyramid>;

// Rgbd Camera ================================================================
class DVO_API RgbdCamera {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  RgbdCamera(size_t width, size_t height, const Eigen::Matrix3d &projection);
  ~RgbdCamera();

  size_t width() const;
  size_t height() const;
  Eigen::Vector4f params() const;

  const Eigen::Matrix3d &projection() const;

  RgbdImagePtr create(const cv::Mat &intensity, const cv::Mat &depth) const;
  RgbdImagePtr create() const;

  void buildPointCloud(const cv::Mat &depth, PointCloud &pointcloud) const;

  // project camera plane onto image plane
  Eigen::Vector2d Project(const Eigen::Vector3d &v3) const {
    const Eigen::Vector3d v3_t = proj_ * v3;
    return Eigen::Vector2d(v3_t[0] / v3_t[2], v3_t[1] / v3_t[2]);
  }

  Eigen::Vector3d UnProject(const Eigen::Vector2d &v2) const {
    const Eigen::Vector3d v3_t = inv_proj_ * Eigen::Vector3d(v2[0], v2[1], 1.0);
    return v3_t;
  }

  const PointCloud &TemplatePointcloud() const { return pointcloud_template_; }

private:
  size_t width_, height_;

  bool hasSameSize(const cv::Mat &img) const;

  Eigen::Matrix3d proj_, inv_proj_;
  PointCloud pointcloud_template_;
};

using RgbdCameraPtr = std::shared_ptr<RgbdCamera>;

// Rgbd Camera Pyramid =========================================================
class DVO_API RgbdCameraPyramid {
public:
  RgbdCameraPyramid() {}
  RgbdCameraPyramid(const RgbdCamera &base);
  RgbdCameraPyramid(size_t base_width, size_t base_height,
                    const Eigen::Matrix3d &base_intrinsics);

  ~RgbdCameraPyramid();

  RgbdImagePyramidPtr create(const cv::Mat &base_intensity,
                             const cv::Mat &base_depth);

  void build(size_t levels);

  const RgbdCamera &level(size_t level);

  const RgbdCamera &level(size_t level) const;

private:
  std::vector<RgbdCameraPtr> levels_;
};

using RgbdCameraPyramidPtr = std::shared_ptr<RgbdCameraPyramid>;

// Rgbd Image ==================================================================
class DVO_API RgbdImage {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  RgbdImage() = delete;
  RgbdImage(const RgbdCamera &camera);
  virtual ~RgbdImage();

  using PointCloud = dvo::core::PointCloud;

  const RgbdCamera &camera() const;
  
  const cv::Mat& rgb() const { 
    if (!loaded_) const_cast<RgbdImage*>(this)->Load();
    return rgb_;
  }
  void rgb(cv::Mat img) { rgb_ = img; }
  std::string rgb_fn() const { return rgb_fn_; }
  
  const cv::Mat& intensity() const { 
    if (!loaded_) const_cast<RgbdImage*>(this)->Load();
    return intensity_; 
  }
  void intensity(cv::Mat img) { intensity_ = img; }
  std::string intensity_fn() const { return intensity_fn_; }
  
  const cv::Mat& depth() const { 
    if (!loaded_) const_cast<RgbdImage*>(this)->Load();
    return depth_;
  }
  void depth(cv::Mat img) { depth_ = img;}
  std::string depth_fn() const { return depth_fn_; }

  const cv::Mat& acceleration() const { return acceleration_; }
  
  double timestamp() const { return timestamp_; }
  void timestamp(double t) { timestamp_ = t; }

  size_t width() const { return width_; }
  size_t height() const { return height_; }

  PointCloud& pointcloud() { return pointcloud_; }
  const PointCloud& pointcloud() const { return pointcloud_; }

  bool hasIntensity() const;
  bool hasDepth() const;
  bool hasRgb() const;

  void initialize();

  void calculateDerivatives();
  bool calculateIntensityDerivatives();
  void calculateDepthDerivatives();

  void calculateNormals();

  void buildPointCloud();

  // void buildPointCloud(const Eigen::Matrix3d& intrinsics);

  void buildAccelerationStructure();

  // inverse warping
  // transformation is the transformation from reference to this image
  void warpIntensity(const AffineTransform &transformation,
                     const PointCloud &reference_pointcloud,
                     const Eigen::Matrix3d &intrinsics, RgbdImage &result,
                     PointCloud &transformed_pointcloud);

  // SSE version
  void warpIntensitySse(const AffineTransform &transformation,
                        const PointCloud &reference_pointcloud,
                        const Eigen::Matrix3d &intrinsics, RgbdImage &result,
                        PointCloud &transformed_pointcloud);
  // SSE version without warped pointcloud
  void warpIntensitySse(const AffineTransform &transformation,
                        const PointCloud &reference_pointcloud,
                        const Eigen::Matrix3d &intrinsics, RgbdImage &result);

  // forward warping
  // transformation is the transformation from this image to the reference image
  void warpIntensityForward(const AffineTransform &transformation,
                            const Eigen::Matrix3d &intrinsics,
                            RgbdImage &result, cv::Mat_<cv::Vec3d> &cloud);
  void warpDepthForward(const AffineTransform &transformation,
                        const Eigen::Matrix3d &intrinsics, RgbdImage &result,
                        cv::Mat_<cv::Vec3d> &cloud);

  void warpDepthForwardAdvanced(const AffineTransform &transformation,
                                const Eigen::Matrix3d &intrinsics,
                                RgbdImage &result);

  bool InImage(const float &x, const float &y) const;
  void ReleaseTrackingData();

  void Load();
  void Unload();

 protected:
  template <typename T>
  void calculateDerivativeX(const cv::Mat &img, cv::Mat &result);

  // template<typename T>
  // void calculateDerivativeXSse(const cv::Mat& img, cv::Mat& result);

  template <typename T>
  void calculateDerivativeY(const cv::Mat &img, cv::Mat &result);

  void calculateDerivativeYSseFloat(const cv::Mat &img, cv::Mat &result);

  enum WarpIntensityOptions {
    WithPointCloud,
    WithoutPointCloud,
  };

  template <int PointCloudOption>
  void warpIntensitySseImpl(const AffineTransform &transformation,
                            const PointCloud &reference_pointcloud,
                            const Eigen::Matrix3d &projection,
                            RgbdImage &result,
                            PointCloud &transformed_pointcloud);
  
  void DeleteUnloadedImages();

  bool loaded_, intensity_requires_calculation_, depth_requires_calculation_,
      pointcloud_requires_build_;
  std::string rgb_fn_, intensity_fn_, depth_fn_; ///< used to unload images to disk

  const RgbdCamera &camera_;

  cv::Mat intensity_;
  cv::Mat intensity_dx_;
  cv::Mat intensity_dy_;

  cv::Mat depth_;
  cv::Mat depth_dx_;
  cv::Mat depth_dy_;

  cv::Mat normals_, angles_;

  cv::Mat rgb_;

  PointCloud pointcloud_;

  using Vec8f = cv::Vec<float, 8>;
  cv::Mat_<Vec8f> acceleration_;

  size_t width_, height_;
  double timestamp_;
};

// Rgbd Image Pyramid ==========================================================
using PointcloudVec =
    std::vector<PointCloud, Eigen::aligned_allocator<PointCloud>>;

class DVO_API RgbdImagePyramid {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  RgbdImagePyramid() : id_(-1) {}

  RgbdImagePyramid(RgbdCameraPyramid &camera, const cv::Mat &intensity,
                   const cv::Mat &depth);

  RgbdImagePyramid(int id, double timestamp, const Eigen::Matrix3d &proj,
                   cv::Mat img, cv::Mat depth, short levels = 4);

  virtual ~RgbdImagePyramid();

  void build(const size_t num_levels);

  int id() const { return id_; }
  void set_id(int id) { id_ = id; }
  void register_id() { id_ = (id_ > 0) ? id_ : IdRegister::Register(); }

  double timestamp() const;
  void timestamp(double t);

  unsigned width(unsigned l = 0) const { return level(l)->width(); }
  unsigned height(unsigned l = 0) const { return level(l)->height(); }

  Eigen::Affine3d pose() const { return pose_; }

  RgbdImagePyramid &pose(const Eigen::Affine3d &pose) {
    pose_ = pose;
    return *this;
  }
  void set_pose(const Eigen::Affine3d &pose) { pose_ = pose; }

  const Eigen::Matrix3d &projection() const {
    return level(0)->camera().projection();
  }

  cv::Mat rgb(unsigned l = 0) const { return level(l)->rgb(); }

  cv::Mat intensity(unsigned l = 0) const { return level(l)->intensity(); }

  cv::Mat depth(unsigned l = 0) const { return level(l)->depth(); }

  RgbdCameraPyramid &camera() { return camera_; }
  const RgbdCameraPyramid &camera() const { return camera_; }

  RgbdImagePtr level(size_t idx) const;

  // project world point to image plane
  Eigen::Vector2d Project(const Eigen::Vector4d &v4, unsigned l = 0) const {
    const Eigen::Vector4d v4_t = pose_.matrix() * v4;
    const Eigen::Vector3d v3 = v4_t.segment<3>(0);
    return level(l)->camera().Project(v3);
  }

  // project camera plane onto image plane
  Eigen::Vector2d Project(const Eigen::Vector3d &v3, unsigned l = 0) const {
    return level(l)->camera().Project(v3);
  }

  Eigen::Vector3d UnProject(const Eigen::Vector2d &v2, unsigned l = 0) const {
    return level(l)->camera().UnProject(v2);
  }

  unsigned NumberOfLevels() const { return levels_.size(); }

  // Get 3D vertice given pixel index
  Eigen::Vector3d ComputePoint(const unsigned int pix_idx) {
    assert(pix_idx >= 0 &&
           pix_idx < camera().level(0).width() * camera().level(0).height());
    const int u = pix_idx % camera().level(0).width();
    const int v = pix_idx / camera().level(0).width();

    return UnProject(Eigen::Vector2d(u, v)) * level(0)->depth().at<float>(v, u);
  }

  // Get 3D vertice given 2D pixel location
  Eigen::Vector3d ComputePoint(const unsigned int u, const unsigned int v) {
    assert(level(0)->InImage(float(u), float(v)));

    return UnProject(Eigen::Vector2d(u, v)) * level(0)->depth().at<float>(v, u);
  }

  PointCloud GetPointcloud() const {
    PointCloud pointcloud;
    camera().level(0).buildPointCloud(level(0)->depth(), pointcloud);
    return pointcloud;
  }

  PointcloudVec &TemplatePointcloudVec() const;

  void Unload(); ///< unload images to disk
  void Load();   ///< load images to memory
  void ReleaseTrackingData();

protected:
  int id_;
  double timestamp_;
  Eigen::Affine3d pose_;
  RgbdCameraPyramid camera_;
  std::vector<RgbdImagePtr> levels_;

public:
  // for serialization
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar &id_;
    ar &timestamp_;
    ar &pose_;

    int w;
    int h;
    Eigen::Matrix3d proj;
    int levels;
    std::string inten_fn, dep_fn, rgb_fn;
    cv::Mat inten, dep;
    if (Archive::is_saving::value) {
      Load();
      w = width();
      h = height();
      proj = projection();

      levels = levels_.size();

      const std::string prefix = std::string("frame_serialized-") +
                                 std::to_string(timestamp_) + "-" +
                                 std::to_string(id_) + "-";
      SaveImages(intensity(), depth(), rgb(), prefix,
        inten_fn, dep_fn, rgb_fn);
      // inten = intensity();
      // dep = depth();
    }

    ar &w;
    ar &h;
    ar &proj;
    ar &levels;

    ar& inten_fn;
    ar& dep_fn;
    ar& rgb_fn;

    // ar &inten;
    // ar &dep;

    if (Archive::is_loading::value) {
      cv::Mat inten, dep, rgb;
      dvo::core::LoadImages(inten_fn, dep_fn, rgb_fn, inten, dep, rgb);

      camera_ = RgbdCameraPyramid(w, h, proj);
      camera_.build(levels);
      levels_.push_back(camera_.level(0).create(rgb, dep));
      level(0)->timestamp(timestamp_);
      build(levels);
    }
  }
};

} // namespace core
} // namespace dvo