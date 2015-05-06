#include <assert.h>
#ifdef WIN32
#include <windows.h>
#endif
#include <dvo/core/rgbd_image.h>
#include <dvo/core/interpolation.h>

namespace dvo {
namespace core {

#ifdef WIN32
std::string ExePath() {
    char buffer[MAX_PATH];
    GetModuleFileName( NULL, buffer, MAX_PATH );
    std::string::size_type pos = std::string( buffer ).find_last_of( "\\/" );
    return std::string( buffer ).substr( 0, pos);
}
#else
std::string ExePath() {
    return "/tmp";
}
#endif

/// Converts the given raw depth image (type CV_16UC1) to a CV_32FC1 image rescaling every pixel with the given scale
/// and replacing 0 with NaNs.
void convertRawDepthImage(const cv::Mat& input, cv::Mat& output, float scale) {
  output.create(input.rows, input.cols, CV_32FC1);

  const unsigned short* input_ptr = input.ptr<unsigned short>();
  float* output_ptr = output.ptr<float>();

  for(int idx = 0; idx < input.size().area(); idx++, input_ptr++, output_ptr++)
  {
    if(*input_ptr == 0)
    {
      *output_ptr = std::numeric_limits<float>::quiet_NaN();
    }
    else
    {
      *output_ptr = ((float) *input_ptr) * scale;
    }
  }
}

void SaveImages(const cv::Mat& in_intensity, 
  const cv::Mat& in_depth, const cv::Mat& in_rgb,
                       const std::string& prefix, std::string &inten_fn,
                       std::string &dep_fn, std::string &rgb_fn) {
  auto intensity = in_intensity.clone();
  auto depth = in_depth.clone();
  auto rgb = in_rgb.clone();
  inten_fn = (inten_fn.empty()) ? prefix + "-intensity.png" : inten_fn;
  if (intensity.type() == CV_32FC1)
    intensity.convertTo(intensity, CV_8UC1);
  if (intensity.type() != CV_8UC1) {
    std::cerr << "SaveImages intensity.type() is not 8bit pixels\n";
    abort();
  }
  cv::imwrite(inten_fn, intensity);

  dep_fn = (dep_fn.empty()) ? prefix + "-depth.png" : dep_fn;
  if (depth.type() == CV_32FC1) {
    depth = depth * 1000;
    depth.convertTo(depth, CV_16UC1);
  }

  if (depth.type() != CV_16UC1 && depth.type() != CV_16SC1) {
    std::cerr << "SaveImages depth.type() is not 16bit pixels\n";
    abort();
  }
  cv::imwrite(dep_fn, depth);

  if (!rgb.empty()) {
    if (rgb.type() != CV_8UC3) {
      std::cerr << "SaveImages RGB.type() is not 24bit pixels\n";
      abort();
    }
    rgb_fn = (rgb_fn.empty()) ? prefix + "-rgb.png" : rgb_fn;
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, CV_RGB2BGR);
    cv::imwrite(rgb_fn, bgr);
  }
}

void LoadImages(std::string inten_fn, std::string dep_fn,
                       std::string rgb_fn, cv::Mat &intensity, cv::Mat &depth,
                       cv::Mat &rgb) {
  intensity = cv::imread(inten_fn, -1);
  depth = cv::imread(dep_fn, -1);

  if (!rgb_fn.empty()) {
    rgb = cv::imread(rgb_fn, -1);
    cv::cvtColor(rgb, rgb, CV_BGR2RGB);
  }
  if (rgb.empty())
    cv::cvtColor(intensity, rgb, CV_GRAY2RGB);
}

template<typename T>
static void pyrDownMeanSmooth(const cv::Mat& in, cv::Mat& out) {
  out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

  for(int y = 0; y < out.rows; ++y) {
    for(int x = 0; x < out.cols; ++x) {
      int x0 = x * 2;
      int x1 = x0 + 1;
      int y0 = y * 2;
      int y1 = y0 + 1;

      out.at<T>(y, x) = (T) ( (in.at<T>(y0, x0) + in.at<T>(y0, x1) + in.at<T>(y1, x0) + in.at<T>(y1, x1)) / 4.0f );
    }
  }
}

template<typename T>
static void pyrDownMeanSmoothIgnoreInvalid(const cv::Mat& in, cv::Mat& out) {
  out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

  for(int y = 0; y < out.rows; ++y) {
    for(int x = 0; x < out.cols; ++x) {
      int x0 = x * 2;
      int x1 = x0 + 1;
      int y0 = y * 2;
      int y1 = y0 + 1;

      T total = 0;
      int cnt = 0;

      if(std::isfinite(in.at<T>(y0, x0))) {
        total += in.at<T>(y0, x0);
        cnt++;
      }

      if(std::isfinite(in.at<T>(y0, x1))) {
        total += in.at<T>(y0, x1);
        cnt++;
      }

      if(std::isfinite(in.at<T>(y1, x0))) {
        total += in.at<T>(y1, x0);
        cnt++;
      }

      if(std::isfinite(in.at<T>(y1, x1))) {
        total += in.at<T>(y1, x1);
        cnt++;
      }

      if(cnt > 0) {
        out.at<T>(y, x) = (T) ( total / cnt );
      } else {
        out.at<T>(y, x) = InvalidDepth;
      }
    }
  }
}

template<typename T>
static void pyrDownMedianSmooth(const cv::Mat& in, cv::Mat& out) {
  out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

  cv::Mat in_smoothed;
  cv::medianBlur(in, in_smoothed, 3);

  for(int y = 0; y < out.rows; ++y) {
    for(int x = 0; x < out.cols; ++x) {
      out.at<T>(y, x) = in_smoothed.at<T>(y * 2, x * 2);
    }
  }
}

template<typename T>
static void pyrDownSubsample(const cv::Mat& in, cv::Mat& out) {
  out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

  for(int y = 0; y < out.rows; ++y) {
    for(int x = 0; x < out.cols; ++x) {
      out.at<T>(y, x) = in.at<T>(y * 2, x * 2);
    }
  }
}

// Rgbd camera ================================================================
RgbdCamera::RgbdCamera(size_t width, size_t height, const Eigen::Matrix3d& projection) :
    width_(width),
    height_(height),
    proj_(projection),
    inv_proj_(projection.inverse()) {
  pointcloud_template_.resize(Eigen::NoChange, width_ * height_);
  int idx = 0;

  for(size_t y = 0; y < height_; ++y) {
    for(size_t x = 0; x < width_; ++x, ++idx) {
      pointcloud_template_(0, idx) = (x - proj_(0, 2)) / proj_(0, 0);
      pointcloud_template_(1, idx) = (y - proj_(1, 2)) / proj_(1, 1);
      pointcloud_template_(2, idx) = 1.0;
      pointcloud_template_(3, idx) = 0.0;
    }
  }
}

RgbdCamera::~RgbdCamera() {
}

size_t RgbdCamera::width() const {
  return width_;
}

size_t RgbdCamera::height() const {
  return height_;
}

Eigen::Vector4f RgbdCamera::params() const {
  return Eigen::Vector4f(proj_(0, 0), proj_(1, 1),
                          proj_(0, 2), proj_(1, 2));
}
const Eigen::Matrix3d& RgbdCamera::projection() const {
  return proj_;
}

RgbdImagePtr RgbdCamera::create(const cv::Mat& intensity, const cv::Mat& depth) const {
  RgbdImagePtr result(new RgbdImage(*this));

  if (intensity.empty() || depth.empty()) {
    printf("RgbdCamera::create() intensity image or depth image is empty\n");
    abort();
  }

  if (intensity.type() == CV_8UC3 && 
      (depth.type() == CV_16UC1 || depth.type() == CV_16SC1)) {
    result->rgb(intensity.clone());
    cv::Mat t;
    cv::cvtColor(result->rgb(), t, CV_RGB2GRAY);
    t.convertTo(t, CV_32F);
    result->intensity(t);
    
    cv::Mat dep;
    core::convertRawDepthImageSse(depth, dep, 0.001);
    result->depth(dep);
  } else {
    if (intensity.type() != CV_32FC1 || depth.type() != CV_32FC1) {
      printf("RgbdCamera::create() intensity image or depth image is not CV_32FC1\n");
      abort();
    }
    result->intensity(intensity);
    result->depth(depth);
  }
  result->initialize();

  return result;
}

RgbdImagePtr RgbdCamera::create() const {
  return RgbdImagePtr(new RgbdImage(*this));
}

bool RgbdCamera::hasSameSize(const cv::Mat& img) const {
  return img.cols == width_ && img.rows == height_;
}

void RgbdCamera::buildPointCloud(const cv::Mat &depth, PointCloud& pointcloud) const {
  assert(hasSameSize(depth));

  pointcloud.resize(Eigen::NoChange, width_ * height_);

  const float* depth_ptr = depth.ptr<float>();
  int idx = 0;

  for(size_t y = 0; y < height_; ++y) {
    for(size_t x = 0; x < width_; ++x, ++depth_ptr, ++idx) {
      pointcloud.col(idx) = pointcloud_template_.col(idx) * (*depth_ptr);
      pointcloud(3, idx) = 1.0;
    }
  }
}

// Rgbd camera pyramid ========================================================
RgbdCameraPyramid::RgbdCameraPyramid(const RgbdCamera& base) {
  levels_.push_back(std::make_shared<RgbdCamera>(base));
}

RgbdCameraPyramid::RgbdCameraPyramid(size_t base_width, size_t base_height,
  const Eigen::Matrix3d& base_projection) {
  levels_.push_back(std::make_shared<RgbdCamera>(base_width, base_height, base_projection));
}

RgbdCameraPyramid::~RgbdCameraPyramid() {
}

RgbdImagePyramidPtr RgbdCameraPyramid::create(
  const cv::Mat& base_intensity, const cv::Mat& base_depth) {
  return RgbdImagePyramidPtr(new RgbdImagePyramid(*this, base_intensity, base_depth));
}

void RgbdCameraPyramid::build(size_t levels) {
  size_t start = levels_.size();

  for(size_t idx = start; idx < levels; ++idx) {
    RgbdCameraPtr& previous = levels_[idx - 1];

    Eigen::Matrix3d projection(previous->projection());
    projection.block<2, 3>(0, 0) *= 0.5;

    levels_.push_back(std::make_shared<RgbdCamera>(previous->width() / 2,
                      previous->height() / 2, projection));
  }
}

const RgbdCamera& RgbdCameraPyramid::level(size_t level) {
  build(level + 1);

  return *levels_[level];
}

const RgbdCamera& RgbdCameraPyramid::level(size_t level) const {
  return *levels_[level];
}

// Rgbd image ==================================================================
RgbdImage::RgbdImage(const RgbdCamera& camera) :
  camera_(camera),
  loaded_(true),
  intensity_requires_calculation_(true),
  depth_requires_calculation_(true),
  pointcloud_requires_build_(true),
  width_(0),
  height_(0) {
}

RgbdImage::~RgbdImage() {
  DeleteUnloadedImages();
}

const RgbdCamera& RgbdImage::camera() const {
  return camera_;
}

void RgbdImage::initialize() {
  assert(hasIntensity() || hasDepth());

  if(hasIntensity() && hasDepth()) {
    assert(intensity_.size() == depth_.size());
  }

  if(hasIntensity()) {
    assert(intensity_.type() == cv::DataType<IntensityType>::type && 
                               intensity_.channels() == 1);
    width_ = intensity_.cols;
    height_ = intensity_.rows;
  }

  if(hasDepth()) {
    assert(depth_.type() == cv::DataType<DepthType>::type && depth_.channels() == 1);
    width_ = depth_.cols;
    height_ = depth_.rows;
  }

  intensity_requires_calculation_ = true;
  depth_requires_calculation_ = true;
  pointcloud_requires_build_ = true;
}

bool RgbdImage::hasIntensity() const {
  return !intensity_.empty();
}

bool RgbdImage::hasRgb() const {
  return !rgb_.empty();
}

bool RgbdImage::hasDepth() const {
  return !depth_.empty();
}

void RgbdImage::calculateDerivatives() {
  calculateIntensityDerivatives();
  calculateDepthDerivatives();
}

bool RgbdImage::calculateIntensityDerivatives() {
  if (!intensity_requires_calculation_) return false;

  assert(hasIntensity());

  calculateDerivativeX<IntensityType>(intensity_, intensity_dx_);
  //calculateDerivativeY<IntensityType>(intensity, intensity_dy);
  calculateDerivativeYSseFloat(intensity_, intensity_dy_);
  /*
  cv::Mat dy_ref, diff;
  calculateDerivativeY<IntensityType>(intensity, dy_ref);
  cv::absdiff(dy_ref, intensity_dy, diff);
  tracker::util::show("diff", diff);
  cv::waitKey(0);
   */
  intensity_requires_calculation_ = false;
  return true;
}

void RgbdImage::calculateDepthDerivatives() {
  if (!depth_requires_calculation_) return;

  assert(hasDepth());

  calculateDerivativeX<DepthType>(depth_, depth_dx_);
  calculateDerivativeY<DepthType>(depth_, depth_dy_);

  depth_requires_calculation_ = false;
}

template<typename T>
void RgbdImage::calculateDerivativeX(const cv::Mat& img, cv::Mat& result) {
  result.create(img.size(), img.type());

  for (int y = 0; y < img.rows; ++y) {
    for (int x = 0; x < img.cols; ++x) {
      int prev = std::max(x - 1, 0);
      int next = std::min(x + 1, img.cols - 1);

      result.at<T>(y, x) = (T) (img.at<T>(y, next) - img.at<T>(y, prev)) * 0.5f;
    }
  }

  //cv::Sobel(img, result, -1, 1, 0, 3, 1.0f / 4.0f, 0, cv::BORDER_REPLICATE);

  // compiler auto-vectorization
  /*
  const float* img_ptr = img.ptr<float>();
  float* result_ptr = result.ptr<float>();

  for(int y = 0; y < img.rows; ++y)
  {
    *result_ptr++ = img_ptr[1] - img_ptr[0];

    for(int x = 1; x < img.cols - 1; ++x, ++img_ptr)
    {
      *result_ptr++ = img_ptr[2] - img_ptr[0];
    }

    *result_ptr++ = img_ptr[1] - img_ptr[0];

    img_ptr++;
  }
   */
}

template<typename T>
void RgbdImage::calculateDerivativeY(const cv::Mat& img, cv::Mat& result) {
  result.create(img.size(), img.type());

  for (int y = 0; y < img.rows; ++y) {
    for (int x = 0; x < img.cols; ++x) {
      int prev = std::max(y - 1, 0);
      int next = std::min(y + 1, img.rows - 1);

      result.at<T>(y, x) = (T) (img.at<T>(next, x) - img.at<T>(prev, x)) * 0.5f;
    }
  }
  //cv::Sobel(img, result, -1, 0, 1, 3, 1.0f / 4.0f, 0, cv::BORDER_REPLICATE);

  // compiler auto-vectorization
  /*
  for(int y = 0; y < img.rows; ++y)
  {
    const float* prev_row = img.ptr<float>(std::max(y - 1, 0), 0);
    const float* next_row = img.ptr<float>(std::min(y + 1, img.rows - 1), 0);
    float* cur_row = result.ptr<float>(y, 0);

    for(int x = 0; x < img.cols; ++x)
    {
      *cur_row++ = *next_row++ - *prev_row++;
    }
  }
   */
}

void RgbdImage::buildPointCloud() {
  if (!loaded_) Load();
  if (!pointcloud_requires_build_) return;

  assert(hasDepth());

  camera_.buildPointCloud(depth_, pointcloud_);

  pointcloud_requires_build_ = false;
}

void RgbdImage::calculateNormals() {
  if (!loaded_) Load();
  if (angles_.total() == 0) {
    normals_ = cv::Mat::zeros(depth_.size(), CV_32FC4);
    angles_.create(depth_.size(), CV_32FC1);

    float *angle_ptr = angles_.ptr<float>();
    cv::Vec4f *normal_ptr = normals_.ptr<cv::Vec4f>();

    int x_max = depth_.cols - 1;
    int y_max = depth_.rows - 1;

    for (int y = 0; y < depth_.rows; ++y) {
      for (int x = 0; x < depth_.cols; ++x, ++angle_ptr, ++normal_ptr) {
        int idx1 = y * depth_.cols + std::max(x-1, 0);
        int idx2 = y * depth_.cols + std::min(x+1, x_max);
        int idx3 = std::max(y-1, 0) * depth_.cols + x;
        int idx4 = std::min(y+1, y_max) * depth_.cols + x;

        Eigen::Vector4f::AlignedMapType n(normal_ptr->val);
        n = (pointcloud_.col(idx2) - pointcloud_.col(idx1))
                .cross3(pointcloud_.col(idx4) - pointcloud_.col(idx3));
        n.normalize();

        *angle_ptr = std::abs(n(2));
      }
    }
  }
}

void RgbdImage::buildAccelerationStructure() {
  if (!loaded_) Load();
  if (acceleration_.total() == 0) {
    calculateDerivatives();
    cv::Mat zeros = cv::Mat::zeros(intensity_.size(), intensity_.type());
    cv::Mat channels[8] = {intensity_, depth_,    intensity_dx_, intensity_dy_,
                           depth_dx_,  depth_dy_, zeros,         zeros};
    cv::merge(channels, 8, acceleration_);
  }
}

void RgbdImage::warpIntensity(
  const AffineTransform& transformationd, const PointCloud& reference_pointcloud,
  const Eigen::Matrix3d& projection, RgbdImage& result,
  PointCloud& transformed_pointcloud) {
  if (!loaded_) Load();
  Eigen::Affine3f transformation = transformationd.cast<float>();

  cv::Mat warped_image(intensity_.size(), intensity_.type());
  cv::Mat warped_depth(depth_.size(), depth_.type());

  float ox = projection(0, 2);
  float oy = projection(1, 2);

  float* warped_intensity_ptr = warped_image.ptr<IntensityType>();
  float* warped_depth_ptr = warped_depth.ptr<DepthType>();

  int outliers = 0;
  int total = 0;
  int idx = 0;

  transformed_pointcloud = transformation * reference_pointcloud;

  for (size_t y = 0; y < height_; ++y) {
    for (size_t x = 0; x < width_;
      ++x, ++idx, ++warped_intensity_ptr, ++warped_depth_ptr) {

      const Eigen::Vector4f& p3d = transformed_pointcloud.col(idx);

      if (!std::isfinite(p3d(2))) {
        *warped_intensity_ptr = Invalid;
        *warped_depth_ptr = InvalidDepth;
        continue;
      }

      float x_projected = (float) (p3d(0) * projection(0, 0) / p3d(2) + ox);
      float y_projected = (float) (p3d(1) * projection(1, 1) / p3d(2) + oy);

      if (InImage(x_projected, y_projected)) {
        float z = (float) p3d(2);

        *warped_intensity_ptr = Interpolation::bilinearWithDepthBuffer(this->intensity_, this->depth_, x_projected, y_projected, z);
        *warped_depth_ptr = z;
      } else {
        *warped_intensity_ptr = Invalid;
        *warped_depth_ptr = InvalidDepth;
        //outliers++;
      }

      //total++;
    }
  }

  result.intensity_ = warped_image;
  result.depth_ = warped_depth;
  result.initialize();
}

void RgbdImage::warpDepthForward(
  const AffineTransform& transformationx, const Eigen::Matrix3d& projection,
  RgbdImage& result, cv::Mat_<cv::Vec3d>& cloud) {
  if (!loaded_) Load();
  Eigen::Affine3d transformation = transformationx.cast<double>();

  cloud = cv::Mat_<cv::Vec3d>(depth_.size(), cv::Vec3d(0, 0, 0));
  cv::Mat warped_depth = cv::Mat::zeros(depth_.size(), depth_.type());
  warped_depth.setTo(InvalidDepth);

  float ox = projection(0, 2);
  float oy = projection(1, 2);

  const float* depth_ptr = depth_.ptr<float>();
  int outliers = 0;
  int total = 0;

  for (size_t y = 0; y < height_; ++y) {
    for (size_t x = 0; x < width_; ++x, ++depth_ptr) {
      if (!std::isfinite(*depth_ptr)) {
        continue;
      }

      float depth = *depth_ptr;
      Eigen::Vector3d p3d((x - ox) * depth / projection(0, 0), (y - oy) * depth / projection(1, 1), depth);
      Eigen::Vector3d p3d_transformed = transformation * p3d;

      float x_projected = (float) (p3d_transformed(0) * projection(0, 0) / p3d_transformed(2) + ox);
      float y_projected = (float) (p3d_transformed(1) * projection(1, 1) / p3d_transformed(2) + oy);

      if (InImage(x_projected, y_projected)) {
        int yi = (int) y_projected, xi = (int) x_projected;

        if(!std::isfinite(warped_depth.at<DepthType>(yi, xi)) ||
           (warped_depth.at<DepthType>(yi, xi) - 0.05) > depth)
          warped_depth.at<DepthType>(yi, xi) = depth;
      }

      p3d = p3d_transformed;

      total++;
      cloud(y, x) = cv::Vec3d(p3d(0), p3d(1), p3d(2));
    }
  }

  result.depth_ = warped_depth;
  result.initialize();
}

void RgbdImage::warpIntensityForward(
  const AffineTransform& transformationx, const Eigen::Matrix3d& projection,
  RgbdImage& result, cv::Mat_<cv::Vec3d>& cloud) {
  if (!loaded_) Load();
  Eigen::Affine3d transformation = transformationx.cast<double>();

  bool identity = transformation.affine().isIdentity(1e-6);

  cloud = cv::Mat_<cv::Vec3d>(intensity_.size(), cv::Vec3d(0, 0, 0));
  cv::Mat warped_image = cv::Mat::zeros(intensity_.size(), intensity_.type());

  float ox = projection(0, 2);
  float oy = projection(1, 2);

  const float* depth_ptr = depth_.ptr<float>();
  int outliers = 0;
  int total = 0;

  for (size_t y = 0; y < height_; ++y) {
    for (size_t x = 0; x < width_; ++x, ++depth_ptr) {
      if(*depth_ptr <= 1e-6f) continue;

      float depth = *depth_ptr;
      Eigen::Vector3d p3d((x - ox) * depth / projection(0, 0), (y - oy) * depth / projection(1, 1), depth);

      if(!identity) {
        Eigen::Vector3d p3d_transformed = transformation * p3d;

        float x_projected = (float) (p3d_transformed(0) * projection(0, 0) / p3d_transformed(2) + ox);
        float y_projected = (float) (p3d_transformed(1) * projection(1, 1) / p3d_transformed(2) + oy);

        if (InImage(x_projected, y_projected)) {
          int xp, yp;
          xp = (int) std::floor(x_projected);
          yp = (int) std::floor(y_projected);

          warped_image.at<IntensityType>(yp, xp) = intensity_.at<IntensityType>(y, x);
        } else {
          outliers++;
        }

        p3d = p3d_transformed;
      }

      total++;
      cloud(y, x) = cv::Vec3d(p3d(0), p3d(1), p3d(2));
    }
  }

  //std::cerr << "warp out: " << outliers << " total: " << total << std::endl;

  if (identity) {
    warped_image = intensity_;
  } else {
    //std::cerr << "did warp" << std::endl;
  }

  result.intensity_ = warped_image;
  result.depth_ = depth_;
  result.initialize();
}

void RgbdImage::warpDepthForwardAdvanced(
  const AffineTransform& transformation, const Eigen::Matrix3d& projection,
  RgbdImage& result) {
  if (!loaded_) Load();
  assert(hasDepth());

  this->buildPointCloud();

  PointCloud transformed_pointcloud = transformation.cast<float>() * pointcloud_;

  cv::Mat warped_depth(depth_.size(), depth_.type());
  warped_depth.setTo(InvalidDepth);

  float z_factor1 = transformation.rotation()(0, 0) + transformation.rotation()(0, 1) * (projection(0, 0) / projection(1, 1));
  float x_factor1 = -transformation.rotation()(2, 0) - transformation.rotation()(2, 1) * (projection(0, 0) / projection(1, 1));

  float z_factor2 = transformation.rotation()(1, 1) + transformation.rotation()(1, 0) * (projection(1, 1) / projection(0, 0));
  float y_factor2 = -transformation.rotation()(2, 1) - transformation.rotation()(2, 0) * (projection(1, 1) / projection(0, 0));

  for (int idx = 0; idx < height_ * width_; ++idx) {
    Vector4 p3d = pointcloud_.col(idx);
    NumType z = p3d(2);

    if (!std::isfinite(z)) continue;

    int x_length = (int) std::ceil(z_factor1 + x_factor1 * p3d(0) / z) + 1; // magic +1
    int y_length = (int) std::ceil(z_factor2 + y_factor2 * p3d(1) / z) + 1; // magic +1

    Vector4 p3d_transformed = transformed_pointcloud.col(idx);
    NumType z_transformed = p3d_transformed(2);

    int x_projected = (int) std::floor(p3d_transformed(0) * projection(0, 0) / z_transformed + projection(0, 2));
    int y_projected = (int) std::floor(p3d_transformed(1) * projection(1, 1) / z_transformed + projection(1, 2));

    // TODO: replace InImage(...) checks, with max(..., 0) on initial value of x_, y_ and  min(..., width/height) for their respective upper bound
    //for (int y_ = y_projected; y_ < y_projected + y_length; y_++)
    //  for (int x_ = x_projected; x_ < x_projected + x_length; x_++)

    int x_begin = std::max(x_projected, 0);
    int y_begin = std::max(y_projected, 0);
    int x_end = std::min(x_projected + x_length, (int) width_);
    int y_end = std::min(y_projected + y_length, (int) height_);

    for (int y = y_begin; y < y_end; ++y) {
      DepthType* v = warped_depth.ptr<DepthType>(y, x_begin);

      for (int x = x_begin; x < x_end; ++x, ++v) {
        if (!std::isfinite(*v) || (*v) > z_transformed) {
          (*v) = (DepthType) z_transformed;
        }
      }
    }
  }

  result.depth_ = warped_depth;
  result.initialize();
}

bool RgbdImage::InImage(const float& x, const float& y) const {
  return x >= 0 && x < width_ && y >= 0 && y < height_;
}

void RgbdImage::Unload() {
  if (!loaded_) return;
  static int count = 0;
  const std::string prefix = ExePath() + "/" + std::string("frame-") +
                             std::to_string(timestamp_) + "-" +
                             std::to_string(count);
  int countz = 0;
  for (int r = 0; r < depth_.rows; ++r) {
    for (int c = 0; c < depth_.cols; ++c) {
      if (depth_.at<float>(r, c) != 0.0f)
        countz += 1;
    }
  }
  SaveImages(intensity_, depth_, rgb_, prefix, intensity_fn_, depth_fn_, rgb_fn_);
  intensity_.release();
  depth_.release();
  rgb_.release();
  count++;
  loaded_ = false;
}  

void RgbdImage::Load() {
  if (loaded_) return;
  cv::Mat inten, dep;
  LoadImages(intensity_fn_, depth_fn_, rgb_fn_, inten, dep, rgb_);
  inten.convertTo(intensity_, CV_32FC1);
  dep.convertTo(depth_, CV_32FC1);
  depth_ = depth_ * 0.001;
  loaded_ = true;
}

void RgbdImage::ReleaseTrackingData() {
  intensity_dx_.release();
  intensity_dy_.release();
  intensity_requires_calculation_ = true;

  depth_dx_.release();
  depth_dy_.release();
  depth_requires_calculation_ = true;

  normals_.release();
  angles_.release();

  pointcloud_.resize(Eigen::NoChange, 0);
  pointcloud_requires_build_ = true;

  acceleration_.release();
}

void RgbdImage::DeleteUnloadedImages() {
  std::remove(intensity_fn_.c_str());
  std::remove(depth_fn_.c_str());
  std::remove(rgb_fn_.c_str());
}

// Rgbd image pyramid ==========================================================
static PointcloudVec pointcloud_template_vec;
RgbdImagePyramid::RgbdImagePyramid(
  RgbdCameraPyramid& camera, const cv::Mat& intensity, const cv::Mat& depth) 
  : id_(-1), camera_(camera) {
  levels_.push_back(camera_.level(0).create(intensity, depth));
  pose_.setIdentity();
}

RgbdImagePyramid::RgbdImagePyramid(
  int id, double timestamp, const Eigen::Matrix3d& proj,
  cv::Mat img, cv::Mat depth, short levels)
  : id_(id), timestamp_(timestamp), camera_(img.cols, img.rows, proj) {
  camera_.build(levels);
  levels_.push_back(camera_.level(0).create(img, depth));
  level(0)->timestamp(timestamp);
  pose_.setIdentity();
  build(levels);
}

RgbdImagePyramid::~RgbdImagePyramid() {
  std::cout << "RgbdImagePyramid deconstructor..." << std::endl;
}

void RgbdImagePyramid::build(const size_t num_levels) {
  if(levels_.size() >= num_levels) return;

  // if we already have some levels, we just need to compute the coarser levels
  size_t first = levels_.size();

  for(size_t idx = first; idx < num_levels; ++idx) {
    levels_.push_back(camera_.level(idx).create());
#if 0  // use original code
    cv::Mat inten, dep;
    pyrDownMeanSmooth<IntensityType>(levels_[idx - 1]->intensity(), inten);
    //pyrDownMeanSmoothIgnoreInvalid<float>(levels[idx - 1].depth, levels[idx].depth);
    pyrDownSubsample<float>(levels_[idx - 1]->depth(), dep);

    levels_[idx]->intensity(inten);
    levels_[idx]->depth(dep);
#else
    const cv::Size size(levels_[idx - 1]->intensity().size().width/2,
                        levels_[idx - 1]->intensity().size().height/2);
    cv::Mat inten, dep;
    cv::resize(levels_[idx - 1]->intensity(), inten, size);
    cv::resize(levels_[idx - 1]->depth(), dep, size,
               0.0, 0.0, cv::INTER_NEAREST);

    levels_[idx]->intensity(inten);
    levels_[idx]->depth(dep);
    levels_[idx]->timestamp(levels_[idx-1]->timestamp());
#endif
    levels_[idx]->initialize();
  }
}

RgbdImagePtr RgbdImagePyramid::level(size_t idx) const {
  assert(idx < levels_.size());
  return levels_[idx];
}

double RgbdImagePyramid::timestamp() const {
  return !levels_.empty() ? levels_[0]->timestamp(): 0.0;
}

void RgbdImagePyramid::timestamp(double t) {
  assert(!levels_.empty());
  return levels_[0]->timestamp(t);
}

PointcloudVec& RgbdImagePyramid::TemplatePointcloudVec() const {
  for (size_t l = pointcloud_template_vec.size(); l < levels_.size(); ++l) {
    pointcloud_template_vec.push_back(levels_[l]->camera().TemplatePointcloud());
  }
  return pointcloud_template_vec;
}

void RgbdImagePyramid::Unload() {
  for (auto img : levels_)
    img->Unload();
}

void RgbdImagePyramid::Load() {
  for (auto img : levels_)
    img->Load();
}

void RgbdImagePyramid::ReleaseTrackingData() {
  for (auto img : levels_)
    img->ReleaseTrackingData();
}
}  // namespace core
}  // namespace dvo