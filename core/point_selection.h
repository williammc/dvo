#pragma once

#include <dvo/core/rgbd_image.h>

#include <dvo/dvo_api.h>

namespace dvo {
namespace core {

class DVO_API PointSelectionPredicate {
public:
  virtual ~PointSelectionPredicate() {}
  virtual bool isPointOk(const size_t &x, const size_t &y, const float &z,
                         const float &idx, const float &idy, const float &zdx,
                         const float &zdy) const = 0;
};

class DVO_API ValidPointPredicate : public PointSelectionPredicate {
public:
  virtual ~ValidPointPredicate() {}
  virtual bool isPointOk(const size_t &x, const size_t &y, const float &z,
                         const float &idx, const float &idy, const float &zdx,
                         const float &zdy) const {
    return z == z && zdx == zdx && zdy == zdy;
  }
};

class DVO_API ValidPointAndGradientThresholdPredicate
    : public PointSelectionPredicate {
public:
  float intensity_threshold;
  float depth_threshold;

  ValidPointAndGradientThresholdPredicate()
      : intensity_threshold(0.0f), depth_threshold(0.0f) {}

  virtual ~ValidPointAndGradientThresholdPredicate() {}

  virtual bool isPointOk(const size_t &x, const size_t &y, const float &z,
                         const float &idx, const float &idy, const float &zdx,
                         const float &zdy) const {
    return z == z && zdx == zdx && zdy == zdy &&
           (std::abs(idx) > intensity_threshold ||
            std::abs(idy) > intensity_threshold ||
            std::abs(zdx) > depth_threshold || std::abs(zdy) > depth_threshold);
  }
};

class DVO_API PointSelection {
public:
  typedef PointWithIntensityAndDepth::VectorType PointVector;
  typedef PointVector::iterator PointIterator;

  PointSelection(const PointSelectionPredicate &predicate);
  PointSelection(dvo::core::RgbdImagePyramid &pyramid,
                 const PointSelectionPredicate &predicate);
  virtual ~PointSelection();

  dvo::core::RgbdImagePyramid &getRgbdImagePyramid();

  void setRgbdImagePyramid(dvo::core::RgbdImagePyramid &pyramid);

  size_t getMaximumNumberOfPoints(const size_t &level);

  void select(const size_t &level, PointIterator &first_point,
              PointIterator &last_point);

  void recycle(dvo::core::RgbdImagePyramid &pyramid);

  bool getDebugIndex(const size_t &level, cv::Mat &dbg_idx);

  void debug(bool v) { debug_ = v; }

  bool debug() const { return debug_; }

private:
  struct Storage {
  public:
    PointVector points;
    PointIterator points_end;
    bool is_cached;

    cv::Mat debug_idx;

    Storage();
    void allocate(size_t max_points);
  };

  dvo::core::RgbdImagePyramid *pyramid_;
  std::vector<Storage> storage_;
  const PointSelectionPredicate &predicate_;

  bool debug_;

  PointIterator selectPointsFromImage(const dvo::core::RgbdImage &img,
                                      const PointIterator &first_point,
                                      const PointIterator &last_point,
                                      cv::Mat &debug_idx);
};

} // namespace core
} // namespace dvo