#include "dvo/core/point_selection.h"
#include "dvo/core/grid_hash.h"

namespace dvo {
namespace core {

PointSelection::PointSelection(const PointSelectionPredicate &predicate)
    : pyramid_(0), predicate_(predicate), debug_(false), grid_filter_(false) {}

PointSelection::PointSelection(dvo::core::RgbdImagePyramid &pyramid,
                               const PointSelectionPredicate &predicate)
    : pyramid_(&pyramid), predicate_(predicate), debug_(false), grid_filter_(false) {}

PointSelection::~PointSelection() {}

void PointSelection::recycle(dvo::core::RgbdImagePyramid &pyramid) {
  setRgbdImagePyramid(pyramid);
}

void PointSelection::setRgbdImagePyramid(dvo::core::RgbdImagePyramid &pyramid) {
  pyramid_ = &pyramid;

  for (size_t idx = 0; idx < storage_.size(); ++idx) {
    storage_[idx].is_cached = false;
  }
}

dvo::core::RgbdImagePyramid &PointSelection::getRgbdImagePyramid() {
  assert(pyramid_ != 0);

  return *pyramid_;
}

size_t PointSelection::getMaximumNumberOfPoints(const size_t &level) {
  return size_t(pyramid_->level(0)->intensity().total() *
                std::pow(0.25, double(level)));
}

bool PointSelection::getDebugIndex(const size_t &level, cv::Mat &dbg_idx) {
  if (debug_ && storage_.size() > level) {
    dbg_idx = storage_[level].debug_idx;

    return dbg_idx.total() > 0;
  } else {
    return false;
  }
}

void PointSelection::select(const size_t &level,
                            PointSelection::PointIterator &first_point,
                            PointSelection::PointIterator &last_point) {
  assert(pyramid_ != 0);

  pyramid_->build(level + 1);

  if (storage_.size() < level + 1)
    storage_.resize(level + 1);

  Storage &storage = storage_[level];

  if (!storage.is_cached || debug_) {
    dvo::core::RgbdImage &img = *pyramid_->level(level);
    img.buildPointCloud();
    img.buildAccelerationStructure();

    if (debug_)
      storage.debug_idx = cv::Mat::zeros(img.intensity().size(), CV_8UC1);

    storage.allocate(img.intensity().total());
    storage.points_end = selectPointsFromImage(
        img, storage.points.begin(), storage.points.end(), storage.debug_idx);

    storage.is_cached = true;
  }

  first_point = storage.points.begin();
  last_point = storage.points_end;
}

struct PerPointData {
  int x, y;
  int idx;
  double value;
};

PointSelection::PointIterator PointSelection::selectPointsFromImage(
    const dvo::core::RgbdImage &img,
    const PointSelection::PointIterator &first_point,
    const PointSelection::PointIterator &last_point, cv::Mat &debug_idx) {
  const auto *points =
      (const PointWithIntensityAndDepth::Point *)img.pointcloud().data();
#ifndef __APPLE__
  const auto *intensity_and_depth =
      img.acceleration().ptr<PointWithIntensityAndDepth::IntensityAndDepth>();
#else
  const auto *intensity_and_depth =
      (PointWithIntensityAndDepth::IntensityAndDepth *)img.acceleration.ptr();
#endif

  auto selected_points_it = first_point;

  // float dt = 1.0f / 30.0f / img.height;

  std::vector<PerPointData> ppdata;
  ppdata.reserve(std::distance(first_point, last_point));

  for (size_t y = 0; y < img.height(); ++y) {
    // float time_interpolation = 1 + (y - 0.5f * img.height) * dt;

    for (size_t x = 0; x < img.width(); ++x, ++points, ++intensity_and_depth) {
      if (predicate_.isPointOk(x, y, points->z, intensity_and_depth->idx,
                               intensity_and_depth->idy,
                               intensity_and_depth->zdx,
                               intensity_and_depth->zdy)) {
        selected_points_it->point = *points;
        selected_points_it->intensity_and_depth = *intensity_and_depth;
        // selected_points_it->intensity_and_depth.time_interpolation =
        // time_interpolation;

        ++selected_points_it;
        ppdata.push_back(
            PerPointData{x, y, ppdata.size(),
                         std::fabs(intensity_and_depth->idx) + std::fabs(intensity_and_depth->idy)});

        if (debug_)
          debug_idx.at<uint8_t>(y, x) = 1;

        if (selected_points_it == last_point)
          return selected_points_it;
      }
    }
  }

  if (!grid_filter_)
    return selected_points_it;

  GridHashBase<PerPointData> gridhash(img.width(), img.height(), 5);
  for (auto d : ppdata) {
    gridhash.insert(d.x, d.y, d);
  }

  std::vector<PerPointData> els;
  els.reserve(gridhash.hash_table.size());
  for (auto& t : gridhash.hash_table) {
    if (t.empty()) continue;
    auto e = *std::max_element(t.begin(), t.end(), [](const PerPointData& lhs, const PerPointData& rhs) {
      return lhs.value > rhs.value;
    });
    els.push_back(e);
  }

  auto grid_it = first_point;
  for (int i = 0; i < int(els.size()); ++i) {
    *grid_it = *(grid_it + els[i].idx);
    grid_it++;
  }

  return grid_it;
}

PointSelection::Storage::Storage()
    : points(), points_end(points.end()), is_cached(false) {}

void PointSelection::Storage::allocate(size_t max_points) {
  if (points.size() < max_points) {
    points.resize(max_points);
  }
}

} // namespace core
} // namespace dvo