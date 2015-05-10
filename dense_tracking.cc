#include "dvo/dense_tracking.h"
#include "dvo/dense_tracking_impl.h"

#include <assert.h>
#include <iomanip>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "dvo/core/datatypes.h"
#include "dvo/util/revertable.h"
#include "dvo/util/stopwatch.h"
#include "dvo/util/id_generator.h"
#include "dvo/util/histogram.h"

namespace dvo {

using namespace dvo::core;
using namespace dvo::util;

const DenseTracker::Config &DenseTracker::getDefaultConfig() {
  static Config defaultConfig;

  return defaultConfig;
}

static const Eigen::IOFormat YamlArrayFmt(Eigen::FullPrecision,
                                          Eigen::DontAlignCols, ",", ",", "",
                                          "", "[", "]");

DenseTracker::DenseTracker(const Config &config)
    : itctx_(cfg),
      weight_calculation_(),
      selection_predicate_(),
      reference_selection_(selection_predicate_) {
  configure(config);
}

DenseTracker::DenseTracker(const DenseTracker &other)
    : itctx_(cfg),
      weight_calculation_(),
      selection_predicate_(),
      reference_selection_(selection_predicate_) {
  configure(other.configuration());
}

void DenseTracker::configure(const Config &config) {
  assert(config.IsSane());

  cfg = config;

  selection_predicate_.intensity_threshold = cfg.IntensityDerivativeThreshold;
  selection_predicate_.depth_threshold = cfg.DepthDerivativeThreshold;

  if (cfg.UseWeighting) {
    weight_calculation_.scaleEstimator(
                            ScaleEstimators::get(cfg.ScaleEstimatorType))
        .scaleEstimator()
        ->configure(cfg.ScaleEstimatorParam);

    weight_calculation_.influenceFunction(
                            InfluenceFunctions::get(cfg.InfluenceFuntionType))
        .influenceFunction()
        ->configure(cfg.InfluenceFunctionParam);
  } else {
    weight_calculation_.scaleEstimator(
                            ScaleEstimators::get(ScaleEstimators::Unit))
        .influenceFunction(InfluenceFunctions::get(InfluenceFunctions::Unit));
  }
}

bool DenseTracker::match(RgbdImagePyramid &reference, RgbdImagePyramid &current,
                         Eigen::Affine3d &current2reference) {
  Result result;
  result.Transformation = current2reference;

  bool success = match(reference, current, result);

  current2reference = result.Transformation;

  return success;
}

bool DenseTracker::match(dvo::core::PointSelection &reference,
                         RgbdImagePyramid &current,
                         Eigen::Affine3d &current2reference) {
  Result result;
  result.Transformation = current2reference;

  bool success = match(reference, current, result);

  current2reference = result.Transformation;

  return success;
}

bool DenseTracker::match(dvo::core::RgbdImagePyramid &reference,
                         dvo::core::RgbdImagePyramid &current,
                         dvo::DenseTracker::Result &result) {
  reference.build(cfg.getNumLevels());
  reference_selection_.setRgbdImagePyramid(reference);

  return match(reference_selection_, current, result);
}

bool DenseTracker::match(dvo::core::PointSelection &reference,
                         dvo::core::RgbdImagePyramid &current,
                         dvo::DenseTracker::Result &result) {
  current.build(cfg.getNumLevels());

  bool success = true;

  if (cfg.UseInitialEstimate) {
    assert(!result.isNaN() && "Provided initialization is NaN!");
  } else {
    result.setIdentity();
  }

  // our first increment is the given guess
  Sophus::SE3d inc(result.Transformation.rotation(),
                   result.Transformation.translation());

  Revertable<Sophus::SE3d> initial(inc);
  Revertable<Sophus::SE3d> estimate;

  bool accept = true;

  if (points_error.size() < reference.getMaximumNumberOfPoints(cfg.LastLevel))
    points_error.resize(reference.getMaximumNumberOfPoints(cfg.LastLevel));
  if (residuals.size() < reference.getMaximumNumberOfPoints(cfg.LastLevel))
    residuals.resize(reference.getMaximumNumberOfPoints(cfg.LastLevel));
  if (weights.size() < reference.getMaximumNumberOfPoints(cfg.LastLevel))
    weights.resize(reference.getMaximumNumberOfPoints(cfg.LastLevel));

  std::vector<uint8_t> valid_residuals;

  bool debug = false;
  if (debug) {
    reference.debug(true);
    valid_residuals.resize(reference.getMaximumNumberOfPoints(cfg.LastLevel));
  }
#if 0
  std::stringstream name;
  name << std::setiosflags(std::ios::fixed) << std::setprecision(2) << current.timestamp() << "_error.avi";

  cv::Size s = reference.getRgbdImagePyramid().level(size_t(cfg.LastLevel)).intensity.size();
  cv::Mat video_frame(s.height, s.width * 2, CV_32FC1), video_frame_u8;
  cv::VideoWriter vw(name.str(), CV_FOURCC('P','I','M','1'), 30, video_frame.size(), false);
  float rgb_max = 0.0;
  float depth_max = 0.0;

  std::stringstream name1;
  name1 << std::setiosflags(std::ios::fixed) << std::setprecision(2) << current.timestamp() << "_ref.png";

  cv::imwrite(name1.str(), current.level(0).rgb);

  std::stringstream name2;
  name2 << std::setiosflags(std::ios::fixed) << std::setprecision(2) << current.timestamp() << "_cur.png";

  cv::imwrite(name2.str(), reference.getRgbdImagePyramid().level(0).rgb);
#endif
  Eigen::Vector2f mean;
  mean.setZero();
  Eigen::Matrix2f precision;
  precision.setZero();
  result.Statistics.Levels.reserve(4);

  dvo::util::stopwatch watch("match() iterations", 3);
  std::array<float, 4> last_bbox;
  int last_level_tmp = 0;
  for (itctx_.Level = cfg.FirstLevel; itctx_.Level >= cfg.LastLevel;
       --itctx_.Level) {
    watch.start();
    result.Statistics.Levels.push_back(LevelStats());
    LevelStats &level_stats = result.Statistics.Levels.back();

    mean.setZero();
    precision.setZero();

    // reset error after every pyramid level? yes because errors from different
    // levels are not comparable
    itctx_.Iteration = 0;
    itctx_.Error = std::numeric_limits<double>::max();
    
    last_level_tmp = itctx_.Level;
    RgbdImage &cur = *current.level(itctx_.Level);
    const auto &K = cur.camera().projection();

    Vector8f wcur, wref;
    // i z idx idy zdx zdy
    float wcur_id = 0.5f, wref_id = 0.5f, wcur_zd = 1.0f, wref_zd = 0.0f;

    wcur << 1.0f / 255.0f, 1.0f, wcur_id *float(K(0, 0)) / 255.0f,
        wcur_id * float(K(1, 1)) / 255.0f, wcur_zd * float(K(0, 0)),
        wcur_zd * float(K(1, 1)), 0.0f, 0.0f;
    wref << -1.0f / 255.0f, -1.0f, wref_id *float(K(0, 0)) / 255.0f,
        wref_id * float(K(1, 1)) / 255.0f, wref_zd * float(K(0, 0)),
        wref_zd * float(K(1, 1)), 0.0f, 0.0f;

    PointSelection::PointIterator first_point, last_point;
    reference.select(itctx_.Level, first_point, last_point);
    cur.buildAccelerationStructure();

    level_stats.Id = itctx_.Level;
    level_stats.MaxValidPixels =
        reference.getMaximumNumberOfPoints(itctx_.Level);
    level_stats.ValidPixels = last_point - first_point;

    NormalEquationsLeastSquares ls;
    Matrix6d A;
    Vector6d x, b;
    x = inc.log();

    ComputeResidualsResult compute_residuals_result;
    compute_residuals_result.first_point_error = points_error.begin();
    compute_residuals_result.first_residual = residuals.begin();
    compute_residuals_result.first_valid_flag = valid_residuals.begin();

#ifdef _DEBUG
    printf("\n==========================================\n");
    printf("DenseTracking\n");
#endif
    level_stats.Iterations.reserve(10);
    do {
      level_stats.Iterations.push_back(IterationStats());
      IterationStats &iter_stats = level_stats.Iterations.back();
      iter_stats.Id = itctx_.Iteration;

      double total_error = 0.0f;
      Eigen::Affine3f transformf;

      inc = Sophus::SE3d::exp(x);
      initial.update() = inc.inverse() * initial();
      estimate.update() = inc * estimate();

      transformf = estimate().matrix().cast<float>();

      if (debug) {
        dvo::core::computeResidualsAndValidFlagsSse(
            first_point, last_point, cur, K, transformf, wref, wcur,
            compute_residuals_result);
      } else {
        dvo::core::computeResidualsSse(first_point, last_point, cur, K,
                                       transformf, wref, wcur,
                                       compute_residuals_result);
      }
      last_bbox = compute_residuals_result.valid_points_bbox;
      size_t n = (compute_residuals_result.last_residual -
                  compute_residuals_result.first_residual);
      iter_stats.ValidConstraints = n;

      if (n < 6) {
        initial.revert();
        estimate.revert();

        level_stats.TerminationCriterion =
            TerminationCriteria::TooFewConstraints;

        break;
      }

      if (itctx_.IsFirstIterationOnLevel()) {
        std::fill(weights.begin(), weights.begin() + n, 1.0f);
      } else {
        dvo::core::computeWeightsSse(compute_residuals_result.first_residual,
                                     compute_residuals_result.last_residual,
                                     weights.begin(), mean, precision);
      }

      precision =
          dvo::core::computeScaleSse(compute_residuals_result.first_residual,
                                     compute_residuals_result.last_residual,
                                     weights.begin(), mean).inverse();

      float ll = computeCompleteDataLogLikelihood(
          compute_residuals_result.first_residual,
          compute_residuals_result.last_residual, weights.begin(), mean,
          precision);

      iter_stats.TDistributionLogLikelihood = -ll;
      iter_stats.TDistributionMean = mean.cast<double>();
      iter_stats.TDistributionPrecision = precision.cast<double>();
      iter_stats.PriorLogLikelihood = cfg.Mu * initial().log().squaredNorm();

      total_error = -ll;  // iter_stats.TDistributionLogLikelihood +
                          // iter_stats.PriorLogLikelihood;

      itctx_.LastError = itctx_.Error;
      itctx_.Error = total_error;

      iter_stats.Error = itctx_.Error;
      iter_stats.WeightedError = computeWeightedErrorSse(
          compute_residuals_result.first_residual,
          compute_residuals_result.last_residual, weights.begin(), precision);
      iter_stats.ValidConstraintsRatio =
          iter_stats.ValidConstraints / double(last_point - first_point);
#ifdef _DEBUG
      printf("\nIteration %d:", int(level_stats.Iterations.size()));
      printf("average residual error: %f,", iter_stats.Error);
      printf("average WeightedError: %f\n", iter_stats.WeightedError);
      printf("ValidConstraintsRatio: %f\n", iter_stats.ValidConstraintsRatio);
#endif
      // accept the last increment?
      accept = itctx_.Error < itctx_.LastError;

      if (!accept) {
        initial.revert();
        estimate.revert();

        level_stats.TerminationCriterion =
            TerminationCriteria::LogLikelihoodDecreased;

        break;
      }
      // now build equation system
      WeightVectorType::iterator w_it = weights.begin();

      Matrix2x6 J, Jw;
      Eigen::Vector2f Ji;
      Vector6 Jz;
      ls.initialize(1);
      for (auto e_it = compute_residuals_result.first_point_error;
           e_it != compute_residuals_result.last_point_error; ++e_it, ++w_it) {
        computeJacobianOfProjectionAndTransformation(e_it->getPointVec4f(), Jw);
        compute3rdRowOfJacobianOfTransformation(e_it->getPointVec4f(), Jz);

        J.row(0) = e_it->getIntensityDerivativeVec2f().transpose() * Jw;
        J.row(1) =
            e_it->getDepthDerivativeVec2f().transpose() * Jw - Jz.transpose();

        ls.update(J, e_it->getIntensityAndDepthVec2f(), (*w_it) * precision);
      }
      ls.finish();

      A = ls.A.cast<double>() + cfg.Mu * Matrix6d::Identity();
      b = ls.b.cast<double>() + cfg.Mu * initial().log();
      x = A.ldlt().solve(b);

      iter_stats.EstimateIncrement = x;
      iter_stats.EstimateInformation = A;

      itctx_.Iteration++;
    } while (accept && x.lpNorm<Eigen::Infinity>() > cfg.Precision &&
             !itctx_.IterationsExceeded());

    if (x.lpNorm<Eigen::Infinity>() <= cfg.Precision)
      level_stats.TerminationCriterion = TerminationCriteria::IncrementTooSmall;

    if (itctx_.IterationsExceeded())
      level_stats.TerminationCriterion =
          TerminationCriteria::IterationsExceeded;

    watch.stopAndPrint();
  }
  LevelStats &last_level = result.Statistics.Levels.back();
  IterationStats &last_iteration =
      last_level.TerminationCriterion !=
              TerminationCriteria::LogLikelihoodDecreased
          ? last_level.Iterations[last_level.Iterations.size() - 1]
          : last_level.Iterations[last_level.Iterations.size() - 2];

  int curr_area = current.level(last_level_tmp)->width()*current.level(last_level_tmp)->height();
  float bbox_area =
      (last_bbox[2] - last_bbox[0]) * (last_bbox[3] - last_bbox[1]);
  float visible_ratio = bbox_area / curr_area;

  result.Transformation = estimate().inverse().matrix();
  result.Information = last_iteration.EstimateInformation * 0.008 * 0.008;
  result.LogLikelihood = last_iteration.TDistributionLogLikelihood +
                         last_iteration.PriorLogLikelihood;
  printf("Tracking Statistics:\n");
  printf("Error:%f, WeightedError:%f, ValidConstraintsRatio:%f, VisibleRatio:%f",
    last_iteration.Error, last_iteration.WeightedError, 
    last_iteration.ValidConstraintsRatio, visible_ratio);
  // float visible_ratio = float(last_iteration.ValidConstraints)/last_level.MaxValidPixels;
  visible_ratio = 1.0f;
  if (last_iteration.WeightedError < 1.0f &&
      last_iteration.ValidConstraintsRatio > 0.6f &&
      visible_ratio > 0.6f)
    result.Status = Result::TrackStatus::GOOD;
#if 0 // for imperial dataset
  else if (last_iteration.WeightedError < 1.5f &&
           last_iteration.ValidConstraintsRatio > 0.5f &&
           visible_ratio > 0.5f)
    result.Status = Result::TrackStatus::OK;
#else // for recording datasets
  else if (last_iteration.WeightedError < 2.0f &&
           last_iteration.ValidConstraintsRatio > 0.6f &&
           visible_ratio > 0.6f)
    result.Status = Result::TrackStatus::OK;
  else if (last_iteration.WeightedError < 1.2f &&
           last_iteration.ValidConstraintsRatio > 0.5f &&
           visible_ratio > 0.3f)
    result.Status = Result::TrackStatus::OK;
  // else if (last_iteration.WeightedError < 2.0f &&
  //          last_iteration.ValidConstraintsRatio > 0.7f &&
  //          visible_ratio > 0.3f)
  //   result.Status = Result::TrackStatus::OK;
#endif
  // else if (last_iteration.WeightedError < 2.0f &&
  //          last_iteration.ValidConstraintsRatio > 0.5f &&
  //          visible_ratio > 0.5f)
  //   result.Status = Result::TrackStatus::BAD;
  else
    result.Status = Result::TrackStatus::FAILURE;

  printf("Match() result:%s\n", Result::ToString(result.Status).c_str());
  return success;
}

cv::Mat DenseTracker::computeIntensityErrorImage(
    dvo::core::RgbdImagePyramid &reference,
    dvo::core::RgbdImagePyramid &current,
    const dvo::core::AffineTransformd &reference2current, size_t level) {
  reference.build(level + 1);
  current.build(level + 1);
  reference_selection_.setRgbdImagePyramid(reference);
  reference_selection_.debug(true);

  std::vector<uint8_t> valid_residuals;

  if (points_error.size() <
      reference_selection_.getMaximumNumberOfPoints(level))
    points_error.resize(reference_selection_.getMaximumNumberOfPoints(level));
  if (residuals.size() < reference_selection_.getMaximumNumberOfPoints(level))
    residuals.resize(reference_selection_.getMaximumNumberOfPoints(level));

  valid_residuals.resize(reference_selection_.getMaximumNumberOfPoints(level));

  PointSelection::PointIterator first_point, last_point;
  reference_selection_.select(level, first_point, last_point);

  RgbdImage &cur = *current.level(level);
  cur.buildAccelerationStructure();
  const auto &K = cur.camera().projection();

  Vector8f wcur, wref;
  // i z idx idy zdx zdy
  float wcur_id = 0.5f, wref_id = 0.5f, wcur_zd = 1.0f, wref_zd = 0.0f;

  wcur << 1.0f / 255.0f, 1.0f, wcur_id *float(K(0, 0)) / 255.0f,
      wcur_id * float(K(1, 1)) / 255.0f, wcur_zd * float(K(0, 0)),
      wcur_zd * float(K(1, 1)), 0.0f, 0.0f;
  wref << -1.0f / 255.0f, -1.0f, wref_id *float(K(0, 0)) / 255.0f,
      wref_id * float(K(1, 1)) / 255.0f, wref_zd * float(K(0, 0)),
      wref_zd * float(K(1, 1)), 0.0f, 0.0f;

  ComputeResidualsResult compute_residuals_result;
  compute_residuals_result.first_point_error = points_error.begin();
  compute_residuals_result.first_residual = residuals.begin();
  compute_residuals_result.first_valid_flag = valid_residuals.begin();

  dvo::core::computeResidualsAndValidFlagsSse(
      first_point, last_point, cur, K, reference2current.cast<float>(), wref,
      wcur, compute_residuals_result);

  cv::Mat result = cv::Mat::zeros(reference.level(level)->intensity().size(),
                                  CV_32FC1),
          debug_idx;

  reference_selection_.getDebugIndex(level, debug_idx);

  uint8_t *valid_pixel_it = debug_idx.ptr<uint8_t>();
  ValidFlagIterator valid_residual_it =
      compute_residuals_result.first_valid_flag;
  ResidualIterator residual_it = compute_residuals_result.first_residual;

  float *result_it = result.ptr<float>();
  float *result_end = result_it + result.total();

  for (; result_it != result_end; ++result_it) {
    if (*valid_pixel_it == 1) {
      if (*valid_residual_it == 1) {
        *result_it = std::abs(residual_it->coeff(0));

        ++residual_it;
      }
      ++valid_residual_it;
    }
    ++valid_pixel_it;
  }

  reference_selection_.debug(false);

  return result;
}

// jacobian computation
inline void DenseTracker::computeJacobianOfProjectionAndTransformation(
    const Vector4 &p, Matrix2x6 &j) {
  NumType z = 1.0f / p(2);
  NumType z_sqr = 1.0f / (p(2) * p(2));

  j(0, 0) = z;
  j(0, 1) = 0.0f;
  j(0, 2) = -p(0) * z_sqr;
  j(0, 3) = j(0, 2) * p(1);         // j(0, 3) = -p(0) * p(1) * z_sqr;
  j(0, 4) = 1.0f - j(0, 2) * p(0);  // j(0, 4) =  (1.0 + p(0) * p(0) * z_sqr);
  j(0, 5) = -p(1) * z;

  j(1, 0) = 0.0f;
  j(1, 1) = z;
  j(1, 2) = -p(1) * z_sqr;
  j(1, 3) = -1.0f + j(1, 2) * p(1);  // j(1, 3) = -(1.0 + p(1) * p(1) * z_sqr);
  j(1, 4) = -j(0, 3);                // j(1, 4) =  p(0) * p(1) * z_sqr;
  j(1, 5) = p(0) * z;
}

inline void DenseTracker::compute3rdRowOfJacobianOfTransformation(
    const Vector4 &p, Vector6 &j) {
  j(0) = 0.0;
  j(1) = 0.0;
  j(2) = 1.0;
  j(3) = p(1);
  j(4) = -p(0);
  j(5) = 0.0;
}

}  // namespace dvo