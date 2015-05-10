#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "dvo/core/datatypes.h"
#include "dvo/core/rgbd_image.h"
#include "dvo/core/point_selection.h"
#include "dvo/core/least_squares.h"
#include "dvo/core/weight_calculation.h"

namespace dvo {

class DVO_API DenseTracker {
public:
  struct DVO_API Config {
    Config();

    size_t getNumLevels() const;

    bool UseEstimateSmoothing() const;

    bool IsSane() const;

    void Print() const;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar &FirstLevel;
      ar &LastLevel;
      ar &MaxIterationsPerLevel;
      ar &Precision;
      ar &Mu;
      ar &UseInitialEstimate;
      ar &UseParallel;
      ar &InfluenceFuntionType;
      ar &InfluenceFunctionParam;
      ar &IntensityDerivativeThreshold;
      ar &DepthDerivativeThreshold;
    }

    int FirstLevel, LastLevel;
    int MaxIterationsPerLevel;
    double Precision;
    double Mu; // precision (1/sigma^2) of prior

    bool UseInitialEstimate;
    bool UseWeighting;

    bool UseParallel;

    dvo::core::InfluenceFunctions::enum_t InfluenceFuntionType;
    float InfluenceFunctionParam;

    dvo::core::ScaleEstimators::enum_t ScaleEstimatorType;
    float ScaleEstimatorParam;

    float IntensityDerivativeThreshold;
    float DepthDerivativeThreshold;
  };

  struct DVO_API TerminationCriteria {
    enum Enum {
      IterationsExceeded = 0,
      IncrementTooSmall,
      LogLikelihoodDecreased,
      TooFewConstraints,
      NumCriteria
    };
  };

  struct DVO_API IterationStats {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    dvo::core::Vector6d EstimateIncrement;
    dvo::core::Matrix6d EstimateInformation;

    size_t Id, ValidConstraints;

    double TDistributionLogLikelihood;
    Eigen::Vector2d TDistributionMean;
    Eigen::Matrix2d TDistributionPrecision;

    double PriorLogLikelihood;
    double Error;
    double WeightedError;
    double ValidConstraintsRatio;

    void InformationEigenValues(dvo::core::Vector6d &eigenvalues) const;

    double InformationConditionNumber() const;
  };
  using IterationStatsVector = std::vector<IterationStats, Eigen::aligned_allocator<IterationStats>>;

  struct DVO_API LevelStats {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    size_t Id, MaxValidPixels, ValidPixels;
    TerminationCriteria::Enum TerminationCriterion;
    IterationStatsVector Iterations;

    bool HasIterationWithIncrement() const;

    IterationStats &LastIterationWithIncrement();
    IterationStats &LastIteration();

    const IterationStats &LastIterationWithIncrement() const;
    const IterationStats &LastIteration() const;
  };
  using LevelStatsVector = std::vector<LevelStats, Eigen::aligned_allocator<LevelStats>>;

  struct DVO_API Stats {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    LevelStatsVector Levels;
  };

  struct DVO_API Result {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    enum struct TrackStatus : int {
      GOOD = 0,
      OK = 1,
      BAD = 2,
      FAILURE = 3,
      UNDEFINED = 4
    } Status;

    dvo::core::AffineTransformd Transformation;
    dvo::core::Matrix6d Information;
    double LogLikelihood;

    Stats Statistics;

    Result();

    static std::string ToString(TrackStatus st) {
      switch (st) {
      case TrackStatus::GOOD:
        return "GOOD";
      case TrackStatus::OK:
        return "OK";
      case TrackStatus::BAD:
        return "BAD";
      case TrackStatus::FAILURE:
        return "FAILURE";
      default:
        return "UNDEFINED";
      }
    }

    size_t validPixels(size_t level = 0) {
      assert(level < Statistics.Levels.size());
      return Statistics.Levels[level].ValidPixels;
    }

    int isMatchConverted(size_t level = 0) {
      return Statistics.Levels[level]
          .TerminationCriterion; // !=
                                 // TerminationCriteria::LogLikelihoodDecreased;
    }

    bool isNaN() const;
    void setIdentity();
    void clearStatistics();
  };

  static const Config &getDefaultConfig();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DenseTracker(const Config &cfg = getDefaultConfig());
  DenseTracker(const dvo::DenseTracker &other);

  const Config &configuration() const { return cfg; }

  void configure(const Config &cfg);

  bool match(dvo::core::RgbdImagePyramid &reference,
             dvo::core::RgbdImagePyramid &current,
             dvo::core::AffineTransformd &current2reference);
  bool match(dvo::core::PointSelection &reference,
             dvo::core::RgbdImagePyramid &current,
             dvo::core::AffineTransformd &current2reference);

  bool match(dvo::core::RgbdImagePyramid &reference,
             dvo::core::RgbdImagePyramid &current,
             dvo::DenseTracker::Result &result);
  bool match(dvo::core::PointSelection &reference,
             dvo::core::RgbdImagePyramid &current,
             dvo::DenseTracker::Result &result);

  cv::Mat computeIntensityErrorImage(
      dvo::core::RgbdImagePyramid &reference,
      dvo::core::RgbdImagePyramid &current,
      const dvo::core::AffineTransformd &current2reference, size_t level = 0);

  static inline void
  computeJacobianOfProjectionAndTransformation(const dvo::core::Vector4 &p,
                                               dvo::core::Matrix2x6 &jacobian);

  static inline void
  compute3rdRowOfJacobianOfTransformation(const dvo::core::Vector4 &p,
                                          dvo::core::Vector6 &j);

  using ResidualVectorType = std::vector<Eigen::Vector2f,
                      Eigen::aligned_allocator<Eigen::Vector2f>>;
  using WeightVectorType = std::vector<float, Eigen::aligned_allocator<float>>;

private:
  struct IterationContext {
    const Config &cfg;

    int Level;
    int Iteration;

    double Error, LastError;

    IterationContext(const Config &cfg);

    // returns true if this is the first iteration
    bool IsFirstIteration() const;

    // returns true if this is the first iteration on the current level
    bool IsFirstIterationOnLevel() const;

    // returns true if this is the first level
    bool IsFirstLevel() const;

    // returns true if this is the last level
    bool IsLastLevel() const;

    bool IterationsExceeded() const;

    // returns LastError - Error
    double ErrorDiff() const;
  };

  Config cfg;

  IterationContext itctx_;

  dvo::core::WeightCalculation weight_calculation_;
  dvo::core::PointSelection reference_selection_;
  dvo::core::ValidPointAndGradientThresholdPredicate selection_predicate_;

  dvo::core::PointWithIntensityAndDepth::VectorType points, points_error;

  ResidualVectorType residuals;
  WeightVectorType weights;
};

} // namespace dvo

namespace std {
template <typename CharT, typename Traits>
inline std::ostream &operator<<(std::basic_ostream<CharT, Traits> &out,
                                const dvo::DenseTracker::Config &config) {
  out << "First Level = " << config.FirstLevel
      << ", Last Level = " << config.LastLevel
      << ", Max Iterations per Level = " << config.MaxIterationsPerLevel
      << ", Precision = " << config.Precision << ", Mu = " << config.Mu
      << ", Use Initial Estimate = "
      << (config.UseInitialEstimate ? "true" : "false")
      << ", Use Weighting = " << (config.UseWeighting ? "true" : "false")
      << ", Scale Estimator = "
      << dvo::core::ScaleEstimators::str(config.ScaleEstimatorType)
      << ", Scale Estimator Param = " << config.ScaleEstimatorParam
      << ", Influence Function = "
      << dvo::core::InfluenceFunctions::str(config.InfluenceFuntionType)
      << ", Influence Function Param = " << config.InfluenceFunctionParam
      << ", Intensity Derivative Threshold = "
      << config.IntensityDerivativeThreshold
      << ", Depth Derivative Threshold = " << config.DepthDerivativeThreshold;

  return out;
}

template <typename CharT, typename Traits>
inline std::ostream &operator<<(std::basic_ostream<CharT, Traits> &o,
                                const dvo::DenseTracker::IterationStats &s) {
  o << "Iteration: " << s.Id << " ValidConstraints: " << s.ValidConstraints
    << " DataLogLikelihood: " << s.TDistributionLogLikelihood
    << " PriorLogLikelihood: " << s.PriorLogLikelihood << std::endl;

  return o;
}

template <typename CharT, typename Traits>
inline std::ostream &operator<<(std::basic_ostream<CharT, Traits> &o,
                                const dvo::DenseTracker::LevelStats &s) {
  std::string termination;

  switch (s.TerminationCriterion) {
  case dvo::DenseTracker::TerminationCriteria::IterationsExceeded:
    termination = "IterationsExceeded";
    break;
  case dvo::DenseTracker::TerminationCriteria::IncrementTooSmall:
    termination = "IncrementTooSmall";
    break;
  case dvo::DenseTracker::TerminationCriteria::LogLikelihoodDecreased:
    termination = "LogLikelihoodDecreased";
    break;
  case dvo::DenseTracker::TerminationCriteria::TooFewConstraints:
    termination = "TooFewConstraints";
    break;
  default:
    break;
  }

  o << "Level: " << s.Id << " Pixel: " << s.ValidPixels << "/"
    << s.MaxValidPixels << " Termination: " << termination
    << " Iterations: " << s.Iterations.size() << std::endl;

  for (auto iter : s.Iterations) {
    o << iter;
  }

  return o;
}

template <typename CharT, typename Traits>
inline std::ostream &operator<<(std::basic_ostream<CharT, Traits> &o,
                                const dvo::DenseTracker::Stats &s) {
  o << s.Levels.size() << " levels" << std::endl;

  for (auto level : s.Levels) {
    o << level;
  }

  return o;
}

} // namespace std