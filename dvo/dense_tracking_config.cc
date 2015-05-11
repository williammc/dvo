#include "dvo/dense_tracking.h"
#include <Eigen/Eigenvalues>

namespace dvo {

DenseTracker::Config::Config()
    : FirstLevel(5), LastLevel(1), MaxIterationsPerLevel(100), Precision(5e-7),
      UseInitialEstimate(false), UseWeighting(true), Mu(0),
      InfluenceFuntionType(dvo::core::InfluenceFunctions::TDistribution),
      InfluenceFunctionParam(
          dvo::core::TDistributionInfluenceFunction::DEFAULT_DOF),
      ScaleEstimatorType(dvo::core::ScaleEstimators::TDistribution),
      ScaleEstimatorParam(dvo::core::TDistributionScaleEstimator::DEFAULT_DOF),
      IntensityDerivativeThreshold(10.0f), DepthDerivativeThreshold(0.01f) {}

size_t DenseTracker::Config::getNumLevels() const { return FirstLevel + 1; }

bool DenseTracker::Config::UseEstimateSmoothing() const { return Mu > 1e-6; }

bool DenseTracker::Config::IsSane() const { return FirstLevel >= LastLevel; }

void DenseTracker::Config::Print() const {
  std::cout << "DenseTracker::Config:\n"
      << "First Level = " << FirstLevel
      << ", Last Level = " << LastLevel
      << ", Max Iterations per Level = " << MaxIterationsPerLevel
      << ", Precision = " << Precision << ", Mu = " << Mu
      << ", Use Initial Estimate = "
      << (UseInitialEstimate ? "true" : "false")
      << ", Use Weighting = " << (UseWeighting ? "true" : "false")
      << ", Scale Estimator = "
      << dvo::core::ScaleEstimators::str(ScaleEstimatorType)
      << ", Scale Estimator Param = " << ScaleEstimatorParam
      << ", Influence Function = "
      << dvo::core::InfluenceFunctions::str(InfluenceFuntionType)
      << ", Influence Function Param = " << InfluenceFunctionParam
      << ", Intensity Derivative Threshold = "
      << IntensityDerivativeThreshold
      << ", Depth Derivative Threshold = " << DepthDerivativeThreshold 
      << "\n";
}

// IteractionContext ===========================================================
DenseTracker::IterationContext::IterationContext(const Config &cfg)
    : cfg(cfg) {}

bool DenseTracker::IterationContext::IsFirstIteration() const {
  return IsFirstLevel() && IsFirstIterationOnLevel();
}

bool DenseTracker::IterationContext::IsFirstIterationOnLevel() const {
  return Iteration == 0;
}

bool DenseTracker::IterationContext::IsFirstLevel() const {
  return cfg.FirstLevel == Level;
}

bool DenseTracker::IterationContext::IsLastLevel() const {
  return cfg.LastLevel == Level;
}

double DenseTracker::IterationContext::ErrorDiff() const {
  return LastError - Error;
}

bool DenseTracker::IterationContext::IterationsExceeded() const {
  int max_iterations = cfg.MaxIterationsPerLevel;

  return Iteration >= max_iterations;
}

// Result ======================================================================
bool DenseTracker::Result::isNaN() const {
  return !std::isfinite(Transformation.matrix().sum()) ||
         !std::isfinite(Information.sum());
}

DenseTracker::Result::Result()
    : LogLikelihood(std::numeric_limits<double>::max()) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  Transformation.linear().setConstant(nan);
  Transformation.translation().setConstant(nan);
  Information.setIdentity();
}

void DenseTracker::Result::setIdentity() {
  Transformation.setIdentity();
  Information.setIdentity();
  LogLikelihood = 0.0;
}

void DenseTracker::Result::clearStatistics() { Statistics.Levels.clear(); }

// IterationStats ==============================================================
void DenseTracker::IterationStats::InformationEigenValues(
    dvo::core::Vector6d &eigenvalues) const {
  Eigen::EigenSolver<dvo::core::Matrix6d> evd(EstimateInformation);
  eigenvalues = evd.eigenvalues().real();
  std::sort(eigenvalues.data(), eigenvalues.data() + eigenvalues.rows());
}

double DenseTracker::IterationStats::InformationConditionNumber() const {
  dvo::core::Vector6d ev;
  InformationEigenValues(ev);

  return std::abs(ev(5) / ev(0));
}

// LevelStats ==================================================================
bool DenseTracker::LevelStats::HasIterationWithIncrement() const {
  int min =
      TerminationCriterion ==
                  DenseTracker::TerminationCriteria::LogLikelihoodDecreased ||
              TerminationCriterion ==
                  DenseTracker::TerminationCriteria::TooFewConstraints
          ? 2
          : 1;

  return Iterations.size() >= min;
}

DenseTracker::IterationStats &
DenseTracker::LevelStats::LastIterationWithIncrement() {
  if (!HasIterationWithIncrement()) {
    std::cerr << "awkward " << *this << std::endl;

    assert(false);
  }

  return TerminationCriterion ==
                 DenseTracker::TerminationCriteria::LogLikelihoodDecreased
             ? Iterations[Iterations.size() - 2]
             : Iterations[Iterations.size() - 1];
}

DenseTracker::IterationStats &DenseTracker::LevelStats::LastIteration() {
  return Iterations.back();
}

const DenseTracker::IterationStats &
DenseTracker::LevelStats::LastIterationWithIncrement() const {
  if (!HasIterationWithIncrement()) {
    std::cerr << "awkward " << *this << std::endl;

    assert(false);
  }
  return TerminationCriterion ==
                 DenseTracker::TerminationCriteria::LogLikelihoodDecreased
             ? Iterations[Iterations.size() - 2]
             : Iterations[Iterations.size() - 1];
}

const DenseTracker::IterationStats &
DenseTracker::LevelStats::LastIteration() const {
  return Iterations.back();
}

} // namespace dvo