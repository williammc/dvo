#pragma once

#include <dvo/dense_tracking.h>

namespace dvo {
namespace core {

typedef PointWithIntensityAndDepth::VectorType::iterator PointIterator;
typedef DenseTracker::ResidualVectorType::iterator ResidualIterator;
typedef DenseTracker::WeightVectorType::iterator WeightIterator;
typedef std::vector<uint8_t>::iterator ValidFlagIterator;

struct ComputeResidualsResult {
  PointIterator first_point_error;
  PointIterator last_point_error;

  ResidualIterator first_residual;
  ResidualIterator last_residual;

  ValidFlagIterator first_valid_flag;
  ValidFlagIterator last_valid_flag;
};

void computeResiduals(const PointIterator &first_point,
                      const PointIterator &last_point, const RgbdImage &current,
                      const Eigen::Matrix3d &projection,
                      const Eigen::Affine3f &transform,
                      const Vector8f &reference_weight,
                      const Vector8f &current_weight,
                      ComputeResidualsResult &result);

void computeResidualsSse(
    const PointIterator &first_point, const PointIterator &last_point,
    const RgbdImage &current, const Eigen::Matrix3d &projection,
    const Eigen::Affine3f &transform, const Vector8f &reference_weight,
    const Vector8f &current_weight, ComputeResidualsResult &result);
void computeResidualsAndValidFlagsSse(
    const PointIterator &first_point, const PointIterator &last_point,
    const RgbdImage &current, const Eigen::Matrix3d &projection,
    const Eigen::Affine3f &transform, const Vector8f &reference_weight,
    const Vector8f &current_weight, ComputeResidualsResult &result);

float computeCompleteDataLogLikelihood(const ResidualIterator &first_residual,
                                       const ResidualIterator &last_residual,
                                       const WeightIterator &first_weight,
                                       const Eigen::Vector2f &mean,
                                       const Eigen::Matrix2f &precision);

float computeWeightedError(const ResidualIterator &first_residual,
                           const ResidualIterator &last_residual,
                           const WeightIterator &first_weight,
                           const Eigen::Matrix2f &precision);
float computeWeightedErrorSse(const ResidualIterator &first_residual,
                              const ResidualIterator &last_residual,
                              const WeightIterator &first_weight,
                              const Eigen::Matrix2f &precision);

// Eigen::Vector2f computeMean(const ResidualIterator& first_residual, const
// ResidualIterator& last_residual, const WeightIterator& first_weight);

Eigen::Matrix2f computeScale(const ResidualIterator &first_residual,
                             const ResidualIterator &last_residual,
                             const WeightIterator &first_weight,
                             const Eigen::Vector2f &mean);
Eigen::Matrix2f computeScaleSse(const ResidualIterator &first_residual,
                                const ResidualIterator &last_residual,
                                const WeightIterator &first_weight,
                                const Eigen::Vector2f &mean);

void computeWeights(const ResidualIterator &first_residual,
                    const ResidualIterator &last_residual,
                    const WeightIterator &first_weight,
                    const Eigen::Vector2f &mean,
                    const Eigen::Matrix2f &precision);
void computeWeightsSse(const ResidualIterator &first_residual,
                       const ResidualIterator &last_residual,
                       const WeightIterator &first_weight,
                       const Eigen::Vector2f &mean,
                       const Eigen::Matrix2f &precision);

void computeMeanScaleAndWeights(const ResidualIterator &first_residual,
                                const ResidualIterator &last_residual,
                                const WeightIterator &first_weight,
                                Eigen::Vector2f &mean,
                                Eigen::Matrix2f &precision);

} // namespace core
} // namespace dvo