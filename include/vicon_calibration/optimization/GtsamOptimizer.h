#pragma once

#include "vicon_calibration/optimization/Optimizer.h"
#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

namespace vicon_calibration {

/**
 * @brief base class for solving the optimization problem which defines the
 * interface for any optimizer of choice.
 */
class GtsamOptimizer : public Optimizer {
public:
  // inherit base class constructor
  using Optimizer::Optimizer;

  /**
   * @brief params specific to gtsam optimizer
   */
  struct GtsamParams {
    double abs_error_tol{1e-9};
    double rel_error_tol{1e-9};
    double lambda_upper_bound{
        1e8}; // the maximum lambda to try before assuming
              // the optimization has failed (default: 1e5)
  };

private:
  void LoadConfig() override;

  void AddInitials() override;

  void Reset() override;

  Eigen::Matrix4d GetUpdatedInitialPose(SensorType type, int id) override;

  Eigen::Matrix4d GetFinalPose(SensorType type, int id) override;

  void AddImageMeasurements() override;

  void AddLidarMeasurements() override;

  void AddLidarCameraMeasurements() override;

  void Optimize() override;

  void UpdateInitials() override;

  GtsamParams gtsam_params_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initials_;
  gtsam::Values initials_updated_;
  gtsam::Values results_;
};

} // end namespace vicon_calibration
