#pragma once

#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/solver.h>
#include <ceres/types.h>

#include <vicon_calibration/Params.h>
#include <vicon_calibration/Utils.h>
#include <vicon_calibration/CeresParams.h>
#include <vicon_calibration/optimization/Optimizer.h>

namespace vicon_calibration {

/**
 * @brief base class for solving the optimization problem which defines the
 * interface for any optimizer of choice.
 */
class CeresOptimizer : public Optimizer {
public:
  CeresOptimizer(const OptimizerInputs& inputs);

private:
  void SetupProblem();

  void AddInitials() override;

  void Reset() override;

  int GetSensorIndex(SensorType type, int id);

  Eigen::Matrix4d GetUpdatedInitialPose(SensorType type, int id) override;

  Eigen::Matrix4d GetFinalPose(SensorType type, int id) override;

  void AddImageMeasurements() override;

  void AddLidarMeasurements() override;

  void Optimize() override;

  void UpdateInitials() override;

  std::vector<std::vector<double>> results_;
  std::vector<std::vector<double>> previous_iteration_results_;
  std::vector<std::vector<double>> initials_;
  std::shared_ptr<ceres::Problem> problem_;
  CeresParams ceres_params_;
  ceres::Solver::Summary ceres_summary_;
  std::unique_ptr<ceres::LossFunction> loss_function_;
  std::unique_ptr<ceres::LocalParameterization> parameterization_;
};

} // end namespace vicon_calibration
