#pragma once

#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/solver.h>
#include <ceres/types.h>

#include <vicon_calibration/Params.h>
#include <vicon_calibration/Utils.h>
#include <vicon_calibration/optimization/Optimizer.h>

namespace vicon_calibration {

/**
 * @brief base class for solving the optimization problem which defines the
 * interface for any optimizer of choice.
 */
class CeresOptimizer : public Optimizer {
public:
  // Inherit base class constructor
  using Optimizer::Optimizer;

  /**
   * @brief params specific to ceres optimizer
   */
  struct CeresParams {
    bool minimizer_progress_to_stdout{false};
    double max_num_iterations{50};
    double max_solver_time_in_seconds{1e6};
    double function_tolerance{1e-6};
    double gradient_tolerance{1e-10};
    double parameter_tolerance{1e-8};
    std::string loss_function{"HUBER"}; // options: HUBER, CAUCHY, NULL
    std::string linear_solver_type{
        "SPARSE_SCHUR"}; // options: SPARSE_SCHUR, DENSE_SCHUR,
                         // SPARSE_NORMAL_CHOLESKY
    std::string preconditioner_type{
        "SCHUR_JACOBI"}; // options: IDENTITY, JACOBI, SCHUR_JACOBI
  };

private:
  void LoadConfig() override;

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

  CeresParams ceres_params_;
  std::vector<std::vector<double>> results_;
  std::vector<std::vector<double>> previous_iteration_results_;
  std::vector<std::vector<double>> initials_;
  std::unique_ptr<ceres::LossFunction> loss_function_;
  std::unique_ptr<ceres::LocalParameterization> se3_parameterization_;
  std::shared_ptr<ceres::Problem> problem_;
  ceres::Solver::Options ceres_solver_options_;
  ceres::Problem::Options ceres_problem_options_;
  ceres::Solver::Summary ceres_summary_;
};

} // end namespace vicon_calibration
