#include <vicon_calibration/optimization/CeresOptimizer.h>

#include <boost/filesystem.hpp>

#include <vicon_calibration/optimization/CeresCameraCostFunction.h>
#include <vicon_calibration/optimization/CeresLidarCostFunction.h>

namespace vicon_calibration {

void CeresOptimizer::LoadConfig() {
  LOG_INFO("Loading Ceres Optimizer Config file: %s",
           inputs_.optimizer_config_path.c_str());

  nlohmann::json J;
  if (!utils::ReadJson(inputs_.optimizer_config_path, J)) {
    LOG_ERROR("Using default ceres optimizer params.");
    return;
  }

  LoadConfigCommon(J);

  // get ceres optimizer specific params
  try {
    nlohmann::json J_ceres = J.at("ceres_options");
    ceres_solver_options_.minimizer_progress_to_stdout =
        J_ceres.at("minimizer_progress_to_stdout");
    ceres_params_.max_num_iterations = J_ceres.at("max_num_iterations");
    ceres_params_.max_solver_time_in_seconds =
        J_ceres.at("max_solver_time_in_seconds");
    ceres_params_.function_tolerance = J_ceres.at("function_tolerance");
    ceres_params_.gradient_tolerance = J_ceres.at("gradient_tolerance");
    ceres_params_.parameter_tolerance = J_ceres.at("parameter_tolerance");
    ceres_params_.loss_function = J_ceres.at("loss_function");
    ceres_params_.linear_solver_type = J_ceres.at("linear_solver_type");
    ceres_params_.preconditioner_type = J_ceres.at("preconditioner_type");
  } catch (const nlohmann::json::exception& e) {
    LOG_ERROR("Cannot load json, one or more missing parameters. Error: %s",
              e.what());
  }
}

void CeresOptimizer::SetupProblem() {
  // set ceres problem options
  ceres_problem_options_.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres_problem_options_.local_parameterization_ownership =
      ceres::DO_NOT_TAKE_OWNERSHIP;

  problem_ = std::make_shared<ceres::Problem>(ceres_problem_options_);

  // set ceres solver params
  ceres_solver_options_.minimizer_progress_to_stdout =
      ceres_params_.minimizer_progress_to_stdout;
  ceres_solver_options_.max_num_iterations = ceres_params_.max_num_iterations;
  ceres_solver_options_.max_solver_time_in_seconds =
      ceres_params_.max_solver_time_in_seconds;
  ceres_solver_options_.function_tolerance = ceres_params_.function_tolerance;
  ceres_solver_options_.gradient_tolerance = ceres_params_.gradient_tolerance;
  ceres_solver_options_.parameter_tolerance = ceres_params_.parameter_tolerance;

  if (ceres_params_.linear_solver_type == "SPARSE_SCHUR") {
    ceres_solver_options_.linear_solver_type = ceres::SPARSE_SCHUR;
  } else if (ceres_params_.linear_solver_type == "DENSE_SCHUR") {
    ceres_solver_options_.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (ceres_params_.linear_solver_type == "SPARSE_NORMAL_CHOLESKY") {
    ceres_solver_options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  } else {
    LOG_ERROR(
        "Invalid linear_solver_type, Options: SPARSE_SCHUR, DENSE_SCHUR, "
        "SPARSE_NORMAL_CHOLESKY. Using default: SPARSE_SCHUR");
    ceres_solver_options_.linear_solver_type = ceres::SPARSE_SCHUR;
  }
  if (ceres_params_.preconditioner_type == "IDENTITY") {
    ceres_solver_options_.preconditioner_type = ceres::IDENTITY;
  } else if (ceres_params_.preconditioner_type == "JACOBI") {
    ceres_solver_options_.preconditioner_type = ceres::JACOBI;
  } else if (ceres_params_.preconditioner_type == "SCHUR_JACOBI") {
    ceres_solver_options_.preconditioner_type = ceres::SCHUR_JACOBI;
  } else {
    LOG_ERROR(
        "Invalid preconditioner_type, Options: IDENTITY, JACOBI, "
        "SCHUR_JACOBI. Using default: SCHUR_JACOBI");
    ceres_solver_options_.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  // set loss function
  if (ceres_params_.loss_function == "HUBER") {
    loss_function_ =
        std::unique_ptr<ceres::LossFunction>(new ceres::HuberLoss(1.0));
  } else if (ceres_params_.loss_function == "CAUCHY") {
    loss_function_ =
        std::unique_ptr<ceres::LossFunction>(new ceres::CauchyLoss(1.0));
  } else if (ceres_params_.loss_function == "NULL") {
    loss_function_ = std::unique_ptr<ceres::LossFunction>(nullptr);
  } else {
    LOG_ERROR(
        "Invalid preconditioner_type, Options: HUBER, CAUCHY, NULL. "
        "Using default: HUBER");
    loss_function_ =
        std::unique_ptr<ceres::LossFunction>(new ceres::HuberLoss(1.0));
  }

  // set local parameterization
  std::unique_ptr<ceres::LocalParameterization> quat_parameterization(
      new ceres::QuaternionParameterization());
  std::unique_ptr<ceres::LocalParameterization> identity_parameterization(
      new ceres::IdentityParameterization(3));
  se3_parameterization_ = std::unique_ptr<ceres::LocalParameterization>(
      new ceres::ProductParameterization(quat_parameterization.release(),
                                         identity_parameterization.release()));

  for (int i = 0; i < results_.size(); i++) {
    problem_->AddParameterBlock(&(results_[i][0]), 7,
                                se3_parameterization_.get());
  }
}

void CeresOptimizer::AddInitials() {
  // add all sensors as the next poses
  for (uint32_t i = 0; i < inputs_.calibration_initials.size(); i++) {
    vicon_calibration::CalibrationResult calib =
        inputs_.calibration_initials[i];
    Eigen::Matrix4d T_SENSOR_VICONBASE =
        utils::InvertTransform(calib.transform);
    Eigen::Matrix3d R = T_SENSOR_VICONBASE.block(0, 0, 3, 3);
    Eigen::Quaternion<double> q = Eigen::Quaternion<double>(R);
    initials_.push_back(std::vector<double>{
        q.w(), q.x(), q.y(), q.z(), T_SENSOR_VICONBASE(0, 3),
        T_SENSOR_VICONBASE(1, 3), T_SENSOR_VICONBASE(2, 3)});
  }

  // copy arrays:
  results_ = initials_;
  previous_iteration_results_ = initials_;
}

void CeresOptimizer::Reset() {
  if (skip_to_next_iteration_) {
    stop_all_vis_ = false;
    skip_to_next_iteration_ = false;
  }
  camera_correspondences_.clear();
  lidar_correspondences_.clear();
  SetupProblem();
}

int CeresOptimizer::GetSensorIndex(SensorType type, int id) {
  for (uint32_t i = 0; i < inputs_.calibration_initials.size(); i++) {
    vicon_calibration::CalibrationResult calib =
        inputs_.calibration_initials[i];
    if (calib.type == type && calib.sensor_id == id) {
      return i;
    }
  }
  throw std::runtime_error{"Queried sensor type and ID not found."};
  return 0;
}

Eigen::Matrix4d CeresOptimizer::GetUpdatedInitialPose(SensorType type, int id) {
  int index = GetSensorIndex(type, id);
  Eigen::Matrix4d T_SENSOR_VICONBASE =
      utils::QuaternionAndTranslationToTransformMatrix(
          previous_iteration_results_[index]);
  return utils::InvertTransform(T_SENSOR_VICONBASE);
}

Eigen::Matrix4d CeresOptimizer::GetFinalPose(SensorType type, int id) {
  int index = GetSensorIndex(type, id);
  Eigen::Matrix4d T_SENSOR_VICONBASE =
      utils::QuaternionAndTranslationToTransformMatrix(results_[index]);
  return utils::InvertTransform(T_SENSOR_VICONBASE);
}

void CeresOptimizer::AddImageMeasurements() {
  LOG_INFO("Setting image measurements");
  int counter = 0;

  for (vicon_calibration::Correspondence corr : camera_correspondences_) {
    counter++;
    std::shared_ptr<CameraMeasurement> measurement =
        inputs_.camera_measurements[corr.sensor_index][corr.measurement_index];
    int target_index = measurement->target_id;
    int camera_index = measurement->camera_id;
    int sensor_index = GetSensorIndex(SensorType::CAMERA, camera_index);

    Eigen::Vector3d P_TARGET;
    if (inputs_.target_params[target_index]->keypoints_camera.size() > 0) {
      P_TARGET = inputs_.target_params[target_index]
                     ->keypoints_camera[corr.target_point_index];
    } else {
      P_TARGET = utils::PCLPointToEigen(
          inputs_.target_params[target_index]->template_cloud->at(
              corr.target_point_index));
    }

    Eigen::Vector2d pixel = utils::PCLPixelToEigen(
        measurement->keypoints->at(corr.measured_point_index));

    Eigen::Vector3d P_VICONBASE =
        (measurement->T_VICONBASE_TARGET * P_TARGET.homogeneous())
            .hnormalized();

    std::unique_ptr<ceres::CostFunction> cost_function(
        CeresCameraCostFunction::Create(
            pixel, P_VICONBASE,
            inputs_.camera_params[camera_index]->camera_model));

    problem_->AddResidualBlock(cost_function.release(), loss_function_.get(),
                               &(results_[sensor_index][0]));
  }
  LOG_INFO("Added %d image measurements.", counter);
}

void CeresOptimizer::AddLidarMeasurements() {
  LOG_INFO("Setting lidar factors");
  int counter = 0;
  for (vicon_calibration::Correspondence corr : lidar_correspondences_) {
    counter++;
    std::shared_ptr<LidarMeasurement> measurement =
        inputs_.lidar_measurements[corr.sensor_index][corr.measurement_index];
    int target_index = measurement->target_id;
    int lidar_index = measurement->lidar_id;
    int sensor_index = GetSensorIndex(SensorType::LIDAR, lidar_index);

    Eigen::Vector3d P_TARGET;
    if (inputs_.target_params[target_index]->keypoints_lidar.size() > 0) {
      P_TARGET = inputs_.target_params[target_index]
                     ->keypoints_lidar[corr.target_point_index];
    } else {
      P_TARGET = utils::PCLPointToEigen(
          inputs_.target_params[target_index]->template_cloud->at(
              corr.target_point_index));
    }
    Eigen::Vector3d P_VICONBASE =
        (measurement->T_VICONBASE_TARGET * P_TARGET.homogeneous())
            .hnormalized();

    Eigen::Vector3d point_measured = utils::PCLPointToEigen(
        measurement->keypoints->at(corr.measured_point_index));

    std::unique_ptr<ceres::CostFunction> cost_function(
        CeresLidarCostFunction::Create(point_measured, P_VICONBASE));

    problem_->AddResidualBlock(cost_function.release(), loss_function_.get(),
                               &(results_[sensor_index][0]));
  }
  LOG_INFO("Added %d lidar measurements.", counter);
}

void CeresOptimizer::Optimize() {
  LOG_INFO("Optimizing Ceres Problem");
  ceres::Solve(ceres_solver_options_, problem_.get(), &ceres_summary_);
  if (optimizer_params_.print_results_to_terminal) {
    std::string report = ceres_summary_.FullReport();
    std::cout << report << "\n";
  }
  LOG_INFO("Done.");
}

void CeresOptimizer::UpdateInitials() {
  // no need to update initials because ceres has already updated them, but we
  // will need the previous iteration array to be updated
  previous_iteration_results_ = results_;
}

}  // end namespace vicon_calibration
