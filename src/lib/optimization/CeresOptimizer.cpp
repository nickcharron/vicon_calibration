#include <vicon_calibration/optimization/CeresOptimizer.h>

#include <boost/filesystem.hpp>

#include <vicon_calibration/optimization/CeresCameraCostFunction.h>
#include <vicon_calibration/optimization/CeresLidarCostFunction.h>

namespace vicon_calibration {

CeresOptimizer::CeresOptimizer(const OptimizerInputs& inputs)
    : Optimizer(inputs), ceres_params_(inputs.ceres_config_path) {}

void CeresOptimizer::SetupProblem() {
  // set ceres problem options
  problem_ = std::make_shared<ceres::Problem>(ceres_params_.ProblemOptions());

  for (int i = 0; i < results_.size(); i++) {
    problem_->AddParameterBlock(
        &(results_[i][0]), 7,
        ceres_params_.SE3QuatTransLocalParametrization().get());
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
  LOG_INFO("Setting lidar measurements");
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
  // auto p = problem_.get();
  // std::cout << "TEST5.0\n";
  // auto p = problem_.get();
  // std::cout << "TEST5.1\n";
  // auto o = ceres_params_.SolverOptions(); 
  // std::cout << "TEST5.1A\n";
  // std::cout << "NumParameterBlocks: " << problem_->NumParameterBlocks() << "\n";
  // std::cout << "NumParameters: " << problem_->NumParameters() << "\n";
  // std::cout << "NumResidualBlocks: " << problem_->NumResidualBlocks() << "\n";
  // std::cout << "NumResiduals: " << problem_->NumResiduals() << "\n";
  // auto s = &ceres_summary_;
  // std::cout << "TEST5.1B\n";
  // ceres::Solve(o, p, s);
  ceres::Solve(ceres_params_.SolverOptions(), problem_.get(), &ceres_summary_);
  // std::cout << "TEST5.2\n";
  if (optimizer_params_.print_results_to_terminal) {
    std::string report = ceres_summary_.FullReport();
    std::cout << report << "\n";
  }
  // std::cout << "TEST5.3\n";
  LOG_INFO("Done.");
}

void CeresOptimizer::UpdateInitials() {
  // no need to update initials because ceres has already updated them, but we
  // will need the previous iteration array to be updated
  previous_iteration_results_ = results_;
}

}  // end namespace vicon_calibration
