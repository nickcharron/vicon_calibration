#include <vicon_calibration/optimization/CeresOptimizer.h>

#include <vicon_calibration/optimization/CeresCameraCostFunction.h>
#include <vicon_calibration/optimization/CeresLidarCostFunction.h>

namespace vicon_calibration {

CeresOptimizer::CeresOptimizer(const OptimizerInputs& inputs)
    : Optimizer(inputs), ceres_params_(inputs.ceres_config_path) {}

void CeresOptimizer::SetupProblem() {
  // set ceres problem options
  problem_ = std::make_shared<ceres::Problem>(ceres_params_.ProblemOptions());
  parameterization_ = ceres_params_.SE3QuatTransLocalParametrization();
  loss_function_ = ceres_params_.LossFunction();
  for (int i = 0; i < results_.size(); i++) {
    problem_->AddParameterBlock(&(results_[i][0]), 7, parameterization_.get());
  }
  for (int i = 0; i < target_camera_corrections_.size(); i++) {
    problem_->AddParameterBlock(&(target_camera_corrections_[i][0]), 7,
                                parameterization_.get());
    if (!optimizer_params_.estimate_target_camera_corrections) {
      problem_->SetParameterBlockConstant(&(target_camera_corrections_[i][0]));
    }
  }
  for (int i = 0; i < target_lidar_corrections_.size(); i++) {
    problem_->AddParameterBlock(&(target_lidar_corrections_[i][0]), 7,
                                parameterization_.get());
    if (!optimizer_params_.estimate_target_lidar_corrections) {
      problem_->SetParameterBlockConstant(&(target_lidar_corrections_[i][0]));
    }
  }
}

void CeresOptimizer::AddInitials() {
  // add all sensors as the next poses
  for (uint32_t i = 0; i < inputs_.calibration_initials.size(); i++) {
    vicon_calibration::CalibrationResult calib =
        inputs_.calibration_initials[i];
    Eigen::Matrix4d T_Sensor_Robot = utils::InvertTransform(calib.transform);
    Eigen::Matrix3d R = T_Sensor_Robot.block(0, 0, 3, 3);
    Eigen::Quaternion<double> q = Eigen::Quaternion<double>(R);
    initials_.push_back(
        std::vector<double>{q.w(), q.x(), q.y(), q.z(), T_Sensor_Robot(0, 3),
                            T_Sensor_Robot(1, 3), T_Sensor_Robot(2, 3)});
  }

  // add target corrections
  for (uint32_t i = 0; i < inputs_.target_params.size(); i++) {
    target_camera_corrections_.push_back(
        std::vector<double>{1, 0, 0, 0, 0, 0, 0});
    target_lidar_corrections_.push_back(
        std::vector<double>{1, 0, 0, 0, 0, 0, 0});
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
    if (calib.type == type && calib.sensor_id == id) { return i; }
  }
  throw std::runtime_error{"Queried sensor type and ID not found."};
  return 0;
}

Eigen::Matrix4d CeresOptimizer::GetUpdatedInitialPose(SensorType type, int id) {
  int index = GetSensorIndex(type, id);
  Eigen::Matrix4d T_Sensor_Robot =
      utils::QuaternionAndTranslationToTransformMatrix(
          previous_iteration_results_[index]);
  return utils::InvertTransform(T_Sensor_Robot);
}

Eigen::Matrix4d CeresOptimizer::GetFinalPose(SensorType type, int id) {
  int index = GetSensorIndex(type, id);
  Eigen::Matrix4d T_Sensor_Robot =
      utils::QuaternionAndTranslationToTransformMatrix(results_[index]);
  return utils::InvertTransform(T_Sensor_Robot);
}

std::vector<Eigen::Matrix4d> CeresOptimizer::GetTargetCameraCorrections() {
  std::vector<Eigen::Matrix4d> corrections;
  for (const auto& target_correction : target_camera_corrections_) {
    Eigen::Matrix4d T =
        utils::QuaternionAndTranslationToTransformMatrix(target_correction);
    corrections.push_back(T);
  }
  return corrections;
}

std::vector<Eigen::Matrix4d> CeresOptimizer::GetTargetLidarCorrections() {
  std::vector<Eigen::Matrix4d> corrections;
  for (const auto& target_correction : target_lidar_corrections_) {
    Eigen::Matrix4d T =
        utils::QuaternionAndTranslationToTransformMatrix(target_correction);
    corrections.push_back(T);
  }
  return corrections;
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

    Eigen::Vector3d P_Target;
    if (inputs_.target_params[target_index]->keypoints_camera.cols() > 0) {
      P_Target = inputs_.target_params[target_index]->keypoints_camera.col(
          corr.target_point_index);
    } else {
      if (inputs_.target_params[target_index]->template_cloud == nullptr) {
        throw std::runtime_error{"No camera keypoints"};
      }
      const auto& p = inputs_.target_params[target_index]->template_cloud->at(
          corr.target_point_index);
      P_Target = Eigen::Vector3d(p.x, p.y, p.z);
    }

    Eigen::Vector2d pixel(
        measurement->keypoints->at(corr.measured_point_index).x,
        measurement->keypoints->at(corr.measured_point_index).y);

    std::unique_ptr<ceres::CostFunction> cost_function(
        CeresCameraCostFunction::Create(
            pixel, P_Target, measurement->T_Robot_Target,
            inputs_.camera_params[camera_index]->camera_model));

    problem_->AddResidualBlock(cost_function.release(), loss_function_.get(),
                               &(results_[sensor_index][0]),
                               &(target_camera_corrections_[target_index][0]));
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

    Eigen::Vector3d P_Target;
    const auto& target_params = inputs_.target_params[target_index];
    if (target_params->use_lidar_keypoints) {
      P_Target = target_params->keypoints_lidar.col(corr.target_point_index);
    } else {
      const auto& p =
          target_params->template_cloud->at(corr.target_point_index);
      P_Target = Eigen::Vector3d(p.x, p.y, p.z);
    }

    const auto& p = measurement->keypoints->at(corr.measured_point_index);
    Eigen::Vector3d point_measured(p.x, p.y, p.z);

    std::unique_ptr<ceres::CostFunction> cost_function(
        CeresLidarCostFunction::Create(point_measured, P_Target,
                                       measurement->T_Robot_Target));

    problem_->AddResidualBlock(cost_function.release(), loss_function_.get(),
                               &(results_[sensor_index][0]),
                               &(target_lidar_corrections_[target_index][0]));
  }
  LOG_INFO("Added %d lidar measurements.", counter);
}

void CeresOptimizer::Optimize() {
  LOG_INFO("Optimizing Ceres Problem");
  ceres::Solve(ceres_params_.SolverOptions(), problem_.get(), &ceres_summary_);

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

} // end namespace vicon_calibration
