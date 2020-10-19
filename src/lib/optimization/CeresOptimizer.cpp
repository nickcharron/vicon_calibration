#include "vicon_calibration/optimization/CeresOptimizer.h"
#include "vicon_calibration/optimization/CeresCameraCostFunction.h"
// #include "vicon_calibration/optimization/CeresLidarCostFunction.h"

#include <Eigen/Geometry>

namespace vicon_calibration {

void CeresOptimizer::LoadConfig() {
  std::string config_path = utils::GetFilePathConfig("OptimizerConfig.json");
  LOG_INFO("Loading GTSAM Graph Config file: %s", config_path.c_str());
  nlohmann::json J;
  std::ifstream file(config_path);
  file >> J;
  LoadConfigCommon(J);

  // get gtsam optimizer specific params
  nlohmann::json J_gtsam = J.at("ceres_options");
  ceres_params_.max_num_iterations = J_gtsam.at("max_num_iterations");
  ceres_params_.max_solver_time_in_seconds =
      J_gtsam.at("max_solver_time_in_seconds");
  ceres_params_.function_tolerance = J_gtsam.at("function_tolerance");
  ceres_params_.gradient_tolerance = J_gtsam.at("gradient_tolerance");
  ceres_params_.parameter_tolerance = J_gtsam.at("parameter_tolerance");
  ceres_params_.loss_function = J_gtsam.at("loss_function");
}

void CeresOptimizer::SetupProblem() {
  problem_ = std::make_unique<ceres::Problem>(ceres_problem_options_);

  // set ceres solver params
  ceres_solver_options_.max_num_iterations = ceres_params_.max_num_iterations;
  ceres_solver_options_.max_solver_time_in_seconds =
      ceres_params_.max_solver_time_in_seconds;
  ceres_solver_options_.function_tolerance = ceres_params_.function_tolerance;
  ceres_solver_options_.gradient_tolerance = ceres_params_.gradient_tolerance;
  ceres_solver_options_.parameter_tolerance = ceres_params_.parameter_tolerance;

  // set ceres problem options
  // ceresProblemOptions_.loss_function_ownership =
  // ceres::DO_NOT_TAKE_OWNERSHIP;
  // ceresProblemOptions_.local_parameterization_ownership =
  // ceres::DO_NOT_TAKE_OWNERSHIP;
  if (ceres_params_.loss_function == "HUBER") {
    loss_function_ = std::make_unique<ceres::HuberLoss>(1.0);
  } else if (ceres_params_.loss_function == "CAUCHY") {
    loss_function_ = std::make_unique<ceres::CauchyLoss>(1.0);
  } else if (ceres_params_.loss_function == "NULL") {
    loss_function_ = nullptr;
  } else {
    throw std::invalid_argument{
        "Invalid loss function type. Options: HUBER, CAUCHY, NULL"};
  }
}

void CeresOptimizer::AddInitials() {
  // first, check that number of sensors is not greater than max.
  if (inputs_.calibration_initials.size() > 20) {
    throw std::runtime_error{
        "Number of sensors greater than max allowed. Increase array "
        "sizes in CeresOptimizer.h for: results_, previous_iteration_results_, "
        "and initials_ member variables."};
  }

  SetupProblem();

  // add all sensors as the next poses
  for (uint32_t i = 0; i < inputs_.calibration_initials.size(); i++) {
    vicon_calibration::CalibrationResult calib =
        inputs_.calibration_initials[i];
    Eigen::Matrix4d T_SENSOR_VICONBASE =
        utils::InvertTransform(calib.transform);
    Eigen::Matrix3d R_SENSOR_VICONBASE = T_SENSOR_VICONBASE.block(0, 0, 3, 3);
    Eigen::AngleAxis<double> AA = Eigen::AngleAxis<double>(R_SENSOR_VICONBASE);

    // convert from Angle-Axis to Rodrigues vector representation
    initials_[i][0] = AA.axis()[0] * AA.angle();
    initials_[i][1] = AA.axis()[1] * AA.angle();
    initials_[i][2] = AA.axis()[2] * AA.angle();
    initials_[i][3] = T_SENSOR_VICONBASE(0, 3);
    initials_[i][4] = T_SENSOR_VICONBASE(1, 3);
    initials_[i][5] = T_SENSOR_VICONBASE(2, 3);
  }

  // copy arrays:
  for (uint32_t i = 0; i < inputs_.calibration_initials.size(); i++) {
    for (uint8_t j = 0; j < 6; j++) {
      results_[i][j] = initials_[i][j];
      previous_iteration_results_[i][j] = initials_[i][j];
    }
  }
}

void CeresOptimizer::Clear() {
  if (skip_to_next_iteration_) {
    stop_all_vis_ = false;
    skip_to_next_iteration_ = false;
  }
  SetupProblem();
  camera_correspondences_.clear();
  lidar_correspondences_.clear();
  lidar_camera_correspondences_.clear();
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

  Eigen::Vector3d rodrigues_vector{previous_iteration_results_[index][0],
                                   previous_iteration_results_[index][1],
                                   previous_iteration_results_[index][2]};
  std::pair<Eigen::Vector3d, double> AA_tmp =
      utils::RodriguesToAngleAxis(rodrigues_vector);
  Eigen::AngleAxis<double> AA(AA_tmp.second, AA_tmp.first);

  Eigen::Matrix4d T_SENSOR_VICONBASE = Eigen::Matrix4d::Identity();
  T_SENSOR_VICONBASE.block(0, 0, 3, 3) = AA.toRotationMatrix();
  T_SENSOR_VICONBASE(0, 3) = previous_iteration_results_[index][3];
  T_SENSOR_VICONBASE(1, 3) = previous_iteration_results_[index][4];
  T_SENSOR_VICONBASE(2, 3) = previous_iteration_results_[index][5];

  return utils::InvertTransform(T_SENSOR_VICONBASE);
}

Eigen::Matrix4d CeresOptimizer::GetFinalPose(SensorType type, int id) {
  int index = GetSensorIndex(type, id);

  Eigen::Vector3d rodrigues_vector{previous_iteration_results_[index][0],
                                   previous_iteration_results_[index][1],
                                   previous_iteration_results_[index][2]};
  std::pair<Eigen::Vector3d, double> AA_tmp =
      utils::RodriguesToAngleAxis(rodrigues_vector);
  Eigen::AngleAxis<double> AA(AA_tmp.second, AA_tmp.first);

  Eigen::Matrix4d T_SENSOR_VICONBASE = Eigen::Matrix4d::Identity();
  T_SENSOR_VICONBASE.block(0, 0, 3, 3) = AA.toRotationMatrix();
  T_SENSOR_VICONBASE(0, 3) = results_[index][3];
  T_SENSOR_VICONBASE(1, 3) = results_[index][4];
  T_SENSOR_VICONBASE(2, 3) = results_[index][5];

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
    ceres::CostFunction* cost_function = CeresCameraCostFunction::Create(
        pixel, P_VICONBASE, inputs_.camera_params[camera_index]->camera_model);
    problem_->AddResidualBlock(cost_function, loss_function_.get(),
                               results_[sensor_index]);
  }
  LOG_INFO("Added %d image measurements.", counter);
}

void CeresOptimizer::AddLidarMeasurements() {
  LOG_ERROR("Lidar cost functions not implemented for Ceres solver.");
  // LOG_INFO("Setting lidar factors");
  // Eigen::Vector3d point_predicted, point_measured;
  // int target_index, lidar_index;
  // // TODO: Figure out a smart way to do this. Do we want to tune the COV
  // based
  // // on the number of points per measurement? ALso, shouldn't this be 2x2?
  // gtsam::Vector3 noise_vec;
  // noise_vec << optimizer_params_.lidar_noise[0],
  //     optimizer_params_.lidar_noise[1], optimizer_params_.lidar_noise[2];
  // gtsam::noiseModel::Diagonal::shared_ptr LidarNoise =
  //     gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  // int counter = 0;
  // for (vicon_calibration::Correspondence corr : lidar_correspondences_) {
  //   counter++;
  //   std::shared_ptr<LidarMeasurement> measurement =
  //       inputs_.lidar_measurements[corr.sensor_index][corr.measurement_index];
  //   target_index = measurement->target_id;
  //   lidar_index = measurement->lidar_id;

  //   if (inputs_.target_params[target_index]->keypoints_lidar.size() > 0) {
  //     point_predicted = inputs_.target_params[target_index]
  //                           ->keypoints_lidar[corr.target_point_index];
  //   } else {
  //     point_predicted = utils::PCLPointToEigen(
  //         inputs_.target_params[target_index]->template_cloud->at(
  //             corr.target_point_index));
  //   }

  //   point_measured = utils::PCLPointToEigen(
  //       measurement->keypoints->at(corr.measured_point_index));
  //   gtsam::Key key = gtsam::Symbol('L', lidar_index);
  //   graph_.emplace_shared<LidarFactor>(key, point_measured, point_predicted,
  //                                      measurement->T_VICONBASE_TARGET,
  //                                      LidarNoise);
  // }
  // LOG_INFO("Added %d lidar factors.", counter);
}

void CeresOptimizer::AddLidarCameraMeasurements() {
  LOG_ERROR("Lidar-Camera cost functions not implemented for Ceres solver.");
  // LOG_INFO("Setting lidar-camera factors");
  // gtsam::Vector2 noise_vec;
  // noise_vec << 10, 10;
  // gtsam::noiseModel::Diagonal::shared_ptr noiseModel =
  //     gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  // gtsam::Key lidar_key, camera_key;
  // Eigen::Vector3d point_detected, P_T_li, P_T_ci;
  // int counter = 0;
  // for (LoopCorrespondence corr : lidar_camera_correspondences_) {
  //   counter++;
  //   lidar_key = gtsam::Symbol('L', corr.lidar_id);
  //   camera_key = gtsam::Symbol('C', corr.camera_id);

  //   // get measured point/pixel expressed in sensor frame
  //   Eigen::Vector2d pixel_detected = utils::PCLPixelToEigen(
  //       inputs_.loop_closure_measurements[corr.measurement_index]
  //           ->keypoints_camera->at(corr.camera_measurement_point_index));
  //   point_detected = utils::PCLPointToEigen(
  //       inputs_.loop_closure_measurements[corr.measurement_index]
  //           ->keypoints_lidar->at(corr.lidar_measurement_point_index));

  //   // get corresponding target points expressed in target frames
  //   P_T_ci = inputs_.target_params[corr.target_id]
  //                ->keypoints_camera[corr.camera_target_point_index];
  //   P_T_li = inputs_.target_params[corr.target_id]
  //                ->keypoints_lidar[corr.lidar_target_point_index];

  //   graph_.emplace_shared<CameraLidarFactor>(
  //       lidar_key, camera_key, pixel_detected, point_detected, P_T_ci,
  //       P_T_li, inputs_.camera_params[corr.camera_id]->camera_model,
  //       noiseModel);
  // }
  // LOG_INFO("Added %d lidar-camera factors.", counter);
}

void CeresOptimizer::Optimize() {
  if (optimizer_params_.print_results_to_terminal) {
    LOG_INFO("No. of parameter blocks: %d", problem_->NumParameterBlocks());
    LOG_INFO("No. of parameters: %d", problem_->NumParameters());
    LOG_INFO("No. of residual blocks: %d", problem_->NumResidualBlocks());
    LOG_INFO("No. of residuals: %d", problem_->NumResiduals());
  }

  LOG_INFO("Optimizing Ceres Problem");
  ceres::Solve(ceres_solver_options_, problem_.get(), &ceres_summary_);
  LOG_INFO("Done.");
}

void CeresOptimizer::UpdateInitials() {
  // no need to update initials because ceres has already updated them, but we
  // will need the previous iteration array to be updated
  for (uint32_t i = 0; i < inputs_.calibration_initials.size(); i++) {
    for (uint8_t j = 0; j < 6; j++) {
      previous_iteration_results_[i][j] = results_[i][j];
    }
  }
}

} // end namespace vicon_calibration
