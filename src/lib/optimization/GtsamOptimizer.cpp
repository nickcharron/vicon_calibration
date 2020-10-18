#include "vicon_calibration/optimization/GtsamOptimizer.h"
#include "vicon_calibration/optimization/GtsamCameraFactor.h"
#include "vicon_calibration/optimization/GtsamCameraFactorInv.h"
#include "vicon_calibration/optimization/GtsamCameraLidarFactor.h"
#include "vicon_calibration/optimization/GtsamLidarFactor.h"
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace vicon_calibration {

void GtsamOptimizer::LoadConfig() {
  std::string config_path = utils::GetFilePathConfig("OptimizerConfig.json");
  LOG_INFO("Loading GTSAM Graph Config file: %s", config_path.c_str());
  nlohmann::json J;
  std::ifstream file(config_path);
  file >> J;
  LoadConfigCommon(J);

  // get gtsam optimizer specific params
  nlohmann::json J_gtsam = J.at("gtsam_options");
  gtsam_params_.abs_error_tol = J_gtsam.at("abs_error_tol");
  gtsam_params_.rel_error_tol = J_gtsam.at("rel_error_tol");
  gtsam_params_.lambda_upper_bound = J_gtsam.at("lambda_upper_bound");
}

void GtsamOptimizer::AddInitials() {
  // add all sensors as the next poses
  for (uint32_t i = 0; i < inputs_.calibration_initials.size(); i++) {
    vicon_calibration::CalibrationResult calib =
        inputs_.calibration_initials[i];
    // Eigen::Matrix4d T_SENSOR_VICONBASE =
    //     utils::InvertTransform(calib.transform);
    Eigen::Matrix4d T_VICONBASE_SENSOR = calib.transform;
    gtsam::Pose3 initial_pose(T_VICONBASE_SENSOR);
    if (calib.type == SensorType::LIDAR) {
      initials_.insert(gtsam::Symbol('L', calib.sensor_id), initial_pose);
    } else if (calib.type == SensorType::CAMERA) {
      initials_.insert(gtsam::Symbol('C', calib.sensor_id), initial_pose);
    } else {
      throw std::invalid_argument{
          "Wrong type of sensor inputted as initial calibration estimate."};
    }
  }
  initials_updated_ = initials_;
}

void GtsamOptimizer::Clear() {
  if (skip_to_next_iteration_) {
    stop_all_vis_ = false;
    skip_to_next_iteration_ = false;
  }
  graph_.erase(graph_.begin(), graph_.end());
  results_.clear();
  camera_correspondences_.clear();
  lidar_correspondences_.clear();
  lidar_camera_correspondences_.clear();
}

Eigen::Matrix4d GtsamOptimizer::GetUpdatedInitialPose(SensorType type, int id) {
  gtsam::Pose3 T_VICONBASE_SENSOR;
  // gtsam::Pose3 T_SENSOR_VICONBASE;
  if (type == SensorType::LIDAR) {
    T_VICONBASE_SENSOR =
        initials_updated_.at<gtsam::Pose3>(gtsam::Symbol('L', id));
  } else if (type == SensorType::CAMERA) {
    T_VICONBASE_SENSOR =
        initials_updated_.at<gtsam::Pose3>(gtsam::Symbol('C', id));
  } else {
    throw std::runtime_error{"Invalid sensor type."};
  }
  // return utils::InvertTransform(T_SENSOR_VICONBASE.matrix());
  return T_VICONBASE_SENSOR.matrix();
}

Eigen::Matrix4d GtsamOptimizer::GetFinalPose(SensorType type, int id) {
  // gtsam::Pose3 T_SENSOR_VICONBASE;
  gtsam::Pose3 T_VICONBASE_SENSOR;
  if (type == SensorType::LIDAR) {
    T_VICONBASE_SENSOR = results_.at<gtsam::Pose3>(gtsam::Symbol('L', id));
  } else if (type == SensorType::CAMERA) {
    T_VICONBASE_SENSOR = results_.at<gtsam::Pose3>(gtsam::Symbol('C', id));
  } else {
    throw std::runtime_error{"Invalid sensor type."};
  }
  // return utils::InvertTransform(T_SENSOR_VICONBASE.matrix());
  return T_VICONBASE_SENSOR.matrix();
}

void GtsamOptimizer::AddImageMeasurements() {
  LOG_INFO("Setting image factors");
  int counter = 0;
  int target_index, camera_index;

  // TODO: Figure out a smart way to do this. Do we want to tune the COV based
  // on the number of points per measurement?
  gtsam::Vector2 noise_vec(optimizer_params_.image_noise[0],
                           optimizer_params_.image_noise[1]);
  gtsam::noiseModel::Diagonal::shared_ptr ImageNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  for (vicon_calibration::Correspondence corr : camera_correspondences_) {
    counter++;
    std::shared_ptr<CameraMeasurement> measurement =
        inputs_.camera_measurements[corr.sensor_index][corr.measurement_index];
    target_index = measurement->target_id;
    camera_index = measurement->camera_id;

    Eigen::Vector3d point;
    if (inputs_.target_params[target_index]->keypoints_camera.size() > 0) {
      point = inputs_.target_params[target_index]
                  ->keypoints_camera[corr.target_point_index];
    } else {
      point = utils::PCLPointToEigen(
          inputs_.target_params[target_index]->template_cloud->at(
              corr.target_point_index));
    }

    Eigen::Vector2d pixel = utils::PCLPixelToEigen(
        measurement->keypoints->at(corr.measured_point_index));
    gtsam::Key key = gtsam::Symbol('C', camera_index);
    graph_.emplace_shared<CameraFactorInv>(
        key, pixel, point, inputs_.camera_params[camera_index]->camera_model,
        measurement->T_VICONBASE_TARGET, ImageNoise);
  }
  LOG_INFO("Added %d image factors.", counter);
}

void GtsamOptimizer::AddLidarMeasurements() {
  LOG_INFO("Setting lidar factors");
  Eigen::Vector3d point_predicted, point_measured;
  int target_index, lidar_index;
  // TODO: Figure out a smart way to do this. Do we want to tune the COV based
  // on the number of points per measurement? ALso, shouldn't this be 2x2?
  gtsam::Vector3 noise_vec;
  noise_vec << optimizer_params_.lidar_noise[0],
      optimizer_params_.lidar_noise[1], optimizer_params_.lidar_noise[2];
  gtsam::noiseModel::Diagonal::shared_ptr LidarNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  int counter = 0;
  for (vicon_calibration::Correspondence corr : lidar_correspondences_) {
    counter++;
    std::shared_ptr<LidarMeasurement> measurement =
        inputs_.lidar_measurements[corr.sensor_index][corr.measurement_index];
    target_index = measurement->target_id;
    lidar_index = measurement->lidar_id;

    if (inputs_.target_params[target_index]->keypoints_lidar.size() > 0) {
      point_predicted = inputs_.target_params[target_index]
                            ->keypoints_lidar[corr.target_point_index];
    } else {
      point_predicted = utils::PCLPointToEigen(
          inputs_.target_params[target_index]->template_cloud->at(
              corr.target_point_index));
    }

    point_measured = utils::PCLPointToEigen(
        measurement->keypoints->at(corr.measured_point_index));
    gtsam::Key key = gtsam::Symbol('L', lidar_index);
    graph_.emplace_shared<LidarFactor>(key, point_measured, point_predicted,
                                       measurement->T_VICONBASE_TARGET,
                                       LidarNoise);
  }
  LOG_INFO("Added %d lidar factors.", counter);
}

void GtsamOptimizer::AddLidarCameraMeasurements() {
  LOG_INFO("Setting lidar-camera factors");
  gtsam::Vector2 noise_vec;
  noise_vec << 10, 10;
  gtsam::noiseModel::Diagonal::shared_ptr noiseModel =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  gtsam::Key lidar_key, camera_key;
  Eigen::Vector3d point_detected, P_T_li, P_T_ci;
  int counter = 0;
  for (LoopCorrespondence corr : lidar_camera_correspondences_) {
    counter++;
    lidar_key = gtsam::Symbol('L', corr.lidar_id);
    camera_key = gtsam::Symbol('C', corr.camera_id);

    // get measured point/pixel expressed in sensor frame
    Eigen::Vector2d pixel_detected = utils::PCLPixelToEigen(
        inputs_.loop_closure_measurements[corr.measurement_index]
            ->keypoints_camera->at(corr.camera_measurement_point_index));
    point_detected = utils::PCLPointToEigen(
        inputs_.loop_closure_measurements[corr.measurement_index]
            ->keypoints_lidar->at(corr.lidar_measurement_point_index));

    // get corresponding target points expressed in target frames
    P_T_ci = inputs_.target_params[corr.target_id]
                 ->keypoints_camera[corr.camera_target_point_index];
    P_T_li = inputs_.target_params[corr.target_id]
                 ->keypoints_lidar[corr.lidar_target_point_index];

    graph_.emplace_shared<CameraLidarFactor>(
        lidar_key, camera_key, pixel_detected, point_detected, P_T_ci, P_T_li,
        inputs_.camera_params[corr.camera_id]->camera_model, noiseModel);
  }
  LOG_INFO("Added %d lidar-camera factors.", counter);
}

void GtsamOptimizer::Optimize() {
  LOG_INFO("Optimizing Gtsam graph");
  gtsam::LevenbergMarquardtParams params;
  params.setVerbosity("TERMINATION");
  params.absoluteErrorTol = gtsam_params_.abs_error_tol;
  params.relativeErrorTol = gtsam_params_.rel_error_tol;
  params.setlambdaUpperBound(gtsam_params_.lambda_upper_bound);
  gtsam::KeyFormatter key_formatter = gtsam::DefaultKeyFormatter;
  gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initials_updated_,
                                               params);
  results_.clear();
  std::exception_ptr eptr;

  try {
    results_ = optimizer.optimize();
  } catch (...) {
    LOG_ERROR("Error optimizing GTSAM Graph. Printing graph and initial "
              "estimates to terminal.");
    graph_.print();
    initials_.print();
    eptr = std::current_exception();
    std::rethrow_exception(eptr);
  }
}

void GtsamOptimizer::UpdateInitials() {
  LOG_INFO("Updating initials");
  initials_updated_ = results_;
}

} // end namespace vicon_calibration
