#include "vicon_calibration/GTSAMGraph.h"
#include "vicon_calibration/CameraFactor.h"
#include "vicon_calibration/CameraLidarFactor.h"
#include "vicon_calibration/LidarFactor.h"
#include "vicon_calibration/utils.h"
#include <algorithm>
#include <fstream>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <pcl/registration/correspondence_estimation.h>

namespace vicon_calibration {

void GTSAMGraph::SetTargetParams(
    std::vector<std::shared_ptr<vicon_calibration::TargetParams>> &target_params) {
  target_params_ = target_params;
}

void GTSAMGraph::SetLidarMeasurements(
    std::vector<vicon_calibration::LidarMeasurement> &lidar_measurements) {
  lidar_measurements_ = lidar_measurements;
}

void GTSAMGraph::SetCameraMeasurements(
    std::vector<vicon_calibration::CameraMeasurement> &camera_measurements) {
  camera_measurements_ = camera_measurements;
}

void GTSAMGraph::SetInitialGuess(
    std::vector<vicon_calibration::CalibrationResult> &initial_guess) {
  calibration_initials_ = initial_guess;
}

void GTSAMGraph::SetCameraParams(
    std::vector<std::shared_ptr<vicon_calibration::CameraParams>> &camera_params) {
  camera_params_ = camera_params;
  for (uint16_t i = 0; i < camera_params_.size(); i++) {
    std::shared_ptr<beam_calibration::CameraModel> cam_pointer;
    cam_pointer =
        beam_calibration::CameraModel::LoadJSON(camera_params_[i]->intrinsics);
    camera_models_.push_back(cam_pointer);
  }
}

void GTSAMGraph::SolveGraph() {
  CheckInputs();
  Clear();
  AddInitials();
  initials_updated_ = initials_;
  // AddLidarMeasurements();
  uint16_t iteration = 0;
  while (!CheckConvergence() && (iteration < max_iterations_)) {
    iteration++;
    SetImageCorrespondences();
    SetLidarCorrespondences();
    SetImageFactors();
    SetLidarFactors();
    // SetLidarCameraFactors();
    Optimize();
    initials_updated_ = results_;
  }
  if (iteration >= max_iterations_) {
    LOG_WARN("Reached max iterations, stopping.");
  }
}

std::vector<vicon_calibration::CalibrationResult> GTSAMGraph::GetResults() {
  for (uint32_t i = 0; i < calibration_initials_.size(); i++) {
    vicon_calibration::CalibrationResult calib;
    calib.to_frame = calibration_initials_[i].to_frame;
    calib.from_frame = calibration_initials_[i].from_frame;
    gtsam::Key sensor_key = i + 1;
    calib.transform = results_.at<gtsam::Pose3>(sensor_key).matrix();
    calibration_results_.push_back(calib);
  }
  return calibration_results_;
}

void GTSAMGraph::Print(std::string &file_name, bool print_to_terminal) {
  if (print_to_terminal) {
    graph_.print();
    gtsam::KeyFormatter print_format = gtsam::DefaultKeyFormatter;
    initials_.print("Printing Initial Values: \n", print_format);
    results_.print("Printing Result Values: \n", print_format);
  }
  std::ofstream graph_file(file_name);
  graph_.saveGraph(graph_file);
  graph_file.close();
}

bool GTSAMGraph::CheckConvergence() {
  new_error_ = graph_.error(results_);

  if (output_errors_) {
    if (new_error_ <= error_tol_)
      std::cout << "error_tol_: " << new_error_ << " < " << error_tol_
                << std::endl;
    else
      std::cout << "error_tol_: " << new_error_ << " > " << error_tol_
                << std::endl;
  }

  if (new_error_ <= error_tol_)
    return true;

  // check if diverges
  double absolute_decrease = current_error_ - new_error_;
  if (output_errors_) {
    if (absolute_decrease <= absolute_error_tol_)
      std::cout << "absolute_decrease: " << std::setprecision(12)
                << absolute_decrease << " < " << absolute_error_tol_
                << std::endl;
    else
      std::cout << "absolute_decrease: " << std::setprecision(12)
                << absolute_decrease << " >= " << absolute_error_tol_
                << std::endl;
  }

  // calculate relative error decrease and update current_error_
  double relative_decrease = absolute_decrease / current_error_;
  if (output_errors_) {
    if (relative_decrease <= relative_error_tol_)
      std::cout << "relative_decrease: " << std::setprecision(12)
                << relative_decrease << " < " << relative_error_tol_
                << std::endl;
    else
      std::cout << "relative_decrease: " << std::setprecision(12)
                << relative_decrease << " >= " << relative_error_tol_
                << std::endl;
  }

  bool converged =
      (relative_error_tol_ && (relative_decrease <= relative_error_tol_)) ||
      (absolute_decrease <= absolute_error_tol_);

  if (converged) {
    if (absolute_decrease >= 0.0)
      std::cout << "converged" << std::endl;
    else
      std::cout
          << "Warning:  stopping nonlinear iterations because error increased"
          << std::endl;

    std::cout << "error_tol_: " << new_error_ << " <? " << error_tol_
              << std::endl;
    std::cout << "absolute_decrease: " << std::setprecision(12)
              << absolute_decrease << " <? " << absolute_error_tol_
              << std::endl;
    std::cout << "relative_decrease: " << std::setprecision(12)
              << relative_decrease << " <? " << relative_error_tol_
              << std::endl;
  }
  current_error_ = new_error_;
  return converged;
}

void GTSAMGraph::CheckInputs() {
  if (lidar_measurements_.size() == 0) {
    LOG_WARN("No lidar measurements inputted to graph.");
  }
  if (camera_measurements_.size() == 0) {
    LOG_WARN("No camera measurements inputted to graph.");
  }
  if (calibration_initials_.size() == 0) {
    throw std::runtime_error{
        "No initial estimates given to graph. Cannot solve."};
  }
  for (uint8_t i = 0; i < target_params_.size(); i++) {
    if (target_params_[i]->template_cloud->size() == 0 ||
        target_params_[i]->template_cloud == nullptr) {
      LOG_ERROR("Target No. %d contains an empty template cloud.", i);
      throw std::runtime_error{
          "Missing target required to solve graph, or target is empty."};
    }
  }

  if (camera_measurements_.size() > 0 && camera_params_.size() == 0) {
    throw std::runtime_error{"No camera params inputted."};
  }
}

void GTSAMGraph::Clear() {
  graph_.erase(graph_.begin(), graph_.end());
  initials_.clear();
  calibration_results_.clear();
  camera_correspondences_.clear();
  lidar_correspondences_.clear();
}

void GTSAMGraph::AddInitials() {
  // add base link frame as initial pose with 100% certainty
  gtsam::Vector6 v6;
  v6 << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
  Eigen::Matrix4d identity_pose;
  identity_pose.setIdentity();
  gtsam::noiseModel::Diagonal::shared_ptr prior_model =
      gtsam::noiseModel::Diagonal::Variances(v6);
  gtsam::Pose3 pose(identity_pose);
  initials_.insert(gtsam::Symbol('B', 0), pose);
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('B', 0), pose,
                                              prior_model));

  // add all sensors as the next poses
  for (uint32_t i = 0; i < calibration_initials_.size(); i++) {
    vicon_calibration::CalibrationResult calib = calibration_initials_[i];
    Eigen::Matrix4d initial_pose_matrix = calib.transform;
    gtsam::Pose3 initial_pose(initial_pose_matrix);
    if (calib.type == SensorType::LIDAR) {
      initials_.insert(gtsam::Symbol('L', calib.sensor_id), initial_pose);
    } else if (calib.type == SensorType::CAMERA) {
      initials_.insert(gtsam::Symbol('C', calib.sensor_id), initial_pose);
    } else {
      throw std::invalid_argument{
          "Wrong type of sensor inputted as initial calibration estimate."};
    }
  }
}

void GTSAMGraph::SetImageCorrespondences() {
  camera_correspondences_.clear();
  for (uint32_t meas_iter = 0; meas_iter < camera_measurements_.size();
       meas_iter++) {
    vicon_calibration::CameraMeasurement measurement =
        camera_measurements_[meas_iter];
    gtsam::Pose3 pose;
    pose = initials_updated_.at<gtsam::Pose3>(
        gtsam::Symbol('C', measurement.camera_id));
    Eigen::Matrix4d T_CAM_VICONBASE;
    T_CAM_VICONBASE = pose.matrix();

    // create point cloud of projected points
    // TODO: Do we want to extract perimeter points as well here?
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_points;
    target_points = target_params_[measurement.target_id]->template_cloud;
    pcl::PointCloud<pcl::PointXY>::Ptr projected_pixels =
        boost::make_shared<pcl::PointCloud<pcl::PointXY>>();
    for (uint32_t i = 0; i < target_points->size(); i++) {
      pcl::PointXY point_projected_pcl;
      Eigen::Vector2d point_projected;
      Eigen::Vector4d point_transformed;
      pcl::PointXYZ point_pcl = target_points->at(i);
      Eigen::Vector4d point_eig(point_pcl.x, point_pcl.y, point_pcl.z, 1);
      point_transformed =
          T_CAM_VICONBASE * measurement.T_VICONBASE_TARGET * point_eig;
      point_projected = camera_models_[measurement.camera_id]->ProjectPoint(
          point_transformed);
      point_projected_pcl = utils::EigenPixelToPCL(point_projected);
      projected_pixels->push_back(point_projected_pcl);
    }

    // get correspondences
    pcl::registration::CorrespondenceEstimation<pcl::PointXY, pcl::PointXY>
        corr_est;
    double max_distance = 500; // in pixels
    pcl::Correspondences correspondences;
    corr_est.setInputSource(measurement.keypoints);
    corr_est.setInputTarget(projected_pixels);
    corr_est.determineCorrespondences(correspondences, max_distance);
    for (uint32_t i = 0; i < correspondences.size(); i++) {
      vicon_calibration::Correspondence correspondence;
      correspondence.target_point_index = correspondences[i].index_match;
      correspondence.measured_point_index = correspondences[i].index_query;
      correspondence.measurement_index = meas_iter;
      camera_correspondences_.push_back(correspondence);
    }
  }
}

void GTSAMGraph::SetLidarCorrespondences() {
  lidar_correspondences_.clear();
  for (uint32_t meas_iter = 0; meas_iter < lidar_measurements_.size();
       meas_iter++) {
    vicon_calibration::LidarMeasurement measurement =
        lidar_measurements_[meas_iter];
    gtsam::Pose3 pose;
    pose = initials_updated_.at<gtsam::Pose3>(
        gtsam::Symbol('L', measurement.lidar_id));
    Eigen::Matrix4d T_LIDAR_VICONBASE, T_LIDAR_TARGET;
    T_LIDAR_VICONBASE = pose.matrix();
    T_LIDAR_TARGET = T_LIDAR_VICONBASE * measurement.T_VICONBASE_TARGET;

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_template =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::transformPointCloud(
        *(target_params_[measurement.target_id]->template_cloud),
        *transformed_template, T_LIDAR_TARGET);

    // get correspondences
    pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
        corr_est;
    double max_distance = 0.005; // in meters
    pcl::Correspondences correspondences;
    corr_est.setInputSource(measurement.keypoints);
    corr_est.setInputTarget(transformed_template);
    corr_est.determineCorrespondences(correspondences, max_distance);
    for (uint32_t i = 0; i < correspondences.size(); i++) {
      vicon_calibration::Correspondence correspondence;
      correspondence.target_point_index = correspondences[i].index_match;
      correspondence.measured_point_index = correspondences[i].index_query;
      correspondence.measurement_index = meas_iter;
      camera_correspondences_.push_back(correspondence);
    }
  }
}

void GTSAMGraph::SetImageFactors() {
  pcl::PointXY pixel_pcl;
  pcl::PointXYZ point_pcl;
  Eigen::Vector3d point_eig(0, 0, 0);
  Eigen::Vector2d pixel_eig(0, 0);
  int target_index, camera_index;
  // TODO: Figure out a smart way to do this. Do we want to tune the COV based
  // on the number of points per measurement?
  gtsam::Vector2 noise_vec;
  noise_vec << 5, 5;
  gtsam::noiseModel::Diagonal::shared_ptr ImageNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  for (vicon_calibration::Correspondence corr : camera_correspondences_) {
    target_index = camera_measurements_[corr.measurement_index].target_id;
    camera_index = camera_measurements_[corr.measurement_index].camera_id;
    point_pcl = target_params_[target_index]->template_cloud->at(
        corr.target_point_index);
    point_eig = utils::PCLPointToEigen(point_pcl);
    pixel_pcl = camera_measurements_[corr.measurement_index].keypoints->at(
        corr.measured_point_index);
    pixel_eig = utils::PCLPixelToEigen(pixel_pcl);

    gtsam::Key key = gtsam::Symbol('C', camera_index);
    Eigen::Matrix4d T_VICONBASE_TARGET =
        camera_measurements_[corr.measurement_index].T_VICONBASE_TARGET;
    graph_.emplace_shared<CameraFactor>(key, pixel_eig, point_eig,
                                        camera_models_[camera_index],
                                        T_VICONBASE_TARGET, ImageNoise);
  }
}

void GTSAMGraph::SetLidarFactors() {
  pcl::PointXYZ point_measured_pcl, point_predicted_pcl;
  Eigen::Vector4d point_eig(0, 0, 0, 1), point_eig_transformed(0, 0, 0, 1);
  Eigen::Vector3d point_predicted, point_measured;
  int target_index, lidar_index;
  // TODO: Figure out a smart way to do this. Do we want to tune the COV based
  // on the number of points per measurement? ALso, shouldn't this be 2x2?
  gtsam::Vector3 noise_vec;
  noise_vec << 0.02, 0.02, 0.02;
  gtsam::noiseModel::Diagonal::shared_ptr LidarNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  for (vicon_calibration::Correspondence corr : lidar_correspondences_) {
    LidarMeasurement measurement = lidar_measurements_[corr.measurement_index];
    target_index = measurement.target_id;
    lidar_index = measurement.lidar_id;
    point_predicted_pcl = target_params_[target_index]->template_cloud->at(
        corr.target_point_index);
    gtsam::Key key = gtsam::Symbol('L', lidar_index);
    point_measured_pcl = measurement.keypoints->at(corr.measured_point_index);
    point_measured = utils::PCLPointToEigen(point_measured_pcl);
    point_eig = utils::PointToHomoPoint(utils::PCLPointToEigen(point_predicted_pcl));
    point_eig_transformed = measurement.T_VICONBASE_TARGET * point_eig;
    point_predicted = utils::HomoPointToPoint(point_eig_transformed);
    graph_.emplace_shared<LidarFactor>(key, point_measured, point_predicted,
                                       measurement.T_VICONBASE_TARGET,
                                       LidarNoise);
  }
}

void GTSAMGraph::SetLidarCameraFactors() {
  gtsam::Vector2 noise_vec;
  noise_vec << 10, 10;
  gtsam::noiseModel::Diagonal::shared_ptr noiseModel =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  gtsam::Key lidar_key, camera_key;
  Eigen::Vector2d pixel_detected;
  Eigen::Vector3d point_detected, P_T_li, P_T_ci;
  for (uint32_t i = 0; i < loop_closure_measurements_.size(); i++) {
    vicon_calibration::LoopClosureMeasurement measurement =
        loop_closure_measurements_[i];
    uint16_t num_factors =
        std::max<uint16_t>(measurement.keypoints_camera->size(),
                           measurement.keypoints_lidar->size());
    for (uint16_t j = 0; j < num_factors; j++) {
      lidar_key = gtsam::Symbol('L', measurement.lidar_id);
      camera_key = gtsam::Symbol('C', measurement.camera_id);
      pixel_detected =
          utils::PCLPixelToEigen(measurement.keypoints_camera->at(j));
      point_detected =
          utils::PCLPointToEigen(measurement.keypoints_lidar->at(j));

      // TODO: Need to calculate the correspondences (pixel_detected -> P_T_ci,
      // point_detected -> P_T_li). Use same approach as other correspondences?
      graph_.emplace_shared<CameraLidarFactor>(
          lidar_key, camera_key, pixel_detected, point_detected, P_T_ci, P_T_li,
          camera_models_[measurement.camera_id], noiseModel);
    }
  }
}

void GTSAMGraph::Optimize() {
  gtsam::LevenbergMarquardtParams params;
  params.setVerbosity("TERMINATION");
  params.absoluteErrorTol = 1e-8;
  params.relativeErrorTol = 1e-8;
  params.setlambdaUpperBound(1e8);
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

} // end namespace vicon_calibration
