#include "vicon_calibration/GTSAMGraph.h"
#include "vicon_calibration/utils.h"
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <fstream>

namespace vicon_calibration {

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

void GTSAMGraph::SolveGraph() {
  std::cout << "TEST4.1\n";
  Clear();
  std::cout << "TEST4.2\n";
  AddInitials();
  std::cout << "TEST4.3\n";
  AddLidarMeasurements();
  std::cout << "TEST4.4\n";
  // AddImageMeasurements();
  Optimize();
  std::cout << "TEST4.5\n";
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

void GTSAMGraph::AddInitials() {
  // add base link frame as initial pose with 100 certainty
  gtsam::Vector6 v6;
  v6 << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
  Eigen::Matrix4d identity_pose;
  identity_pose.setIdentity();
  gtsam::noiseModel::Diagonal::shared_ptr prior_model =
      gtsam::noiseModel::Diagonal::Variances(v6);
  gtsam::Key first_key = 0;
  gtsam::Pose3 pose(identity_pose);
  initials_.insert(first_key, pose);
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
      first_key, pose, prior_model));

  // add all sensors as the next poses
  for (uint32_t i = 0; i < calibration_initials_.size(); i++) {
    gtsam::Key key = i + 1;
    vicon_calibration::CalibrationResult calib = calibration_initials_[i];
    Eigen::Matrix4d initial_pose_matrix = calib.transform;
    gtsam::Pose3 initial_pose(initial_pose_matrix);
    initials_.insert(key, initial_pose);
  }
}

void GTSAMGraph::AddLidarMeasurements() {
  // TODO: Make this a parameter in a config file for the graph
  // set noise model
  gtsam::Vector6 v6;
  v6 << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
  gtsam::noiseModel::Diagonal::shared_ptr noise_model =
      gtsam::noiseModel::Diagonal::Variances(v6);

  // loop through all lidar measurements and add factors
  for (uint64_t meas_iter = 0; meas_iter < lidar_measurements_.size();
       meas_iter++) {
    gtsam::Key from_key = 0;

    // get sensor key associated with this measurement
    gtsam::Key to_key;
    for (uint32_t sensor_iter = 0; sensor_iter < calibration_initials_.size();
         sensor_iter++) {
      if (lidar_measurements_[meas_iter].lidar_frame ==
          calibration_initials_[sensor_iter].to_frame) {
        gtsam::Key this_key = sensor_iter + 1;
        to_key = this_key;
      } else if (sensor_iter == calibration_initials_.size() - 1) {
        LOG_ERROR("Cannot locate GTSAM key associated with lidar measurement.");
      }
    }
    Eigen::Matrix4d T_VICONBASE_LIDAR, T_VICONBASE_TARGET, T_LIDAR_TARGET;
    vicon_calibration::LidarMeasurement measurement = lidar_measurements_[meas_iter];
    T_VICONBASE_TARGET = measurement.T_VICONBASE_TARGET;
    T_LIDAR_TARGET = measurement.T_LIDAR_TARGET;
    T_VICONBASE_LIDAR = T_VICONBASE_TARGET * T_LIDAR_TARGET.inverse();
    std::cout << "T_VICONBASE_LIDAR: \n" << T_VICONBASE_LIDAR << "\n";
    std::cout << "T_LIDAR_VICONBASE: \n" << T_VICONBASE_LIDAR.inverse() << "\n";
    gtsam::Pose3 pose(T_VICONBASE_LIDAR);
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(from_key, to_key, pose,
                                                  noise_model));
  }
}

void GTSAMGraph::AddImageMeasurements() {
  //
}

void GTSAMGraph::Clear() {
  graph_.erase(graph_.begin(), graph_.end());
  initials_.clear();
  calibration_results_.clear();
}

void GTSAMGraph::Optimize() {
  gtsam::LevenbergMarquardtParams params;
  params.setVerbosity("TERMINATION");
  params.absoluteErrorTol = 1e-8;
  params.relativeErrorTol = 1e-8;
  params.setlambdaUpperBound(1e8);
  gtsam::KeyFormatter key_formatter = gtsam::DefaultKeyFormatter;
  gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initials_, params);
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
