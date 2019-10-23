#include "vicon_calibration/GTSAMGraph.h"
#include "vicon_calibration/utils.h"
#include <fstream>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <pcl/registration/correspondence_estimation.h>

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

void GTSAMGraph::SetCameraParams(
    std::vector<vicon_calibration::CameraParams> &camera_params) {
  camera_params_ = camera_params;
  for (uint16_t i = 0; i < camera_params_.size(); i++) {
    std::shared_ptr<beam_calibration::CameraModel> cam_pointer;
    cam_pointer =
        beam_calibration::CameraModel::LoadJSON(camera_params_[i].intrinsics);
    camera_models_.push_back(cam_pointer);
  }
}

void GTSAMGraph::SolveGraph() {
  CheckInputs();
  Clear();
  AddInitials();
  initials_updated_ = initials_;
  // AddLidarMeasurements();
  while (!CheckConvergence()) {
    SetImageCorrespondances();
    SetImageFactors();
    Optimize();
    initials_updated_ = results_;
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
  if (target_points_.size() == 0 && camera_measurements_.size() > 0) {
    throw std::runtime_error{"Missing target required to solve graph."};
  }
  if (camera_measurements_.size() > 0 && camera_params_.size() == 0) {
    throw std::runtime_error{"No camera params inputted."};
  }
}

void GTSAMGraph::Clear() {
  graph_.erase(graph_.begin(), graph_.end());
  initials_.clear();
  calibration_results_.clear();
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
  initials_.insert(first_key, pose);
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('B', 0), pose,
                                              prior_model));

  // add all sensors as the next poses
  int lidar_count = 0;
  int camera_count = 0;
  for (uint32_t i = 0; i < lidar_initials_.size(); i++) {
    vicon_calibration::CalibrationResult calib = calibration_initials_[i];
    Eigen::Matrix4d initial_pose_matrix = calib.transform;
    gtsam::Pose3 initial_pose(initial_pose_matrix);
    if (calibration_initials_.type == "LIDAR") {
      lidar_count++;
      initials_.insert(gtsam::Symbol('L', lidar_count), initial_pose);
    } else if (calibration_initials_.type == "CAMERA") {
      camera_count++;
      initials_.insert(gtsam::Symbol('C', camera_count), initial_pose);
    } else {
      throw std::invalid_argument{
          "Wrong type of sensor inputted as initial calibration estimate."};
    }
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
    Eigen::Affine3d T_VICONBASE_LIDAR, T_VICONBASE_TARGET, T_LIDAR_TARGET,
        T_TARGET_LIDAR, T_LIDAR_VICONBASE;
    vicon_calibration::LidarMeasurement measurement =
        lidar_measurements_[meas_iter];
    T_VICONBASE_TARGET.matrix() = measurement.T_VICONBASE_TARGET;
    T_LIDAR_TARGET.matrix() = measurement.T_LIDAR_TARGET;
    T_VICONBASE_LIDAR = T_VICONBASE_TARGET * T_LIDAR_TARGET.inverse();
    gtsam::Pose3 pose(T_VICONBASE_LIDAR.matrix());
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(from_key, to_key, pose,
                                                  noise_model));
  }
}

void GTSAMGraph::SetImageCorrespondances() {
  camera_corresspondances_.clear();
  for (uint32_t meas_iter = 0; meas_iter < camera_measurements_.size();
       meas_iter++) {
    vicon_calibration::CameraMeasurement measurement =
        camera_measurements_[meas_iter];

    // create point cloud of projected points
    pcl::PointCloud<pcl::PointXY>::Ptr projected_pixels =
        boost : make_shared<pcl::PointCloud<pcl::PointXY>>();
    for (uint32_t i = 0; i < target_points_.size(); i++) {
      pcl::PointXY point_projected_pcl;
      Eigen::Vector2d point_projected;
      Eigen::Vector4d point_transformed;
      gtsam::Pose3 pose;
      initials_updated_.get(gtsam::Symbol('C', measurement.camera_id), pose);
      Eigen::Matrix4d T_CAM_VICONBASE = pose; // TODO: not sure how to convert this
      point_transformed =
          T_CAM_VICONBASE * measurement.T_VICONBASE_TARGET * target_points_[i];
      point_projected =
          camera_models_[measurement.camera_id].Project(point_transformed);
      point_projected_pcl.x = point_projected[0];
      point_projected_pcl.y = point_projected[1];
      projected_pixels->push_back(point_projected_pcl);
    }

    // create point cloud of measured pixels
    pcl::PointCloud<pcl::PointXY>::Ptr measured_pixels =
        boost : make_shared<pcl::PointCloud<pcl::PointXY>>();
    for (uint32_t i = 0; i < measurement.measured_points.size(); i++) {
      pcl::PointXY point_measured_pcl;
      point_measured_pcl.x = measurement.measured_points[i].x;
      point_measured_pcl.y = measurement.measured_points[i].y;
      measured_pixels->push_back(point_measured_pcl);
    }

    // get correspondances
    pcl::CorrespondenceEstimation<pcl::PointXY, pcl::PointXY> corr_est;
    double max_distance = 500; // in pixels
    pcl::Correspondences correspondances;
    corr_est.setInputSource(measured_pixels);
    corr_est.setInputTarget(projected_pixels);
    corr_est.determineCorrespondences(correspondances, max_distance);
    for (uint32_t i = 0; i < correspondances.size(); i++) {
      pcl::PointXY pixel_pcl = measured_pixels[correspondances[i].index_query];
      Eigen::Vector4d pixel_eig =
          target_points_[correspondances[i].index_match];
      gtsam::Point2 pixel;
      gtsam::Point3 point;
      pixel.x = pixel_pcl.x;
      pixel.y = pixel_pcl.y;
      point.x = pixel_eig[0];
      point.y = pixel_eig[1];
      point.z = pixel_eig[2];
      vicon_calibration::CameraCorresspondance camera_corresspondance;
      camera_corresspondance.pixel = pixel;
      camera_corresspondance.point = point;
      camera_corresspondance.camera_id = measurement.camera_id;
      camera_corresspondance.target_id = measurement.target_id;
      camera_corresspondances_.push_back(camera_corresspondance);
    }
  }
}

void GTSAMGraph::LoadTargetPoints(std::string &template_cloud_path) {
  PointCloud::Ptr template_cloud = boost::make_shared<PointCloud>();
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud_path,
                                          *template_cloud) == -1) {
    LOG_ERROR("Couldn't read template file: %s\n", template_cloud_path.c_str());
  }
  target_points_.clear();
  Eigen::Vector4d point_target;
  point_target[3] = 1;
  for (PointCloud::iterator it = template_cloud->begin();
       it != template_cloud->end(); ++it) {
    point_target[0] = it->x;
    point_target[1] = it->y;
    point_target[2] = it->z;
    target_points_.push_back(point_target);
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
