#include "vicon_calibration/gtsam/Graph.h"
#include "vicon_calibration/gtsam/CameraFactor.h"
#include "vicon_calibration/gtsam/CameraLidarFactor.h"
#include "vicon_calibration/gtsam/LidarFactor.h"
#include "vicon_calibration/utils.h"
#include <algorithm>
#include <fstream>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace vicon_calibration {

void Graph::SetTargetParams(
    std::vector<std::shared_ptr<vicon_calibration::TargetParams>>
        &target_params) {
  target_params_ = target_params;

  // Downsample template cloud
  pcl::VoxelGrid<pcl::PointXYZ> vox;
  vox.setLeafSize(template_downsample_size_[0], template_downsample_size_[1],
                  template_downsample_size_[2]);
  for (int i = 0; i < target_params_.size(); i++) {
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> downsampled_cloud =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    vox.setInputCloud(target_params_[i]->template_cloud);
    vox.filter(*downsampled_cloud);
    target_params_[i]->template_cloud = downsampled_cloud;
  }
}

void Graph::SetLidarMeasurements(
    std::vector<vicon_calibration::LidarMeasurement> &lidar_measurements) {
  lidar_measurements_ = lidar_measurements;
  LOG_INFO("Added %d lidar measurements", lidar_measurements.size());
}

void Graph::SetCameraMeasurements(
    std::vector<vicon_calibration::CameraMeasurement> &camera_measurements) {
  camera_measurements_ = camera_measurements;
  LOG_INFO("Added %d camera measurements", camera_measurements.size());
}

void Graph::SetInitialGuess(
    std::vector<vicon_calibration::CalibrationResult> &initial_guess) {
  calibration_initials_ = initial_guess;
}

void Graph::SetCameraParams(
    std::vector<std::shared_ptr<vicon_calibration::CameraParams>>
        &camera_params) {
  camera_params_ = camera_params;
  for (uint16_t i = 0; i < camera_params_.size(); i++) {
    std::shared_ptr<beam_calibration::CameraModel> cam_pointer;
    cam_pointer =
        beam_calibration::CameraModel::LoadJSON(camera_params_[i]->intrinsics);
    camera_models_.push_back(cam_pointer);
  }
}

void Graph::SolveGraph() {
  this->LoadConfig();
  initials_.clear();
  initials_updated_.clear();
  calibration_results_.clear();
  CheckInputs();
  AddInitials();
  initials_updated_ = initials_;
  uint16_t iteration = 0;
  bool converged = false;
  while (!converged) {
    iteration++;
    LOG_INFO("Iteration: %d", iteration);
    Clear();
    SetImageCorrespondences();
    SetLidarCorrespondences();
    SetImageFactors();
    SetLidarFactors();
    // SetLidarCameraFactors();
    Optimize();
    converged = HasConverged(iteration);
    LOG_INFO("Updating initials");
    initials_updated_ = results_;
  }
  if (iteration >= max_iterations_) {
    LOG_WARN("Reached max iterations, stopping.");
  } else {
    LOG_INFO("Converged after %d iterations.", iteration);
  }
}

void Graph::LoadConfig() {
  std::string config_path = utils::GetFilePathConfig("GraphCong.json");
  nlohmann::json J;
  std::ifstream file(config_path);
  file >> J;
  max_iterations_ = J.at("max_iterations");
  show_camera_measurements_ = J.at("show_camera_measurements");
  show_lidar_measurements_ = J.at("show_lidar_measurements");
  extract_image_target_perimeter_ = J.at("extract_image_target_perimeter");
  concave_hull_alpha_ = J.at("concave_hull_alpha");
  max_pixel_cor_dist_ = J.at("max_pixel_cor_dist");
  abs_error_tol_ = J.at("abs_error_tol");
  rel_error_tol_ = J.at("rel_error_tol");
  lambda_upper_bound_ = J.at("lambda_upper_bound");
  std::vector<double> tmp;
  for (const auto &val : J["error_tol"]) {
    tmp.push_back(val);
  }
  if (tmp.size() != 6) {
    throw std::invalid_argument{
        "Invalid number of inputs to error_tol. Expecting 6."};
  }
  error_tol_ = tmp;
  tmp.clear();
  for (const auto &val : J["image_noise"]) {
    tmp.push_back(val);
  }
  if (tmp.size() != 2) {
    throw std::invalid_argument{
        "Invalid number of inputs to image_noise. Expecting 2."};
  }
  image_noise_ = tmp;
  tmp.clear();
  for (const auto &val : J["lidar_noise"]) {
    tmp.push_back(val);
  }
  if (tmp.size() != 3) {
    throw std::invalid_argument{
        "Invalid number of inputs to lidar_noise. Expecting 3."};
  }
  lidar_noise_ = tmp;
  tmp.clear();
  for (const auto &val : J["template_downsample_size"]) {
    tmp.push_back(val);
  }
  if (tmp.size() != 3) {
    throw std::invalid_argument{
        "Invalid number of inputs to template_downsample_size. Expecting 3."};
  }
  template_downsample_size_ = tmp;
}

std::vector<vicon_calibration::CalibrationResult> Graph::GetResults() {
  for (uint32_t i = 0; i < calibration_initials_.size(); i++) {
    vicon_calibration::CalibrationResult calib = calibration_initials_[i];
    gtsam::Key sensor_key;
    if (calib.type == SensorType::LIDAR) {
      sensor_key = gtsam::symbol('L', calib.sensor_id);
    } else if (calib.type == SensorType::CAMERA) {
      sensor_key = gtsam::symbol('C', calib.sensor_id);
    } else {
      throw std::runtime_error{"Invalid sensor type in calibration initials"};
    }
    calib.transform = results_.at<gtsam::Pose3>(sensor_key).matrix();
    calibration_results_.push_back(calib);
  }
  return calibration_results_;
}

void Graph::Print(std::string &file_name, bool print_to_terminal) {
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

void Graph::CheckInputs() {
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

void Graph::Clear() {
  graph_.erase(graph_.begin(), graph_.end());
  results_.clear();
  camera_correspondences_.clear();
  lidar_correspondences_.clear();
}

void Graph::AddInitials() {
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

void Graph::SetImageCorrespondences() {
  LOG_INFO("Setting image correspondances");
  int counter = 0;
  Eigen::Matrix4d T_VICONBASE_CAM, T_CAM_TARGET;
  for (uint32_t meas_iter = 0; meas_iter < camera_measurements_.size();
       meas_iter++) {
    vicon_calibration::CameraMeasurement measurement =
        camera_measurements_[meas_iter];
    gtsam::Pose3 pose;
    pose = initials_updated_.at<gtsam::Pose3>(
        gtsam::Symbol('C', measurement.camera_id));
    T_VICONBASE_CAM = pose.matrix();
    T_CAM_TARGET = utils::InvertTransform(T_VICONBASE_CAM) *
                   measurement.T_VICONBASE_TARGET;

    // convert measurement to 3D (set z to 0)
    pcl::PointCloud<pcl::PointXYZ>::Ptr measurement_3d =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::PointXYZ point;
    for (pcl::PointCloud<pcl::PointXY>::iterator it =
             measurement.keypoints->begin();
         it != measurement.keypoints->end(); ++it) {
      point.x = it->x;
      point.y = it->y;
      point.z = 0;
      measurement_3d->push_back(point);
    }

    // Check keypoints to see if we want to find correspondances between
    // keypoints or between all target points
    bool use_target_keypoints{false};
    if (target_params_[measurement.target_id]->keypoints_camera.size() > 0) {
      use_target_keypoints = true;
    }

    // get point cloud of projected keypoints
    pcl::PointCloud<pcl::PointXYZ>::Ptr projected_keypoints =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_keypoints =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (use_target_keypoints) {
      // use keypoints specified in json
      Eigen::Vector4d keypoint_homo;
      Eigen::Vector3d keypoint_transformed;
      pcl::PointXYZ keypoint_transformed_pcl;
      for (Eigen::Vector3d keypoint :
           target_params_[measurement.target_id]->keypoints_camera) {
        keypoint_homo = utils::PointToHomoPoint(keypoint);
        keypoint_transformed =
            utils::HomoPointToPoint(T_CAM_TARGET * keypoint_homo);
        keypoint_transformed_pcl = utils::EigenPointToPCL(keypoint_transformed);
        transformed_keypoints->push_back(keypoint_transformed_pcl);
      }
    } else {
      // use all points from template cloud
      pcl::transformPointCloud(
          *(target_params_[measurement.target_id]->template_cloud),
          *transformed_keypoints, T_CAM_TARGET);
    }

    for (uint32_t i = 0; i < transformed_keypoints->size(); i++) {
      pcl::PointXYZ point_pcl = transformed_keypoints->at(i);
      Eigen::Vector2d point_projected;
      if (camera_params_[measurement.camera_id]->images_distorted) {
        point_projected = camera_models_[measurement.camera_id]->ProjectPoint(
            utils::PCLPointToEigen(point_pcl));
      } else {
        point_projected =
            camera_models_[measurement.camera_id]->ProjectUndistortedPoint(
                utils::PCLPointToEigen(point_pcl));
      }
      projected_keypoints->push_back(
          pcl::PointXYZ(point_projected[0], point_projected[1], 0));
    }

    boost::shared_ptr<pcl::Correspondences> correspondences =
        boost::make_shared<pcl::Correspondences>();
    pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
        corr_est;

    if (extract_image_target_perimeter_ && !use_target_keypoints) {
      // keep only perimeter points
      pcl::PointCloud<pcl::PointXYZ>::Ptr hull_cloud =
          boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      pcl::PointIndices::Ptr hull_point_correspondances =
          boost::make_shared<pcl::PointIndices>();
      pcl::ConcaveHull<pcl::PointXYZ> concave_hull;
      concave_hull.setInputCloud(projected_keypoints);
      concave_hull.setAlpha(concave_hull_alpha_);
      concave_hull.setKeepInformation(true);
      concave_hull.reconstruct(*hull_cloud);
      concave_hull.getHullPointIndices(*hull_point_correspondances);

      // get correspondences
      boost::shared_ptr<pcl::Correspondences> correspondences_tmp =
          boost::make_shared<pcl::Correspondences>();
      corr_est.setInputSource(measurement_3d);
      corr_est.setInputTarget(hull_cloud);
      corr_est.determineCorrespondences(*correspondences_tmp,
                                        max_pixel_cor_dist_);
      for (int i = 0; i < correspondences_tmp->size(); i++) {
        int measurement_index = correspondences_tmp->at(i).index_query;
        int hull_index = correspondences_tmp->at(i).index_match;
        int target_index = hull_point_correspondances->indices.at(hull_index);
        correspondences->push_back(
            pcl::Correspondence(measurement_index, target_index, 0));
      }
      if (show_camera_measurements_) {
        this->ViewCameraMeasurements(measurement_3d, hull_cloud,
                                     correspondences_tmp);
      }
    } else {
      // get correspondences
      corr_est.setInputSource(measurement_3d);
      corr_est.setInputTarget(projected_keypoints);
      corr_est.determineCorrespondences(*correspondences, max_pixel_cor_dist_);
      if (show_camera_measurements_) {
        ViewCameraMeasurements(measurement_3d, projected_keypoints,
                               correspondences);
      }
    }

    for (uint32_t i = 0; i < correspondences->size(); i++) {
      counter++;
      vicon_calibration::Correspondence correspondence;
      correspondence.measured_point_index = correspondences->at(i).index_query;
      correspondence.target_point_index = correspondences->at(i).index_match;
      correspondence.measurement_index = meas_iter;
      camera_correspondences_.push_back(correspondence);
    }
  }
  LOG_INFO("Added %d image correspondances.", counter);
}

void Graph::SetLidarCorrespondences() {
  LOG_INFO("Setting lidar correspondances");
  int counter = 0;
  Eigen::Matrix4d T_VICONBASE_LIDAR, T_LIDAR_TARGET;
  for (uint32_t meas_iter = 0; meas_iter < lidar_measurements_.size();
       meas_iter++) {
    vicon_calibration::LidarMeasurement measurement =
        lidar_measurements_[meas_iter];
    gtsam::Pose3 pose;
    pose = initials_updated_.at<gtsam::Pose3>(
        gtsam::Symbol('L', measurement.lidar_id));
    T_VICONBASE_LIDAR = pose.matrix();
    T_LIDAR_TARGET = utils::InvertTransform(T_VICONBASE_LIDAR) *
                     measurement.T_VICONBASE_TARGET;

    // Check keypoints to see if we want to find correspondances between
    // keypoints or between all target points
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_keypoints =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (target_params_[measurement.target_id]->keypoints_lidar.size() > 0) {
      // use keypoints specified in json
      Eigen::Vector4d keypoint_homo;
      Eigen::Vector3d keypoint_transformed;
      pcl::PointXYZ keypoint_transformed_pcl;
      for (Eigen::Vector3d keypoint :
           target_params_[measurement.target_id]->keypoints_lidar) {
        keypoint_homo = utils::PointToHomoPoint(keypoint);
        keypoint_transformed =
            utils::HomoPointToPoint(T_LIDAR_TARGET * keypoint_homo);
        keypoint_transformed_pcl = utils::EigenPointToPCL(keypoint_transformed);
        transformed_keypoints->push_back(keypoint_transformed_pcl);
      }
    } else {
      // use all points from template cloud
      pcl::transformPointCloud(
          *(target_params_[measurement.target_id]->template_cloud),
          *transformed_keypoints, T_LIDAR_TARGET);
    }

    // get correspondences
    pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
        corr_est;
    boost::shared_ptr<pcl::Correspondences> correspondences =
        boost::make_shared<pcl::Correspondences>();
    corr_est.setInputSource(measurement.keypoints);
    corr_est.setInputTarget(transformed_keypoints);
    corr_est.determineCorrespondences(*correspondences, max_point_cor_dist_);
    if (show_lidar_measurements_) {
      this->ViewLidarMeasurements(measurement.keypoints, transformed_keypoints,
                                  correspondences, "measured keypoints",
                                  "estimated keypoints");
    }
    for (uint32_t i = 0; i < correspondences->size(); i++) {
      counter++;
      vicon_calibration::Correspondence correspondence;
      correspondence.target_point_index = correspondences->at(i).index_match;
      correspondence.measured_point_index = correspondences->at(i).index_query;
      correspondence.measurement_index = meas_iter;
      lidar_correspondences_.push_back(correspondence);
    }
  }
  LOG_INFO("Added %d lidar correspondances.", counter);
}

void Graph::SetImageFactors() {
  LOG_INFO("Setting image factors");
  int counter = 0;
  Eigen::Vector3d point(0, 0, 0);
  Eigen::Vector2d pixel(0, 0);
  int target_index, camera_index;

  // TODO: Figure out a smart way to do this. Do we want to tune the COV based
  // on the number of points per measurement?
  gtsam::Vector2 noise_vec;
  noise_vec << image_noise_[0], image_noise_[1];
  gtsam::noiseModel::Diagonal::shared_ptr ImageNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  for (vicon_calibration::Correspondence corr : camera_correspondences_) {
    counter++;
    CameraMeasurement measurement =
        camera_measurements_[corr.measurement_index];
    target_index = measurement.target_id;
    camera_index = measurement.camera_id;

    if (target_params_[target_index]->keypoints_camera.size() > 0) {
      point = target_params_[target_index]
                  ->keypoints_camera[corr.target_point_index];
    } else {
      point = utils::PCLPointToEigen(
          target_params_[target_index]->template_cloud->at(
              corr.target_point_index));
    }

    pixel = utils::PCLPixelToEigen(
        measurement.keypoints->at(corr.measured_point_index));
    gtsam::Key key = gtsam::Symbol('C', camera_index);
    graph_.emplace_shared<CameraFactor>(
        key, pixel, point, camera_models_[camera_index],
        measurement.T_VICONBASE_TARGET, ImageNoise,
        camera_params_[camera_index]->images_distorted);
  }
  LOG_INFO("Added %d image factors.", counter);
}

void Graph::SetLidarFactors() {
  LOG_INFO("Setting lidar factors");
  Eigen::Vector3d point_predicted, point_measured;
  int target_index, lidar_index;
  // TODO: Figure out a smart way to do this. Do we want to tune the COV based
  // on the number of points per measurement? ALso, shouldn't this be 2x2?
  gtsam::Vector3 noise_vec;
  noise_vec << lidar_noise_[0], lidar_noise_[1], lidar_noise_[2];
  gtsam::noiseModel::Diagonal::shared_ptr LidarNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  int counter = 0;
  for (vicon_calibration::Correspondence corr : lidar_correspondences_) {
    counter++;
    LidarMeasurement measurement = lidar_measurements_[corr.measurement_index];
    target_index = measurement.target_id;
    lidar_index = measurement.lidar_id;

    if (target_params_[target_index]->keypoints_lidar.size() > 0) {
      point_predicted = target_params_[target_index]
                            ->keypoints_lidar[corr.target_point_index];
    } else {
      point_predicted = utils::PCLPointToEigen(
          target_params_[target_index]->template_cloud->at(
              corr.target_point_index));
    }

    point_measured = utils::PCLPointToEigen(
        measurement.keypoints->at(corr.measured_point_index));
    gtsam::Key key = gtsam::Symbol('L', lidar_index);
    graph_.emplace_shared<LidarFactor>(key, point_measured, point_predicted,
                                       measurement.T_VICONBASE_TARGET,
                                       LidarNoise);
  }
  LOG_INFO("Added %d lidar factors.", counter);
}

void Graph::SetLidarCameraFactors() {
  LOG_INFO("Setting lidar-camera factors");
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
          camera_models_[measurement.camera_id], noiseModel,
          camera_params_[measurement.camera_id]->images_distorted);
    }
  }
}

void Graph::Optimize() {
  LOG_INFO("Optimizing graph");
  gtsam::LevenbergMarquardtParams params;
  params.setVerbosity("TERMINATION");
  params.absoluteErrorTol = abs_error_tol_;
  params.relativeErrorTol = rel_error_tol_;
  params.setlambdaUpperBound(lambda_upper_bound_);
  gtsam::KeyFormatter key_formatter = gtsam::DefaultKeyFormatter;
  gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initials_updated_,
                                               params);
  results_.clear();
  std::exception_ptr eptr;

  try {
    results_ = optimizer.optimize();
    // LOG_INFO("Printing Graph:");
    // graph_.print();
    LOG_INFO("Printing Initials:");
    initials_.print();
    LOG_INFO("Printing Results:");
    results_.print();
  } catch (...) {
    LOG_ERROR("Error optimizing GTSAM Graph. Printing graph and initial "
              "estimates to terminal.");
    graph_.print();
    initials_.print();
    eptr = std::current_exception();
    std::rethrow_exception(eptr);
  }
}

bool Graph::HasConverged(uint16_t iteration) {
  if (iteration == 0) {
    return false;
  } else if (iteration == max_iterations_) {
    return true;
  }

  // Loop through results
  Eigen::Matrix4d T_last, T_curr;
  Eigen::Vector3d R_curr, R_last, t_curr, t_last, R_error, t_error;
  double eR, et, eR_abs, et_abs;
  for (uint32_t i = 0; i < calibration_initials_.size(); i++) {
    gtsam::Key sensor_key;
    if (calibration_initials_[i].type == SensorType::LIDAR) {
      sensor_key = gtsam::symbol('L', calibration_initials_[i].sensor_id);
    } else if (calibration_initials_[i].type == SensorType::CAMERA) {
      sensor_key = gtsam::symbol('C', calibration_initials_[i].sensor_id);
    } else {
      throw std::runtime_error{"Invalid sensor type in calibration initials"};
    }
    T_last = initials_updated_.at<gtsam::Pose3>(sensor_key).matrix();
    T_curr = results_.at<gtsam::Pose3>(sensor_key).matrix();

    // Check all DOFs to see if the change is greater than the tolerance
    R_curr = utils::RToLieAlgebra(T_curr.block(0, 0, 3, 3));
    R_last = utils::RToLieAlgebra(T_last.block(0, 0, 3, 3));
    t_curr = T_curr.block(0, 3, 3, 1);
    t_last = T_last.block(0, 3, 3, 1);
    R_error = R_curr - R_last;
    t_error = t_curr - t_last;
    for (int i = 0; i < 3; i++) {
      eR = R_error[i];
      et = t_error[i];
      eR_abs = sqrt(eR * eR * 1000 * 1000) / 1000;
      et_abs = sqrt(et * et * 1000 * 1000) / 1000;
      if (eR_abs > error_tol_[i]) {
        return false;
      }
      if (et_abs > error_tol_[i + 3]) {
        return false;
      }
    }
  }
  return true;
}

void Graph::ViewCameraMeasurements(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &c1,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &c2,
    const boost::shared_ptr<pcl::Correspondences> &correspondences) {
  PointCloudColor::Ptr c1_col = boost::make_shared<PointCloudColor>();
  PointCloudColor::Ptr c2_col = boost::make_shared<PointCloudColor>();
  uint32_t rgb1 = (static_cast<uint32_t>(255) << 16 |
                   static_cast<uint32_t>(0) << 8 | static_cast<uint32_t>(0));
  uint32_t rgb2 = (static_cast<uint32_t>(0) << 16 |
                   static_cast<uint32_t>(255) << 8 | static_cast<uint32_t>(0));
  pcl::PointXYZRGB point;
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it = c1->begin();
       it != c1->end(); ++it) {
    point.x = it->x;
    point.y = it->y;
    point.z = it->z;
    point.rgb = *reinterpret_cast<float *>(&rgb1);
    c1_col->push_back(point);
  }
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it = c2->begin();
       it != c2->end(); ++it) {
    point.x = it->x;
    point.y = it->y;
    point.z = it->z;
    point.rgb = *reinterpret_cast<float *>(&rgb2);
    c2_col->push_back(point);
  }
  pcl::visualization::PCLVisualizer::Ptr pcl_viewer =
      boost::make_shared<pcl::visualization::PCLVisualizer>();
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1_(
      c1_col);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2_(
      c2_col);
  pcl_viewer->addPointCloud<pcl::PointXYZRGB>(c1_col, rgb1_, "Cloud1");
  pcl_viewer->addPointCloud<pcl::PointXYZRGB>(c2_col, rgb2_, "Cloud2");
  pcl_viewer->addCorrespondences<pcl::PointXYZRGB>(c1_col, c2_col,
                                                   *correspondences);
  std::cout << "\nViewer Legend:\n"
            << "  Red   -> detected points\n"
            << "  Green -> projected points\n";
  while (!pcl_viewer->wasStopped()) {
    pcl_viewer->spinOnce(10);
  }
  pcl_viewer->removeAllPointClouds();
  pcl_viewer->close();
  pcl_viewer->resetStoppedFlag();
}

void Graph::ViewLidarMeasurements(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &c1,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &c2,
    const boost::shared_ptr<pcl::Correspondences> &correspondences,
    const std::string &c1_name, const std::string &c2_name) {
  PointCloudColor::Ptr c1_col = boost::make_shared<PointCloudColor>();
  PointCloudColor::Ptr c2_col = boost::make_shared<PointCloudColor>();

  uint32_t rgb1 = (static_cast<uint32_t>(255) << 16 |
                   static_cast<uint32_t>(0) << 8 | static_cast<uint32_t>(0));
  uint32_t rgb2 = (static_cast<uint32_t>(0) << 16 |
                   static_cast<uint32_t>(255) << 8 | static_cast<uint32_t>(0));
  pcl::PointXYZRGB point;
  for (PointCloud::iterator it = c1->begin(); it != c1->end(); ++it) {
    point.x = it->x;
    point.y = it->y;
    point.z = it->z;
    point.rgb = *reinterpret_cast<float *>(&rgb1);
    c1_col->push_back(point);
  }
  for (PointCloud::iterator it = c2->begin(); it != c2->end(); ++it) {
    point.x = it->x;
    point.y = it->y;
    point.z = it->z;
    point.rgb = *reinterpret_cast<float *>(&rgb2);
    c2_col->push_back(point);
  }
  pcl::visualization::PCLVisualizer::Ptr pcl_viewer =
      boost::make_shared<pcl::visualization::PCLVisualizer>();
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1_(
      c1_col);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2_(
      c2_col);
  pcl_viewer->addPointCloud<pcl::PointXYZRGB>(c1_col, rgb1_, c1_name);
  pcl_viewer->addPointCloud<pcl::PointXYZRGB>(c2_col, rgb2_, c2_name);
  pcl_viewer->addCorrespondences<pcl::PointXYZRGB>(c1_col, c2_col,
                                                   *correspondences);
  std::cout << "\nViewer Legend:\n"
            << "  Red   -> " << c1_name << "\n"
            << "  Green -> " << c2_name << "\n";
  while (!pcl_viewer->wasStopped()) {
    pcl_viewer->spinOnce(10);
  }
  pcl_viewer->removeAllPointClouds();
  pcl_viewer->close();
  pcl_viewer->resetStoppedFlag();
}

} // end namespace vicon_calibration
