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

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

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
    std::vector<std::vector<std::shared_ptr<LidarMeasurement>>>
        &lidar_measurements) {
  lidar_measurements_ = lidar_measurements;
  LOG_INFO("Added measurements for %d lidar(s)", lidar_measurements.size());
}

void Graph::SetCameraMeasurements(
    std::vector<std::vector<std::shared_ptr<CameraMeasurement>>>
        &camera_measurements) {
  camera_measurements_ = camera_measurements;
  LOG_INFO("Added measurements for %d camera(s)", camera_measurements.size());
}

void Graph::SetLoopClosureMeasurements(
    std::vector<std::shared_ptr<LoopClosureMeasurement>>
        &loop_closure_measurements) {
  loop_closure_measurements_ = loop_closure_measurements;
  LOG_INFO("Added %d loop closure measurements",
           loop_closure_measurements_.size());
}

void Graph::SetInitialGuess(
    std::vector<vicon_calibration::CalibrationResult> &initial_guess) {
  calibration_initials_ = initial_guess;
}

void Graph::SetCameraParams(
    std::vector<std::shared_ptr<vicon_calibration::CameraParams>>
        &camera_params) {
  camera_params_ = camera_params;
}

void Graph::SolveGraph() {
  this->LoadConfig();
  if (show_camera_measurements_ || show_lidar_measurements_ ||
      show_loop_closure_correspondences_) {
    pcl_viewer_ = boost::make_shared<pcl::visualization::PCLVisualizer>();
  }
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
    SetLoopClosureCorrespondences();
    SetImageFactors();
    SetLidarFactors();
    SetLidarCameraFactors();
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

// TODO: move this to json tools object
void Graph::LoadConfig() {
  std::string config_path = utils::GetFilePathConfig("GraphConfig.json");
  LOG_INFO("Loading GTSAM Graph Config file: %s", config_path.c_str());
  nlohmann::json J;
  std::ifstream file(config_path);
  file >> J;
  max_iterations_ = J.at("max_iterations");
  show_camera_measurements_ = J.at("show_camera_measurements");
  show_lidar_measurements_ = J.at("show_lidar_measurements");
  show_loop_closure_correspondences_ =
      J.at("show_loop_closure_correspondences");
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
  print_results_to_terminal_ = J.at("print_results_to_terminal");
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
  if(skip_to_next_iteration_){
    stop_all_vis_ = false;
    skip_to_next_iteration_ = false;
  }
  graph_.erase(graph_.begin(), graph_.end());
  results_.clear();
  camera_correspondences_.clear();
  lidar_correspondences_.clear();
  lidar_camera_correspondences_.clear();
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
  LOG_INFO("Setting image correspondences");
  int counter = 0;
  Eigen::Matrix4d T_VICONBASE_CAM, T_CAM_TARGET;
  std::shared_ptr<CameraMeasurement> measurement;
  for (uint8_t cam_iter = 0; cam_iter < camera_measurements_.size();
       cam_iter++) {
    for (uint32_t meas_iter = 0;
         meas_iter < camera_measurements_[cam_iter].size(); meas_iter++) {
      if (camera_measurements_[cam_iter][meas_iter] == nullptr) {
        continue;
      }
      measurement = camera_measurements_[cam_iter][meas_iter];
      gtsam::Pose3 pose;
      pose = initials_updated_.at<gtsam::Pose3>(
          gtsam::Symbol('C', measurement->camera_id));
      T_VICONBASE_CAM = pose.matrix();
      T_CAM_TARGET = utils::InvertTransform(T_VICONBASE_CAM) *
                     measurement->T_VICONBASE_TARGET;

      // convert measurement to 3D (set z to 0)
      pcl::PointCloud<pcl::PointXYZ>::Ptr measurement_3d =
          boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      pcl::PointXYZ point;
      for (pcl::PointCloud<pcl::PointXY>::iterator it =
               measurement->keypoints->begin();
           it != measurement->keypoints->end(); ++it) {
        point.x = it->x;
        point.y = it->y;
        point.z = 0;
        measurement_3d->push_back(point);
      }

      // Check keypoints to see if we want to find correspondences between
      // keypoints or between all target points
      bool use_target_keypoints{false};
      if (target_params_[measurement->target_id]->keypoints_camera.size() > 0) {
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
             target_params_[measurement->target_id]->keypoints_camera) {
          keypoint_homo = utils::PointToHomoPoint(keypoint);
          keypoint_transformed =
              utils::HomoPointToPoint(T_CAM_TARGET * keypoint_homo);
          keypoint_transformed_pcl =
              utils::EigenPointToPCL(keypoint_transformed);
          transformed_keypoints->push_back(keypoint_transformed_pcl);
        }
      } else {
        // use all points from template cloud
        pcl::transformPointCloud(
            *(target_params_[measurement->target_id]->template_cloud),
            *transformed_keypoints, T_CAM_TARGET);
      }

      for (uint32_t i = 0; i < transformed_keypoints->size(); i++) {
        pcl::PointXYZ point_pcl = transformed_keypoints->at(i);
        Eigen::Vector2d point_projected;
        if (camera_params_[measurement->camera_id]->images_distorted) {
          point_projected = camera_params_[measurement->camera_id]
                                ->camera_model->ProjectPoint(
                                    utils::PCLPointToEigen(point_pcl));
        } else {
          point_projected = camera_params_[measurement->camera_id]
                                ->camera_model->ProjectUndistortedPoint(
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
        pcl::PointIndices::Ptr hull_point_correspondences =
            boost::make_shared<pcl::PointIndices>();
        pcl::ConcaveHull<pcl::PointXYZ> concave_hull;
        concave_hull.setInputCloud(projected_keypoints);
        concave_hull.setAlpha(concave_hull_alpha_);
        concave_hull.setKeepInformation(true);
        concave_hull.reconstruct(*hull_cloud);
        concave_hull.getHullPointIndices(*hull_point_correspondences);

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
          int target_index = hull_point_correspondences->indices.at(hull_index);
          correspondences->push_back(
              pcl::Correspondence(measurement_index, target_index, 0));
        }
        if (show_camera_measurements_ && !stop_all_vis_) {
          this->ViewCameraMeasurements(
              measurement_3d, hull_cloud, correspondences_tmp,
              "measured camera points", "projected camera points");
        }
      } else {
        // get correspondences
        corr_est.setInputSource(measurement_3d);
        corr_est.setInputTarget(projected_keypoints);
        corr_est.determineCorrespondences(*correspondences,
                                          max_pixel_cor_dist_);
        if (show_camera_measurements_ && !stop_all_vis_) {
          ViewCameraMeasurements(measurement_3d, projected_keypoints,
                                 correspondences, "measured camera points",
                                 "projected camera points");
        }
      }

      for (uint32_t i = 0; i < correspondences->size(); i++) {
        counter++;
        vicon_calibration::Correspondence correspondence;
        correspondence.measured_point_index =
            correspondences->at(i).index_query;
        correspondence.target_point_index = correspondences->at(i).index_match;
        correspondence.measurement_index = meas_iter;
        correspondence.sensor_index = cam_iter;
        camera_correspondences_.push_back(correspondence);
      }
    }
  }
  LOG_INFO("Added %d image correspondences.", counter);
}

void Graph::SetLidarCorrespondences() {
  LOG_INFO("Setting lidar correspondences");
  int counter = 0;
  Eigen::Matrix4d T_VICONBASE_LIDAR, T_LIDAR_TARGET;
  std::shared_ptr<LidarMeasurement> measurement;
  for (uint8_t lidar_iter = 0; lidar_iter < lidar_measurements_.size();
       lidar_iter++) {
    for (uint32_t meas_iter = 0;
         meas_iter < lidar_measurements_[lidar_iter].size(); meas_iter++) {
      if (lidar_measurements_[lidar_iter][meas_iter] == nullptr) {
        continue;
      }
      measurement = lidar_measurements_[lidar_iter][meas_iter];
      gtsam::Pose3 pose;
      pose = initials_updated_.at<gtsam::Pose3>(
          gtsam::Symbol('L', measurement->lidar_id));
      T_VICONBASE_LIDAR = pose.matrix();
      T_LIDAR_TARGET = utils::InvertTransform(T_VICONBASE_LIDAR) *
                       measurement->T_VICONBASE_TARGET;

      // Check keypoints to see if we want to find correspondences between
      // keypoints or between all target points
      pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_keypoints =
          boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      if (target_params_[measurement->target_id]->keypoints_lidar.size() > 0) {
        // use keypoints specified in json
        Eigen::Vector4d keypoint_homo;
        Eigen::Vector3d keypoint_transformed;
        pcl::PointXYZ keypoint_transformed_pcl;
        for (Eigen::Vector3d keypoint :
             target_params_[measurement->target_id]->keypoints_lidar) {
          keypoint_homo = utils::PointToHomoPoint(keypoint);
          keypoint_transformed =
              utils::HomoPointToPoint(T_LIDAR_TARGET * keypoint_homo);
          keypoint_transformed_pcl =
              utils::EigenPointToPCL(keypoint_transformed);
          transformed_keypoints->push_back(keypoint_transformed_pcl);
        }
      } else {
        // use all points from template cloud
        pcl::transformPointCloud(
            *(target_params_[measurement->target_id]->template_cloud),
            *transformed_keypoints, T_LIDAR_TARGET);
      }

      // get correspondences
      pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
          corr_est;
      boost::shared_ptr<pcl::Correspondences> correspondences =
          boost::make_shared<pcl::Correspondences>();
      corr_est.setInputSource(measurement->keypoints);
      corr_est.setInputTarget(transformed_keypoints);
      corr_est.determineCorrespondences(*correspondences, max_point_cor_dist_);
      if (show_lidar_measurements_ && !stop_all_vis_) {
        this->ViewLidarMeasurements(
            measurement->keypoints, transformed_keypoints, correspondences,
            "measured lidar keypoints", "estimated lidar keypoints");
      }
      for (uint32_t i = 0; i < correspondences->size(); i++) {
        counter++;
        vicon_calibration::Correspondence correspondence;
        correspondence.target_point_index = correspondences->at(i).index_match;
        correspondence.measured_point_index =
            correspondences->at(i).index_query;
        correspondence.measurement_index = meas_iter;
        correspondence.sensor_index = lidar_iter;
        lidar_correspondences_.push_back(correspondence);
      }
    }
  }
  LOG_INFO("Added %d lidar correspondences.", counter);
}

void Graph::SetLoopClosureCorrespondences() {
  Eigen::Matrix4d T_SENSOR_TARGET, T_VICONBASE_SENSOR;
  Eigen::Vector3d keypoint_transformed;
  Eigen::Vector2d keypoint_projected;
  Eigen::Vector3d keypoint_projected_3d;
  gtsam::Key key;
  std::shared_ptr<LoopClosureMeasurement> measurement;
  if (show_loop_closure_correspondences_ && !stop_all_vis_) {
    LOG_INFO("Showing lidar-camera loop closure measurement correspondences");
  }

  for (int meas_iter = 0; meas_iter < loop_closure_measurements_.size();
       meas_iter++) {
    measurement = loop_closure_measurements_[meas_iter];

    // Transform lidar target keypoints to lidar frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr estimated_lidar_keypoints =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for (Eigen::Vector3d keypoint :
         target_params_[measurement->target_id]->keypoints_lidar) {
      // get transform from target to lidar
      key = gtsam::symbol('L', measurement->lidar_id);
      T_VICONBASE_SENSOR = initials_updated_.at<gtsam::Pose3>(key).matrix();
      T_SENSOR_TARGET = utils::InvertTransform(T_VICONBASE_SENSOR) *
                        measurement->T_VICONBASE_TARGET;
      keypoint_transformed = utils::HomoPointToPoint(
          T_SENSOR_TARGET * utils::PointToHomoPoint(keypoint));
      estimated_lidar_keypoints->push_back(
          utils::EigenPointToPCL(keypoint_transformed));
    }

    // Get lidar correspondences
    pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
        lidar_corr_est;
    boost::shared_ptr<pcl::Correspondences> lidar_correspondences =
        boost::make_shared<pcl::Correspondences>();
    lidar_corr_est.setInputSource(measurement->keypoints_lidar);
    lidar_corr_est.setInputTarget(estimated_lidar_keypoints);
    lidar_corr_est.determineCorrespondences(*lidar_correspondences,
                                            max_point_cor_dist_);

    // Transform camera target keypoints to camera frame and project to image
    pcl::PointCloud<pcl::PointXYZ>::Ptr estimated_camera_keypoints =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for (Eigen::Vector3d keypoint :
         target_params_[measurement->target_id]->keypoints_camera) {
      // get transform from target to camera
      key = gtsam::symbol('C', measurement->camera_id);
      T_VICONBASE_SENSOR = initials_updated_.at<gtsam::Pose3>(key).matrix();
      T_SENSOR_TARGET = utils::InvertTransform(T_VICONBASE_SENSOR) *
                        measurement->T_VICONBASE_TARGET;
      keypoint_transformed = utils::HomoPointToPoint(
          T_SENSOR_TARGET * utils::PointToHomoPoint(keypoint));
      if (camera_params_[measurement->camera_id]->images_distorted) {
        keypoint_projected =
            camera_params_[measurement->camera_id]->camera_model->ProjectPoint(
                keypoint_transformed);
      } else {
        keypoint_projected =
            camera_params_[measurement->camera_id]
                ->camera_model->ProjectUndistortedPoint(keypoint_transformed);
      }
      keypoint_projected_3d =
          Eigen::Vector3d(keypoint_projected[0], keypoint_projected[1], 0);
      estimated_camera_keypoints->push_back(
          utils::EigenPointToPCL(keypoint_projected_3d));
    }

    // convert measurement to 3D (set z to 0)
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_measurement_3d =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::PointXYZ point;
    for (pcl::PointCloud<pcl::PointXY>::iterator it =
             measurement->keypoints_camera->begin();
         it != measurement->keypoints_camera->end(); ++it) {
      point.x = it->x;
      point.y = it->y;
      point.z = 0;
      camera_measurement_3d->push_back(point);
    }

    // Get camera correspondences
    pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
        camera_corr_est;
    boost::shared_ptr<pcl::Correspondences> camera_correspondences =
        boost::make_shared<pcl::Correspondences>();
    camera_corr_est.setInputSource(camera_measurement_3d);
    camera_corr_est.setInputTarget(estimated_camera_keypoints);
    camera_corr_est.determineCorrespondences(*camera_correspondences,
                                             max_pixel_cor_dist_);

    // create correspondence and add to list
    uint32_t num_corr = std::min<uint16_t>(camera_correspondences->size(),
                                           lidar_correspondences->size());
    LoopCorrespondence corr;
    for (uint32_t i = 0; i < num_corr; i++) {
      corr.camera_target_point_index =
          camera_correspondences->at(i).index_match;
      corr.camera_measurement_point_index =
          camera_correspondences->at(i).index_query;
      corr.lidar_target_point_index = lidar_correspondences->at(i).index_match;
      corr.lidar_measurement_point_index =
          lidar_correspondences->at(i).index_query;
      corr.camera_id = measurement->camera_id;
      corr.lidar_id = measurement->lidar_id;
      corr.measurement_index = meas_iter;
      corr.target_id = measurement->target_id;
      lidar_camera_correspondences_.push_back(corr);
    }

    if (show_loop_closure_correspondences_ && !stop_all_vis_) {
      this->ViewLidarMeasurements(
          measurement->keypoints_lidar, estimated_lidar_keypoints,
          lidar_correspondences, "measured lidar keypoints",
          "estimated lidar keypoints");
    }
    if (show_loop_closure_correspondences_ && !stop_all_vis_){
      this->ViewCameraMeasurements(
          camera_measurement_3d, estimated_camera_keypoints,
          camera_correspondences, "measured camera points",
          "projected camera points");
    }
  }
  LOG_INFO("Added %d lidar-camera correspondences",
           lidar_camera_correspondences_.size());
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
    std::shared_ptr<CameraMeasurement> measurement =
        camera_measurements_[corr.sensor_index][corr.measurement_index];
    target_index = measurement->target_id;
    camera_index = measurement->camera_id;

    if (target_params_[target_index]->keypoints_camera.size() > 0) {
      point = target_params_[target_index]
                  ->keypoints_camera[corr.target_point_index];
    } else {
      point = utils::PCLPointToEigen(
          target_params_[target_index]->template_cloud->at(
              corr.target_point_index));
    }

    pixel = utils::PCLPixelToEigen(
        measurement->keypoints->at(corr.measured_point_index));
    gtsam::Key key = gtsam::Symbol('C', camera_index);
    graph_.emplace_shared<CameraFactor>(
        key, pixel, point, camera_params_[camera_index]->camera_model,
        measurement->T_VICONBASE_TARGET, ImageNoise,
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
    std::shared_ptr<LidarMeasurement> measurement =
        lidar_measurements_[corr.sensor_index][corr.measurement_index];
    target_index = measurement->target_id;
    lidar_index = measurement->lidar_id;

    if (target_params_[target_index]->keypoints_lidar.size() > 0) {
      point_predicted = target_params_[target_index]
                            ->keypoints_lidar[corr.target_point_index];
    } else {
      point_predicted = utils::PCLPointToEigen(
          target_params_[target_index]->template_cloud->at(
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

void Graph::SetLidarCameraFactors() {
  LOG_INFO("Setting lidar-camera factors");
  gtsam::Vector2 noise_vec;
  noise_vec << 10, 10;
  gtsam::noiseModel::Diagonal::shared_ptr noiseModel =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  gtsam::Key lidar_key, camera_key;
  Eigen::Vector2d pixel_detected;
  Eigen::Vector3d point_detected, P_T_li, P_T_ci;
  int counter = 0;
  for (LoopCorrespondence corr : lidar_camera_correspondences_) {
    counter++;
    lidar_key = gtsam::Symbol('L', corr.lidar_id);
    camera_key = gtsam::Symbol('C', corr.camera_id);

    // get measured point/pixel expressed in sensor frame
    pixel_detected = utils::PCLPixelToEigen(
        loop_closure_measurements_[corr.measurement_index]
            ->keypoints_camera->at(corr.camera_measurement_point_index));
    point_detected = utils::PCLPointToEigen(
        loop_closure_measurements_[corr.measurement_index]->keypoints_lidar->at(
            corr.lidar_measurement_point_index));

    // get corresponding target points expressed in target frames
    P_T_ci = target_params_[corr.target_id]
                 ->keypoints_camera[corr.camera_target_point_index];
    P_T_li = target_params_[corr.target_id]
                 ->keypoints_lidar[corr.lidar_target_point_index];

    graph_.emplace_shared<CameraLidarFactor>(
        lidar_key, camera_key, pixel_detected, point_detected, P_T_ci, P_T_li,
        camera_params_[corr.camera_id]->camera_model, noiseModel,
        camera_params_[corr.camera_id]->images_distorted);
  }
  LOG_INFO("Added %d lidar-camera factors.", counter);
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
    if (print_results_to_terminal_) {
      LOG_INFO("Printing Initials:");
      initials_.print();
      LOG_INFO("Printing Results:");
      results_.print();
    }
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
    const boost::shared_ptr<pcl::Correspondences> &correspondences,
    const std::string &c1_name, const std::string &c2_name) {
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
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1_(
      c1_col);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2_(
      c2_col);
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(c1_col, rgb1_, "Cloud1");
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(c2_col, rgb2_, "Cloud2");
  pcl_viewer_->addCorrespondences<pcl::PointXYZRGB>(c1_col, c2_col,
                                                    *correspondences);
  pcl_viewer_->resetCameraViewpoint();
  std::cout << "\nViewer Legend:\n"
            << "  Red   -> " << c1_name << "\n"
            << "  Green -> " << c2_name << "\n"
            << "Press [c] to continue\n"
            << "Press [n] to skip to next iteration\n"
            << "Press [s] to stop showing these measurements.\n";
  while (!pcl_viewer_->wasStopped() && !close_viewer_) {
    pcl_viewer_->spinOnce(10);
    pcl_viewer_->registerKeyboardCallback(
        &Graph::ConfirmMeasurementKeyboardCallback, *this);
    std::this_thread::sleep_for(10ms);
  }
  close_viewer_ = false;
  pcl_viewer_->removeAllPointClouds();
  pcl_viewer_->removeAllShapes();
  pcl_viewer_->close();
  pcl_viewer_->resetStoppedFlag();
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
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1_(
      c1_col);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2_(
      c2_col);
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(c1_col, rgb1_, c1_name);
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(c2_col, rgb2_, c2_name);
  pcl_viewer_->addCorrespondences<pcl::PointXYZRGB>(c1_col, c2_col,
                                                    *correspondences);
  pcl_viewer_->resetCameraViewpoint();
  std::cout << "\nViewer Legend:\n"
            << "  Red   -> " << c1_name << "\n"
            << "  Green -> " << c2_name << "\n"
            << "Press [c] to continue\n"
            << "Press [n] to skip to next iteration\n"
            << "Press [s] to stop showing these measurements\n";
  while (!pcl_viewer_->wasStopped() && !close_viewer_) {
    pcl_viewer_->spinOnce(10);
    pcl_viewer_->registerKeyboardCallback(
        &Graph::ConfirmMeasurementKeyboardCallback, *this);
    std::this_thread::sleep_for(10ms);
  }
  close_viewer_ = false;
  pcl_viewer_->removeAllPointClouds();
  pcl_viewer_->removeAllShapes();
  pcl_viewer_->close();
  pcl_viewer_->resetStoppedFlag();
}

void Graph::ConfirmMeasurementKeyboardCallback(
    const pcl::visualization::KeyboardEvent &event, void *viewer_void) {
  if (event.getKeySym() == "s" && event.keyDown()) {
    stop_all_vis_ = true;
    close_viewer_ = true;
  } else if (event.getKeySym() == "c" && event.keyDown()) {
    close_viewer_ = true;
  } else if (event.getKeySym() == "n" && event.keyDown()) {
    skip_to_next_iteration_ = true;
    stop_all_vis_ = true;
    close_viewer_ = true;
  }
}

} // end namespace vicon_calibration
