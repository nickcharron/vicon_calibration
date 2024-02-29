#include <vicon_calibration/optimization/Optimizer.h>

#include <algorithm>
#include <fstream>
#include <thread>

#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/concave_hull.h>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

Optimizer::Optimizer(const OptimizerInputs& inputs) : inputs_(inputs) {
  LOG_INFO("Loading Optimizer Config file: %s",
           inputs_.optimizer_config_path.c_str());

  nlohmann::json J;
  if (!utils::ReadJson(inputs_.optimizer_config_path, J)) {
    LOG_ERROR("Using default ceres optimizer params.");
    return;
  }

  LoadConfigCommon(J);

  LOG_INFO("Added measurements for %d lidar(s)",
           static_cast<int>(inputs_.lidar_measurements.size()));
  LOG_INFO("Added measurements for %d camera(s)",
           static_cast<int>(inputs_.camera_measurements.size()));

  // Downsample template cloud
  pcl::VoxelGrid<pcl::PointXYZ> vox;
  vox.setLeafSize(optimizer_params_.template_downsample_size[0],
                  optimizer_params_.template_downsample_size[1],
                  optimizer_params_.template_downsample_size[2]);
  for (int i = 0; i < inputs_.target_params.size(); i++) {
    if (inputs_.target_params[i]->template_cloud->size() > 0) {
      std::shared_ptr<PointCloud> downsampled_cloud =
          std::make_shared<PointCloud>();
      vox.setInputCloud(inputs_.target_params[i]->template_cloud);
      vox.filter(*downsampled_cloud);
      inputs_.target_params[i]->template_cloud = downsampled_cloud;
    }
  }
}

void Optimizer::Solve() {
  ResetViewer();
  CheckInputs();
  AddInitials();

  uint16_t iteration = 0;
  bool converged = false;
  while (!converged) {
    iteration++;
    LOG_INFO("Iteration: %d", iteration);
    Reset();
    GetImageCorrespondences();
    GetLidarCorrespondences();
    if (optimizer_params_.match_centroids_on_first_iter_only &&
        iteration == 1) {
      optimizer_params_.match_centroids = false;
    }
    AddImageMeasurements();
    AddLidarMeasurements();
    Optimize();
    converged = HasConverged(iteration);
    UpdateInitials();
  }
  if (iteration >= optimizer_params_.max_correspondence_iterations) {
    LOG_WARN("Reached max iterations, stopping.");
  } else {
    LOG_INFO("Converged after %d iterations.", iteration);
  }
}

CalibrationResults Optimizer::GetResults() {
  calibration_results_.clear();
  for (uint32_t i = 0; i < inputs_.calibration_initials.size(); i++) {
    vicon_calibration::CalibrationResult calib =
        inputs_.calibration_initials[i];
    calib.transform = GetFinalPose(calib.type, calib.sensor_id);
    calibration_results_.push_back(calib);
  }
  return calibration_results_;
}

void Optimizer::LoadConfigCommon(const nlohmann::json& J) {
  try {
    optimizer_params_.viz_point_size = J.at("viz_point_size");
    optimizer_params_.viz_corr_line_width = J.at("viz_corr_line_width");
    optimizer_params_.max_correspondence_iterations =
        J.at("max_correspondence_iterations");
    optimizer_params_.show_camera_measurements =
        J.at("show_camera_measurements");
    optimizer_params_.show_lidar_measurements = J.at("show_lidar_measurements");
    optimizer_params_.extract_image_target_perimeter =
        J.at("extract_image_target_perimeter");
    optimizer_params_.output_errors = J.at("output_errors");
    optimizer_params_.concave_hull_alpha = J.at("concave_hull_alpha");
    optimizer_params_.max_pixel_cor_dist = J.at("max_pixel_cor_dist");
    optimizer_params_.max_point_cor_dist = J.at("max_point_cor_dist");
    optimizer_params_.match_centroids = J.at("match_centroids");
    optimizer_params_.match_centroids_on_first_iter_only =
        J.at("match_centroids_on_first_iter_only");
    optimizer_params_.print_results_to_terminal =
        J.at("print_results_to_terminal");
    optimizer_params_.estimate_target_corrections =
        J.at("estimate_target_corrections");

    std::vector<double> viewer_backround_color = J["viewer_backround_color"];
    optimizer_params_.viewer_backround_color = viewer_backround_color;

    std::vector<double> error_tol_tmp = J["error_tol"];
    optimizer_params_.error_tol = error_tol_tmp;

    std::vector<double> template_downsample_size_tmp =
        J["template_downsample_size"];
    optimizer_params_.template_downsample_size = template_downsample_size_tmp;
  } catch (const nlohmann::json::exception& e) {
    LOG_ERROR("Cannot load json, one or more missing parameters. Error: %s",
              e.what());
  }

  if (optimizer_params_.error_tol.size() != 2) {
    throw std::invalid_argument{
        "Invalid number of inputs to error_tol. Expecting 6."};
  }

  if (optimizer_params_.template_downsample_size.size() != 3) {
    throw std::invalid_argument{
        "Invalid number of inputs to template_downsample_size. Expecting 3."};
  }
}

void Optimizer::ResetViewer() {
  if (optimizer_params_.show_camera_measurements ||
      optimizer_params_.show_lidar_measurements) {
    pcl_viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>();
  }
}

void Optimizer::CheckInputs() {
  if (inputs_.lidar_measurements.size() == 0) {
    LOG_WARN("No lidar measurements inputted to optimizer.");
  }
  if (inputs_.camera_measurements.size() == 0) {
    LOG_WARN("No camera measurements inputted to optimizer.");
  }
  if (inputs_.calibration_initials.size() == 0) {
    throw std::runtime_error{
        "No initial estimates given to optimizer. Cannot solve."};
  }
  for (uint8_t i = 0; i < inputs_.target_params.size(); i++) {
    if (inputs_.target_params[i]->template_cloud->size() == 0 ||
        inputs_.target_params[i]->template_cloud == nullptr) {
      LOG_WARN("Target No. %d contains an empty template cloud.", i);
      // throw std::runtime_error{
      //     "Missing target required to solve problem, or target is empty."};
    }
  }

  if (inputs_.camera_measurements.size() > 0 &&
      inputs_.camera_params.size() == 0) {
    throw std::runtime_error{"No camera params inputted."};
  }
}

void Optimizer::GetImageCorrespondences() {
  LOG_INFO("Setting image correspondences");
  int counter = 0;
  Eigen::Matrix4d T_Robot_Camera, T_Camera_Target;
  std::shared_ptr<CameraMeasurement> measurement;
  for (uint8_t cam_iter = 0; cam_iter < inputs_.camera_measurements.size();
       cam_iter++) {
    for (uint32_t meas_iter = 0;
         meas_iter < inputs_.camera_measurements[cam_iter].size();
         meas_iter++) {
      if (inputs_.camera_measurements[cam_iter][meas_iter] == nullptr) {
        continue;
      }
      measurement = inputs_.camera_measurements[cam_iter][meas_iter];

      T_Robot_Camera =
          GetUpdatedInitialPose(SensorType::CAMERA, measurement->camera_id);
      T_Camera_Target =
          utils::InvertTransform(T_Robot_Camera) * measurement->T_Robot_Target;

      // convert measurement to 3D (set z to 0)
      PointCloud::Ptr measurement_3d = std::make_shared<PointCloud>();
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
      if (inputs_.target_params[measurement->target_id]
              ->keypoints_camera.cols() > 0) {
        use_target_keypoints = true;
      }

      // get point cloud of projected keypoints
      PointCloud::Ptr projected_keypoints = std::make_shared<PointCloud>();
      PointCloud::Ptr transformed_keypoints = std::make_shared<PointCloud>();
      if (use_target_keypoints) {
        // use keypoints specified in json
        const auto& kpts =
            inputs_.target_params[measurement->target_id]->keypoints_camera;
        for (int k = 0; k < kpts.cols(); k++) {
          Eigen::Vector3d keypoint = kpts.col(k);
          Eigen::Vector4d keypoint_transformed =
              T_Camera_Target * keypoint.homogeneous();
          transformed_keypoints->emplace_back(keypoint_transformed[0],
                                              keypoint_transformed[1],
                                              keypoint_transformed[2]);
        }
      } else {
        // use all points from template cloud
        pcl::transformPointCloud(
            *(inputs_.target_params[measurement->target_id]->template_cloud),
            *transformed_keypoints, T_Camera_Target);
      }

      for (uint32_t i = 0; i < transformed_keypoints->size(); i++) {
        Eigen::Vector3d p(transformed_keypoints->at(i).x,
                          transformed_keypoints->at(i).y,
                          transformed_keypoints->at(i).z);
        bool point_projected_valid;
        Eigen::Vector2d point_projected;
        inputs_.camera_params[measurement->camera_id]
            ->camera_model->ProjectPoint(p, point_projected,
                                         point_projected_valid);
        if (!point_projected_valid) { continue; }
        projected_keypoints->push_back(
            pcl::PointXYZ(point_projected[0], point_projected[1], 0));
      }

      std::shared_ptr<pcl::Correspondences> correspondences =
          std::make_shared<pcl::Correspondences>();
      pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
          corr_est;

      if (optimizer_params_.extract_image_target_perimeter &&
          !use_target_keypoints) {
        // keep only perimeter points
        PointCloud::Ptr hull_cloud = std::make_shared<PointCloud>();
        pcl::PointIndices::Ptr hull_point_correspondences =
            std::make_shared<pcl::PointIndices>();
        pcl::ConcaveHull<pcl::PointXYZ> concave_hull;
        concave_hull.setInputCloud(projected_keypoints);
        concave_hull.setAlpha(optimizer_params_.concave_hull_alpha);
        concave_hull.setKeepInformation(true);
        concave_hull.reconstruct(*hull_cloud);
        concave_hull.getHullPointIndices(*hull_point_correspondences);

        // calculate centroids and translate target to match
        PointCloud::Ptr transformed_keypoints_temp;
        if (optimizer_params_.match_centroids) {
          transformed_keypoints_temp =
              MatchCentroids(measurement_3d, hull_cloud);
        } else {
          transformed_keypoints_temp = hull_cloud;
        }

        // get correspondences
        std::shared_ptr<pcl::Correspondences> correspondences_tmp =
            std::make_shared<pcl::Correspondences>();
        corr_est.setInputSource(measurement_3d);
        corr_est.setInputTarget(transformed_keypoints_temp);
        corr_est.determineCorrespondences(*correspondences_tmp,
                                          optimizer_params_.max_pixel_cor_dist);
        for (int i = 0; i < correspondences_tmp->size(); i++) {
          int measurement_index = correspondences_tmp->at(i).index_query;
          int hull_index = correspondences_tmp->at(i).index_match;
          int target_index = hull_point_correspondences->indices.at(hull_index);
          correspondences->push_back(
              pcl::Correspondence(measurement_index, target_index, 0));
        }
        if (optimizer_params_.show_camera_measurements && !stop_all_vis_) {
          ViewCameraMeasurements(measurement_3d, hull_cloud,
                                 correspondences_tmp, "measured camera points",
                                 "projected camera points");
        }
      } else {
        // calculate centroids and translate target to match
        PointCloud::Ptr transformed_keypoints_temp;
        if (optimizer_params_.match_centroids) {
          transformed_keypoints_temp =
              MatchCentroids(measurement_3d, projected_keypoints);
        } else {
          transformed_keypoints_temp = projected_keypoints;
        }

        // get correspondences
        corr_est.setInputSource(measurement_3d);
        corr_est.setInputTarget(transformed_keypoints_temp);
        corr_est.determineCorrespondences(*correspondences,
                                          optimizer_params_.max_pixel_cor_dist);
        if (optimizer_params_.show_camera_measurements && !stop_all_vis_) {
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

void Optimizer::GetLidarCorrespondences() {
  LOG_INFO("Setting lidar correspondences");
  int counter = 0;
  Eigen::Matrix4d T_Robot_Lidar, T_Lidar_Target;
  std::shared_ptr<LidarMeasurement> measurement;
  for (uint8_t lidar_iter = 0; lidar_iter < inputs_.lidar_measurements.size();
       lidar_iter++) {
    for (uint32_t meas_iter = 0;
         meas_iter < inputs_.lidar_measurements[lidar_iter].size();
         meas_iter++) {
      if (inputs_.lidar_measurements[lidar_iter][meas_iter] == nullptr) {
        continue;
      }
      measurement = inputs_.lidar_measurements[lidar_iter][meas_iter];

      T_Robot_Lidar =
          GetUpdatedInitialPose(SensorType::LIDAR, measurement->lidar_id);
      T_Lidar_Target =
          utils::InvertTransform(T_Robot_Lidar) * measurement->T_Robot_Target;

      // Check keypoints to see if we want to find correspondences between
      // keypoints or between all target points
      PointCloud::Ptr transformed_keypoints = std::make_shared<PointCloud>();
      const auto& kpts =
          inputs_.target_params[measurement->target_id]->keypoints_lidar;
      for (int k = 0; k < kpts.cols(); k++) {
        Eigen::Vector3d keypoint = kpts.col(k);
        Eigen::Vector4d keypoint_transformed =
            T_Lidar_Target * keypoint.homogeneous();
        transformed_keypoints->emplace_back(keypoint_transformed[0],
                                            keypoint_transformed[1],
                                            keypoint_transformed[2]);
      }

      if (kpts.cols() == 0) {
        // use all points from template cloud
        pcl::transformPointCloud(
            *(inputs_.target_params[measurement->target_id]->template_cloud),
            *transformed_keypoints, T_Lidar_Target);
      }

      // calculate centroids and translate target to match
      PointCloud::Ptr transformed_keypoints_temp;
      if (optimizer_params_.match_centroids) {
        transformed_keypoints_temp =
            MatchCentroids(measurement->keypoints, transformed_keypoints);
      } else {
        transformed_keypoints_temp = transformed_keypoints;
      }

      // get correspondences
      pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
          corr_est;
      std::shared_ptr<pcl::Correspondences> correspondences =
          std::make_shared<pcl::Correspondences>();
      corr_est.setInputSource(measurement->keypoints);
      corr_est.setInputTarget(transformed_keypoints_temp);
      corr_est.determineCorrespondences(*correspondences,
                                        optimizer_params_.max_point_cor_dist);
      if (optimizer_params_.show_lidar_measurements && !stop_all_vis_) {
        ViewLidarMeasurements(measurement->keypoints, transformed_keypoints,
                              correspondences, "measured lidar keypoints",
                              "estimated lidar keypoints");
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

PointCloud::Ptr Optimizer::MatchCentroids(const PointCloud::Ptr& source_cloud,
                                          const PointCloud::Ptr& target_cloud) {
  PointCloud::Ptr target_translated = std::make_shared<PointCloud>();
  Eigen::Vector4d source_centroid, target_centroid;
  pcl::compute3DCentroid(*source_cloud, source_centroid);
  pcl::compute3DCentroid(*target_cloud, target_centroid);
  Eigen::Vector3d t_SOURCE_TARGET =
      source_centroid.hnormalized() - target_centroid.hnormalized();
  Eigen::Matrix4d T_Source_Target;
  T_Source_Target.setIdentity();
  T_Source_Target.block(0, 3, 3, 1) = t_SOURCE_TARGET;
  pcl::transformPointCloud(*target_cloud, *target_translated, T_Source_Target);
  return target_translated;
}

void Optimizer::ViewCameraMeasurements(
    const PointCloud::Ptr& c1, const PointCloud::Ptr& c2,
    const std::shared_ptr<pcl::Correspondences>& correspondences,
    const std::string& c1_name, const std::string& c2_name) {
  PointCloudColor::Ptr c1_col = std::make_shared<PointCloudColor>();
  PointCloudColor::Ptr c2_col = std::make_shared<PointCloudColor>();
  uint32_t rgb1 = (static_cast<uint32_t>(255) << 16 |
                   static_cast<uint32_t>(0) << 8 | static_cast<uint32_t>(0));
  uint32_t rgb2 = (static_cast<uint32_t>(0) << 16 |
                   static_cast<uint32_t>(255) << 8 | static_cast<uint32_t>(0));
  pcl::PointXYZRGB point;
  for (PointCloud::iterator it = c1->begin(); it != c1->end(); ++it) {
    point.x = it->x;
    point.y = it->y;
    point.z = it->z;
    point.rgb = *reinterpret_cast<float*>(&rgb1);
    c1_col->push_back(point);
  }
  for (PointCloud::iterator it = c2->begin(); it != c2->end(); ++it) {
    point.x = it->x;
    point.y = it->y;
    point.z = it->z;
    point.rgb = *reinterpret_cast<float*>(&rgb2);
    c2_col->push_back(point);
  }
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1_(
      c1_col);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2_(
      c2_col);
  ResetViewer();
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(c1_col, rgb1_, c1_name);
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(c2_col, rgb2_, c2_name);
  std::string shape_id = "correspondences";
  pcl_viewer_->addCorrespondences<pcl::PointXYZRGB>(c1_col, c2_col,
                                                    *correspondences, shape_id);
  pcl_viewer_->setBackgroundColor(optimizer_params_.viewer_backround_color[0],
                                  optimizer_params_.viewer_backround_color[1],
                                  optimizer_params_.viewer_backround_color[2]);
  pcl_viewer_->setShapeRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
      optimizer_params_.viz_corr_line_width, shape_id);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
      optimizer_params_.viz_point_size, c1_name);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
      optimizer_params_.viz_point_size, c2_name);

  std::cout << "\nViewer Legend:\n"
            << "  Red   -> " << c1_name << "\n"
            << "  Green -> " << c2_name << "\n"
            << "Press [c] to continue\n"
            << "Press [n] to skip to next iteration\n"
            << "Press [s] to stop showing these measurements.\n";
  while (!pcl_viewer_->wasStopped() && !close_viewer_) {
    pcl_viewer_->spinOnce(10);
    pcl_viewer_->registerKeyboardCallback(
        &Optimizer::ConfirmMeasurementKeyboardCallback, *this);
    std::this_thread::sleep_for(10ms);
  }
  close_viewer_ = false;
  pcl_viewer_->removeAllPointClouds();
  pcl_viewer_->removeAllShapes();
  pcl_viewer_->resetStoppedFlag();
  pcl_viewer_->close();
}

void Optimizer::ViewLidarMeasurements(
    const PointCloud::Ptr& c1, const PointCloud::Ptr& c2,
    const std::shared_ptr<pcl::Correspondences>& correspondences,
    const std::string& c1_name, const std::string& c2_name) {
  PointCloudColor::Ptr c1_col = std::make_shared<PointCloudColor>();
  PointCloudColor::Ptr c2_col = std::make_shared<PointCloudColor>();

  uint32_t rgb1 = (static_cast<uint32_t>(255) << 16 |
                   static_cast<uint32_t>(0) << 8 | static_cast<uint32_t>(0));
  uint32_t rgb2 = (static_cast<uint32_t>(0) << 16 |
                   static_cast<uint32_t>(255) << 8 | static_cast<uint32_t>(0));
  pcl::PointXYZRGB point;
  for (PointCloud::iterator it = c1->begin(); it != c1->end(); ++it) {
    point.x = it->x;
    point.y = it->y;
    point.z = it->z;
    point.rgb = *reinterpret_cast<float*>(&rgb1);
    c1_col->push_back(point);
  }
  for (PointCloud::iterator it = c2->begin(); it != c2->end(); ++it) {
    point.x = it->x;
    point.y = it->y;
    point.z = it->z;
    point.rgb = *reinterpret_cast<float*>(&rgb2);
    c2_col->push_back(point);
  }
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1_(
      c1_col);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2_(
      c2_col);
  ResetViewer();
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(c1_col, rgb1_, c1_name);
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(c2_col, rgb2_, c2_name);
  std::string shape_id = "correspondences";
  pcl_viewer_->addCorrespondences<pcl::PointXYZRGB>(c1_col, c2_col,
                                                    *correspondences, shape_id);
  pcl_viewer_->setBackgroundColor(optimizer_params_.viewer_backround_color[0],
                                  optimizer_params_.viewer_backround_color[1],
                                  optimizer_params_.viewer_backround_color[2]);
  pcl_viewer_->setShapeRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
      optimizer_params_.viz_corr_line_width, shape_id);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
      optimizer_params_.viz_point_size, c1_name);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
      optimizer_params_.viz_point_size, c2_name);

  std::cout << "\nViewer Legend:\n"
            << "  Red   -> " << c1_name << "\n"
            << "  Green -> " << c2_name << "\n"
            << "Press [c] to continue\n"
            << "Press [n] to skip to next iteration\n"
            << "Press [s] to stop showing these measurements\n";
  while (!pcl_viewer_->wasStopped() && !close_viewer_) {
    pcl_viewer_->spinOnce(10);
    pcl_viewer_->registerKeyboardCallback(
        &Optimizer::ConfirmMeasurementKeyboardCallback, *this);
    std::this_thread::sleep_for(10ms);
  }
  close_viewer_ = false;
  pcl_viewer_->removeAllPointClouds();
  pcl_viewer_->removeAllShapes();
  pcl_viewer_->close();
  pcl_viewer_->resetStoppedFlag();
}

void Optimizer::ConfirmMeasurementKeyboardCallback(
    const pcl::visualization::KeyboardEvent& event, void* viewer_void) {
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

bool Optimizer::HasConverged(uint16_t iteration) {
  if (iteration == 0) {
    return false;
  } else if (iteration == optimizer_params_.max_correspondence_iterations) {
    return true;
  }

  // Loop through results
  Eigen::Matrix4d T_last, T_curr;
  Eigen::Matrix3d R_error;
  Eigen::Vector3d t_curr, t_last, t_error, rpy_error;
  for (uint32_t i = 0; i < inputs_.calibration_initials.size(); i++) {
    T_last = GetUpdatedInitialPose(inputs_.calibration_initials[i].type,
                                   inputs_.calibration_initials[i].sensor_id);
    T_curr = GetFinalPose(inputs_.calibration_initials[i].type,
                          inputs_.calibration_initials[i].sensor_id);

    // Check all DOFs to see if the change is greater than the tolerance
    double error_t_m =
        (T_curr.block(0, 3, 3, 1) - T_last.block(0, 3, 3, 1)).norm();
    double error_r_rad = utils::CalculateRotationError(
        T_curr.block(0, 0, 3, 3), T_last.block(0, 0, 3, 3));

    if (optimizer_params_.output_errors) {
      // Output transforms:
      std::string transform_name =
          "T_" + inputs_.calibration_initials[i].to_frame + "_" +
          inputs_.calibration_initials[i].from_frame;
      utils::OutputTransformInformation(T_last, transform_name + "_prev");
      utils::OutputTransformInformation(T_last, transform_name + "_current");

      // Output errors:
      std::cout << "rotation error (deg): " << utils::RadToDeg(error_r_rad)
                << "\n"
                << "rotation threshold (deg): "
                << utils::RadToDeg(optimizer_params_.error_tol[0]) << "\n"
                << "translation error (m): " << error_t_m << "\n"
                << "translation threshold (m): "
                << optimizer_params_.error_tol[1] << "\n";
    }
    if (error_r_rad > optimizer_params_.error_tol[0]) { return false; }
    if (error_t_m > optimizer_params_.error_tol[1]) { return false; }
  }
  return true;
}

} // end namespace vicon_calibration
