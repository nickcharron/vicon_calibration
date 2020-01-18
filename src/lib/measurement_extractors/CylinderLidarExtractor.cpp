#include "vicon_calibration/measurement_extractors/CylinderLidarExtractor.h"

namespace vicon_calibration {

void CylinderLidarExtractor::GetKeypoints() {
  scan_best_points_ = boost::make_shared<PointCloud>();

  if (!test_registration_) {
    return;
  }

  measurement_failed_ = false;
  boost::shared_ptr<PointCloud> scan_registered;
  scan_registered = boost::make_shared<PointCloud>();
  IterativeClosestPointCustom<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setTransformationEpsilon(icp_transform_epsilon_);
  icp.setEuclideanFitnessEpsilon(icp_euclidean_epsilon_);
  icp.setMaximumIterations(icp_max_iterations_);
  icp.setMaxCorrespondenceDistance(icp_max_correspondence_dist_);
  if (crop_scan_) {
    icp.setInputSource(scan_cropped_);
  } else {
    icp.setInputSource(scan_in_);
  }
  icp.setInputTarget(target_params_->template_cloud);
  icp.align(*scan_registered, T_LIDAR_TARGET_EST_.inverse().cast<float>());

  if (!icp.hasConverged()) {
    measurement_failed_ = true;
    measurement_valid_ = false;
    measurement_complete_ = true;
    if (show_measurements_) {
      std::cout << "ICP failed. Displaying cropped scan." << std::endl;
      boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>
          scan_cropped_coloured;
      scan_cropped_coloured =
          boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
      Eigen::MatrixXd T_identity = Eigen::MatrixXd(4,4);
      T_identity.setIdentity();
      scan_cropped_coloured = this->ColourPointCloud(scan_cropped_, 255, 0, 0);
      this->AddColouredPointCloudToViewer(scan_cropped_coloured,
                                          "Coloured Cropped Scan ", T_identity);
      this->AddPointCloudToViewer(scan_in_, "Input Scan", T_identity);
      this->ShowFailedMeasurement();
      pcl_viewer_->resetStoppedFlag();
    }
    return;
  }

  Eigen::MatrixXd T_LIDAR_TARGET_OPT =
      icp.getFinalTransformation().inverse().cast<double>();
  Eigen::Vector4d point_end_tgt(0.3, 0, 0, 1);
  Eigen::Vector4d point_end_est = T_LIDAR_TARGET_EST_ * point_end_tgt;
  Eigen::Vector4d point_end_opt = T_LIDAR_TARGET_OPT * point_end_tgt;

  Eigen::Vector4d error_end = point_end_opt - point_end_est;
  error_end[0] = sqrt(error_end[0] * error_end[0] * 1000000) / 1000;
  error_end[1] = sqrt(error_end[1] * error_end[1] * 1000000) / 1000;
  error_end[2] = sqrt(error_end[2] * error_end[2] * 1000000) / 1000;

  Eigen::Vector3d error_origin = T_LIDAR_TARGET_OPT.block(0, 3, 3, 1) -
                                 T_LIDAR_TARGET_EST_.block(0, 3, 3, 1);
  error_origin[0] = sqrt(error_origin[0] * error_origin[0] * 1000000) / 1000;
  error_origin[1] = sqrt(error_origin[1] * error_origin[1] * 1000000) / 1000;
  error_origin[2] = sqrt(error_origin[2] * error_origin[2] * 1000000) / 1000;

  if (error_origin[0] > dist_acceptance_criteria_ ||
      error_origin[1] > dist_acceptance_criteria_ ||
      error_origin[2] > dist_acceptance_criteria_ ||
      error_end[0] > dist_acceptance_criteria_ ||
      error_end[1] > dist_acceptance_criteria_ ||
      error_end[2] > dist_acceptance_criteria_) {
    measurement_valid_ = false;
  } else {
    measurement_valid_ = true;
  }

  if (show_measurements_) {
    if (!measurement_valid_) {
      std::cout << "-----------------------------\n"
                << "Measurement Invalid\n"
                << "Target origin error: [" << error_origin[0] << ", "
                << error_origin[2] << ", " << error_origin[2] << "]\n"
                << "Target end error: [" << error_end[0] << ", "
                << error_end[2] << ", " << error_end[2] << "]\n"
                << "Distance acceptance criteria: " << dist_acceptance_criteria_
                << "\n";
    } else {
      std::cout << "-----------------------------\n"
                << "Measurement Valid\n"
                << "Target origin error: [" << error_origin[0] << ", "
                << error_origin[2] << ", " << error_origin[2] << "]\n"
                << "Target end error: [" << error_end[0] << ", "
                << error_end[2] << ", " << error_end[2] << "]\n"
                << "Distance acceptance criteria: " << dist_acceptance_criteria_
                << "\n";
    }

    // Display clouds for testing
    // transform template cloud from target to lidar
    PointCloudColor::Ptr estimated_template_cloud =
        boost::make_shared<PointCloudColor>();
    estimated_template_cloud =
        this->ColourPointCloud(target_params_->template_cloud, 0, 0, 255);
    pcl::transformPointCloud(*estimated_template_cloud,
                             *estimated_template_cloud,
                             T_LIDAR_TARGET_EST_.cast<float>());
    this->AddColouredPointCloudToViewer(estimated_template_cloud,
                                        "estimated template cloud ",
                                        T_LIDAR_TARGET_EST_);

    PointCloudColor::Ptr measured_template_cloud =
        boost::make_shared<PointCloudColor>();
    measured_template_cloud =
        this->ColourPointCloud(target_params_->template_cloud, 0, 255, 0);
    pcl::transformPointCloud(*measured_template_cloud, *measured_template_cloud,
                             T_LIDAR_TARGET_OPT.cast<float>());
    this->AddColouredPointCloudToViewer(
        measured_template_cloud, "measured template cloud", T_LIDAR_TARGET_OPT);
    Eigen::Matrix4d T_identity;
    T_identity.setIdentity();
    this->AddPointCloudToViewer(scan_cropped_, "cropped scan", T_identity);
    this->ShowFinalTransformation();
  }

  pcl::CorrespondencesPtr correspondences = icp.getCorrespondencesPtr();
  for (uint32_t i = 0; i < correspondences->size(); i++) {
    pcl::Correspondence corr_i = (*correspondences)[i];
    int source_index_i = corr_i.index_query;
    if (corr_i.distance < max_keypoint_distance_) {
      pcl::PointXYZ point;
      if (crop_scan_) {
        point = scan_cropped_->points[source_index_i];
      } else {
        point = scan_in_->points[source_index_i];
      }
      scan_best_points_->points.push_back(point);
    }
  }
  this->SaveMeasurement();
}

void CylinderLidarExtractor::SaveMeasurement() {
  if (!measurement_valid_) {
    return;
  }
  if (!crop_scan_) {
    keypoints_measured_ = scan_in_;
  } else {
    keypoints_measured_ = scan_best_points_;
  }
}

} // namespace vicon_calibration
