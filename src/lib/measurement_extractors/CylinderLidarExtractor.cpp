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
      boost::shared_ptr<PointCloudColor>
          scan_cropped_coloured;
      scan_cropped_coloured =
          boost::make_shared<PointCloudColor>();
      Eigen::MatrixXd T_identity = Eigen::MatrixXd(4, 4);
      T_identity.setIdentity();
      scan_cropped_coloured = utils::ColorPointCloud(scan_cropped_, 255, 0, 0);
      this->AddColouredPointCloudToViewer(scan_cropped_coloured,
                                          "red_cloud", T_identity);
      this->AddPointCloudToViewer(scan_in_, "white_cloud", T_identity);
      this->ShowFailedMeasurement();
      pcl_viewer_->resetStoppedFlag();
    }
    return;
  }

  T_LIDAR_TARGET_OPT_ = icp.getFinalTransformation().inverse().cast<double>();
  this->CalculateErrors();
  this->CheckErrors();

  // remove points with large correspondence distances
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

void CylinderLidarExtractor::CalculateErrors() {
  Eigen::Vector4d point_end_tgt(0.3, 0, 0, 1);
  Eigen::Vector4d point_orig_tgt(0, 0, 0, 1);
  Eigen::Vector4d point_orig_lid_est = T_LIDAR_TARGET_EST_ * point_orig_tgt;
  Eigen::Vector4d point_orig_lid_opt = T_LIDAR_TARGET_OPT_ * point_orig_tgt;
  Eigen::Vector4d point_end_lid_est = T_LIDAR_TARGET_EST_ * point_end_tgt;
  Eigen::Vector4d point_end_lid_opt = T_LIDAR_TARGET_OPT_ * point_end_tgt;

  // Calculate two error quantities:
  // (1) angles (azimuth and zenith),
  // (2) origin location
  Eigen::Vector4d dxdydz_est = point_end_lid_est - point_orig_lid_est;
  Eigen::Vector4d dxdydz_opt = point_end_lid_opt - point_orig_lid_opt;
  double azimuth_est = atan(dxdydz_est[1] / dxdydz_est[0]);
  double zenith_est = atan(dxdydz_est[2] / sqrt(dxdydz_est[0] * dxdydz_est[0] +
                                                dxdydz_est[1] * dxdydz_est[1]));
  double azimuth_opt = atan(dxdydz_opt[1] / dxdydz_opt[0]);
  double zenith_opt = atan(dxdydz_opt[2] / sqrt(dxdydz_opt[0] * dxdydz_opt[0] +
                                                dxdydz_opt[1] * dxdydz_opt[1]));

  azimuth_est = utils::WrapToTwoPi(azimuth_est);
  zenith_est = utils::WrapToTwoPi(zenith_est);
  azimuth_opt = utils::WrapToTwoPi(azimuth_opt);
  zenith_opt = utils::WrapToTwoPi(zenith_opt);

  Eigen::Vector2d error_angle;
  error_angle[0] = sqrt((azimuth_opt - azimuth_est) *
                        (azimuth_opt - azimuth_est) * 1000000) /
                   1000;
  error_angle[1] =
      sqrt((zenith_opt - zenith_est) * (zenith_opt - zenith_est) * 1000000) /
      1000;

  Eigen::Vector3d error_origin = T_LIDAR_TARGET_OPT_.block(0, 3, 3, 1) -
                                 T_LIDAR_TARGET_EST_.block(0, 3, 3, 1);

  error_[0] = RAD_TO_DEG * error_angle[0];
  error_[1] = RAD_TO_DEG * error_angle[1];
  error_[2] = error_origin.norm() / point_orig_lid_est.block(0, 0, 3, 1).norm();

  if(error_[0] > 180){
    error_[0] = error_[0] - 180;
  }
  if(error_[1] > 180){
    error_[1] = error_[1] - 180;
  }

}

void CylinderLidarExtractor::CheckErrors() {
  if (error_[0] > rot_acceptance_criteria_ ||
      error_[1] > rot_acceptance_criteria_ ||
      error_[2] > dist_acceptance_criteria_) {
    measurement_valid_ = false;
  } else {
    measurement_valid_ = true;
  }

  if (show_measurements_) {
    if (!measurement_valid_) {
      std::cout << "-----------------------------\n"
                << "Measurement Invalid\n"
                << "Target relative distance error: " << error_[2] << "\n"
                << "Relative distance acceptance criteria: "
                << dist_acceptance_criteria_ << "\n"
                << "Target angle errors [azimuth, zenith]: [" << error_[0]
                << ", " << error_[1] << "]\n"
                << "Rotation acceptance criteria: " << rot_acceptance_criteria_
                << "\n";
    } else {
      std::cout << "-----------------------------\n"
                << "Measurement Valid\n"
                << "Target relative distance error: " << error_[2] << "\n"
                << "Relative distance acceptance criteria: "
                << dist_acceptance_criteria_ << "\n"
                << "Target angle errors [azimuth, zenith]: [" << error_[0]
                << ", " << error_[1] << "]\n"
                << "Rotation acceptance criteria: " << rot_acceptance_criteria_
                << "\n";
    }

    // Display clouds for testing
    // transform template cloud from target to lidar
    PointCloudColor::Ptr estimated_template_cloud =
        boost::make_shared<PointCloudColor>();
    estimated_template_cloud =
        utils::ColorPointCloud(target_params_->template_cloud, 0, 0, 255);
    pcl::transformPointCloud(*estimated_template_cloud,
                             *estimated_template_cloud,
                             T_LIDAR_TARGET_EST_.cast<float>());
    this->AddColouredPointCloudToViewer(estimated_template_cloud,
                                        "blue_cloud",
                                        T_LIDAR_TARGET_EST_);

    PointCloudColor::Ptr measured_template_cloud =
        boost::make_shared<PointCloudColor>();
    measured_template_cloud =
        utils::ColorPointCloud(target_params_->template_cloud, 0, 255, 0);
    pcl::transformPointCloud(*measured_template_cloud, *measured_template_cloud,
                             T_LIDAR_TARGET_OPT_.cast<float>());
    this->AddColouredPointCloudToViewer(measured_template_cloud,
                                        "green_cloud",
                                        T_LIDAR_TARGET_OPT_);
    Eigen::Matrix4d T_identity;
    T_identity.setIdentity();
    this->AddPointCloudToViewer(scan_cropped_, "white_cloud", T_identity);
    this->ShowFinalTransformation();
  }
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
