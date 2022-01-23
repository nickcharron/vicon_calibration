#include <vicon_calibration/measurement_extractors/DiamondCornersLidarExtractor.h>

#include <math.h>

#include <pcl/kdtree/kdtree_flann.h>

namespace vicon_calibration {

void DiamondCornersLidarExtractor::GetKeypoints() {
  measurement_valid_ = true;

  // setup icp
  PointCloud::Ptr scan_registered = std::make_shared<PointCloud>();
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setTransformationEpsilon(icp_transform_epsilon_);
  icp.setEuclideanFitnessEpsilon(icp_euclidean_epsilon_);
  icp.setMaximumIterations(icp_max_iterations_);
  icp.setMaxCorrespondenceDistance(icp_max_correspondence_dist_);

  icp.setInputSource(scan_isolated_);
  icp.setInputTarget(target_params_->template_cloud);
  icp.align(*scan_registered,
            utils::InvertTransform(T_LIDAR_TARGET_EST_).cast<float>());

  if (!icp.hasConverged()) {
    measurement_valid_ = false;
    if (show_measurements_) { std::cout << "ICP failed." << std::endl; }
    T_LIDAR_TARGET_OPT_ = T_LIDAR_TARGET_EST_;
  } else {
    measurement_valid_ = true;
    T_LIDAR_TARGET_OPT_ =
        utils::InvertTransform(icp.getFinalTransformation().cast<double>());
  }

  // transform keypoints from json using opt. transform and store
  Eigen::Vector4d keypoint_trans;
  keypoints_measured_->clear();
  for (int k = 0; k < target_params_->keypoints_lidar.cols(); k++) {
    Eigen::Vector4d keypoint(target_params_->keypoints_lidar(0, k),
                             target_params_->keypoints_lidar(1, k),
                             target_params_->keypoints_lidar(2, k), 1);
    keypoint_trans = T_LIDAR_TARGET_OPT_ * keypoint;
    keypoints_measured_->push_back(
        pcl::PointXYZ(keypoint_trans[0], keypoint_trans[1], keypoint_trans[2]));
  }
}

void DiamondCornersLidarExtractor::CheckMeasurementValid() {
  if (!measurement_valid_) { return; }

  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  PointCloud::Ptr template_transformed = std::make_shared<PointCloud>();
  pcl::transformPointCloud(*target_params_->template_cloud,
                           *template_transformed,
                           T_LIDAR_TARGET_OPT_.cast<float>());
  kd_tree.setInputCloud(template_transformed);
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it = scan_isolated_->begin();
       it != scan_isolated_->end(); ++it) {
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    int num_neighbors =
        kd_tree.radiusSearch(*it, allowable_keypoint_error_,
                             pointIdxRadiusSearch, pointRadiusSquaredDistance);
    if (num_neighbors == 0) {
      measurement_valid_ = false;
      return;
    }
  }
  measurement_valid_ = true;
}

} // namespace vicon_calibration
