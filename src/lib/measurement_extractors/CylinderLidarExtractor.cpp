#include "vicon_calibration/measurement_extractors/CylinderLidarExtractor.h"

#include <pcl/search/impl/search.hpp>

namespace vicon_calibration {

void CylinderLidarExtractor::GetKeypoints() {
  PointCloud::Ptr scan_registered = boost::make_shared<PointCloud>();
  IterativeClosestPointCustom<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setTransformationEpsilon(icp_transform_epsilon_);
  icp.setEuclideanFitnessEpsilon(icp_euclidean_epsilon_);
  icp.setMaximumIterations(icp_max_iterations_);
  icp.setMaxCorrespondenceDistance(icp_max_correspondence_dist_);
  icp.setInputSource(scan_isolated_);
  icp.setInputTarget(target_params_->template_cloud);
  icp.align(*scan_registered, T_LIDAR_TARGET_EST_.inverse().cast<float>());

  if (!icp.hasConverged()) {
    measurement_valid_ = false;
    if (show_measurements_) {
      std::cout << "ICP failed." << std::endl;
    }
    T_LIDAR_TARGET_OPT_ = T_LIDAR_TARGET_EST_;
    keypoints_measured_ = scan_isolated_;
    return;
  }

  keypoints_measured_->clear();
  measurement_valid_ = true;
  T_LIDAR_TARGET_OPT_ =
      utils::InvertTransform(icp.getFinalTransformation().cast<double>());

  // remove points with large correspondence distances
  pcl::CorrespondencesPtr correspondences = icp.getCorrespondencesPtr();
  for (uint32_t i = 0; i < correspondences->size(); i++) {
    pcl::Correspondence corr_i = (*correspondences)[i];
    int source_index_i = corr_i.index_query;
    if (corr_i.distance < max_keypoint_distance_) {
      pcl::PointXYZ point = scan_isolated_->points[source_index_i];
      keypoints_measured_->points.push_back(point);
    }
  }
}


void CylinderLidarExtractor::CheckMeasurementValid() {
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  PointCloud::Ptr template_transformed = boost::make_shared<PointCloud>();
  pcl::transformPointCloud(*target_params_->template_cloud,
                           *template_transformed,
                           T_LIDAR_TARGET_OPT_.cast<float>());
  kd_tree.setInputCloud(template_transformed);
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it =
           keypoints_measured_->begin();
       it != keypoints_measured_->end(); ++it) {
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
