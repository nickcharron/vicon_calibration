#include <vicon_calibration/measurement_extractors/CylinderLidarExtractor.h>

#include <pcl/search/impl/search.hpp>

namespace vicon_calibration {

void CylinderLidarExtractor::GetKeypoints() {
  measurement_valid_ = true;
  
  PointCloud::Ptr scan_registered = std::make_shared<PointCloud>();
  IterativeClosestPointCustom<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setTransformationEpsilon(icp_transform_epsilon_);
  icp.setEuclideanFitnessEpsilon(icp_euclidean_epsilon_);
  icp.setMaximumIterations(icp_max_iterations_);
  icp.setMaxCorrespondenceDistance(icp_max_correspondence_dist_);
  icp.setInputSource(scan_isolated_);
  icp.setInputTarget(target_params_->template_cloud);
  icp.align(*scan_registered, T_Lidar_Target_Est_.inverse().cast<float>());

  if (!icp.hasConverged()) {
    measurement_valid_ = false;
    if (show_measurements_) {
      std::cout << "ICP failed." << std::endl;
    }
    T_Lidar_Target_Opt_ = T_Lidar_Target_Est_;
    keypoints_measured_ = scan_isolated_;
    return;
  }

  keypoints_measured_->clear();
  measurement_valid_ = true;
  T_Lidar_Target_Opt_ =
      utils::InvertTransform(icp.getFinalTransformation().cast<double>());

  // get correspondences from ICP and store as keypoints
  pcl::Correspondences correspondences = *icp.getCorrespondencesPtr();
  for (pcl::Correspondence correspondence : correspondences) {
    pcl::PointXYZ point = scan_isolated_->at(correspondence.index_query);
    keypoints_measured_->push_back(point);
  }
}


void CylinderLidarExtractor::CheckMeasurementValid() {
  if(!measurement_valid_){
    return;
  }

  if (keypoints_measured_->size() < min_num_keypoints_) {
    measurement_valid_ = false;
  } else {
    measurement_valid_ = true;
  }

}

} // namespace vicon_calibration
