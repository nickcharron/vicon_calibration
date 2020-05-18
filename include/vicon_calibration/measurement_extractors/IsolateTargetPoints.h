#pragma once

#include "vicon_calibration/utils.h"

#include <pcl/filters/extract_indices.h>

namespace vicon_calibration {

class IsolateTargetPoints {
public:
  IsolateTargetPoints() = default;

  ~IsolateTargetPoints() = default;

  void SetConfig(const std::string &config_file);

  void SetTransformEstimate(const Eigen::MatrixXd &T_TARGET_LIDAR);

  void SetScan(const PointCloud::Ptr &scan_in);

  void SetTargetParams(const std::shared_ptr<TargetParams> &target_params);

  void SetLidarParams(const std::shared_ptr<LidarParams> &lidar_params);

  PointCloud::Ptr GetPoints();

  std::vector<PointCloud::Ptr> GetClusters();

  PointCloud::Ptr GetCroppedScan();

private:
  void LoadConfig();

  bool CheckInputs();

  void CropScan();

  void ClusterPoints();

  void GetTargetCluster();

  // This takes a point cloud, runs PCA to get principal components, then
  // projects the points into the new vector space with origin at the centroid
  // Then it calculates the min and max in all 3 axes to get 3d dimensions.
  // If the target is 2d, it returns the area, otherwise the volume
  // This should result in the same volume calculation regardless of input cloud
  // position and orientation. It's not quite the minimal but it's close
  double CalculateMinimalSize(const PointCloud::Ptr &cloud);

  // member variables
  PointCloud::Ptr scan_in_;
  PointCloud::Ptr scan_cropped_;
  PointCloud::Ptr scan_isolated_;
  Eigen::MatrixXd T_TARGET_LIDAR_;
  bool transform_estimate_set_{false};
  std::vector<pcl::PointIndices> cluster_indices_;

  // params
  std::string config_file_{""};
  bool crop_scan_{true};
  std::shared_ptr<TargetParams> target_params_{nullptr};
  std::shared_ptr<LidarParams> lidar_params_{nullptr};
  double isolator_size_weight_{0.5};
  double isolator_distance_weight_{0.5};
  double clustering_multiplier_{1.4};
  int min_cluster_size_{30};
  int max_cluster_size_{10000};
  bool output_cluster_scores_{true};
};

} // namespace vicon_calibration
