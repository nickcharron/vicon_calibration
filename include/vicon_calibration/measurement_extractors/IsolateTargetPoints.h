#pragma once

#include "vicon_calibration/utils.h"

namespace vicon_calibration {

class IsolateTargetPoints {
public:
  IsolateTargetPoints() = default;

  ~IsolateTargetPoints() = default;

  void LoadConfig(const std::string &config_file);

  void SetTransformEstimate(const Eigen::Matrix4d &T_TARGET_LIDAR);

  void SetScan(const PointCloud::Ptr &scan_in);

  void SetTargetParams(const TargetParams &target_params);

  PointCloud::Ptr GetPoints();

private:
  bool CheckInputs();

  void CropScan();

  PointCloud::Ptr scan_in_{nullptr};
  PointCloud::Ptr scan_cropped_{nullptr};
  PointCloud::Ptr scan_isolated_{nullptr};
  Eigen::Matrix4d T_TARGET_LIDAR_{NULL}; // TODO: check this
  bool crop_scan_{true};
  TargetParams target_params_{NULL};
  double isolator_volume_weight_{0.5};
  double isolator_distance_weight_{0.5};
};

} // namespace vicon_calibration
