#include "vicon_calibration/LidarExtractor.h"
#include <boost/make_shared.hpp>

namespace vicon_calibration {
LidarExtractor::LidarExtractor() {
  keypoints_measured_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
}

void LidarExtractor::SetLidarParams(LidarParams &lidar_params) {
  lidar_params_ = lidar_params;
}

void LidarExtractor::SetTargetParams(TargetParams &target_params) {
  target_params_ = target_params;
}

bool LidarExtractor::GetMeasurementValid() {
  if (!measurement_complete_) {
    throw std::invalid_argument {
      "Cannot retrieve measurement, please run ExtractKeypoints before "
      "attempting to retrieve measurement."
    };
  }
  return measurement_valid_;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr LidarExtractor::GetMeasurement() {
  if (!measurement_complete_) {
    throw std::invalid_argument {
      "Cannot retrieve measurement, please run ExtractKeypoints before "
      "attempting to retrieve measurement."
    };
  }
  return keypoints_measured_;
}

} // namespace vicon_calibration
