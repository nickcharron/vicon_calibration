#include "vicon_calibration/LidarExtractor.h"
#include <boost/make_shared.hpp>

namespace vicon_calibration {
LidarExtractor::LidarExtractor() {
  keypoints_measured_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
}

void LidarExtractor::SetLidarParams(
    std::shared_ptr<vicon_calibration::LidarParams> &lidar_params) {
  lidar_params_ = lidar_params;
  lidar_params_set_ = true;
}

void LidarExtractor::SetTargetParams(
    std::shared_ptr<vicon_calibration::TargetParams> &target_params) {
  target_params_ = target_params;
  target_params_set_ = true;
  if (target_params_->crop_scan[0] == 0 && target_params_->crop_scan[1] == 0 &&
      target_params_->crop_scan[2] == 0) {
    crop_scan_ = false;
  } else {
    crop_scan_ = true;
  }
}

void LidarExtractor::SetShowMeasurements(bool show_measurements) {
  show_measurements_ = show_measurements;
}

bool LidarExtractor::GetMeasurementValid() {
  if (!measurement_complete_) {
    throw std::invalid_argument{
        "Cannot retrieve measurement, please run ExtractKeypoints before "
        "attempting to retrieve measurement."};
  }
  return measurement_valid_;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr LidarExtractor::GetMeasurement() {
  if (!measurement_complete_) {
    throw std::invalid_argument{
        "Cannot retrieve measurement, please run ExtractKeypoints before "
        "attempting to retrieve measurement."};
  }
  return keypoints_measured_;
}

} // namespace vicon_calibration
