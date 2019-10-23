#include "vicon_calibration/CameraExtractor.h"
#include <boost/make_shared.hpp>

namespace vicon_calibration {
CameraExtractor::CameraExtractor() {
  keypoints_measured_ = boost::make_shared<pcl::PointCloud<pcl::PointXY>>();
}

void CameraExtractor::SetCameraParams(CameraParams &camera_params) {
  camera_params_ = camera_params;
}

void CameraExtractor::SetTargetParams(TargetParams &target_params) {
  target_params_ = target_params;
}

bool CameraExtractor::GetMeasurementValid() {
  if (!measurement_complete_) {
    throw std::invalid_argument {
      "Cannot retrieve measurement, please run ExtractKeypoints before "
      "attempting to retrieve measurement."
    };
  }
  return measurement_valid_;
}

pcl::PointCloud<pcl::PointXY>::Ptr CameraExtractor::GetMeasurement() {
  if (!measurement_complete_) {
    throw std::invalid_argument {
      "Cannot retrieve measurement, please run ExtractKeypoints before "
      "attempting to retrieve measurement."
    };
  }
  return keypoints_measured_;
}

} // namespace vicon_calibration
