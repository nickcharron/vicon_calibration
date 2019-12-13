#include "vicon_calibration/CameraExtractor.h"
#include <boost/make_shared.hpp>

namespace vicon_calibration {
CameraExtractor::CameraExtractor() {
  keypoints_measured_ = boost::make_shared<pcl::PointCloud<pcl::PointXY>>();
}

void CameraExtractor::SetCameraParams(std::shared_ptr<vicon_calibration::CameraParams> &camera_params) {
  camera_params_ = camera_params;
  camera_model_ = beam_calibration::CameraModel::LoadJSON(camera_params->intrinsics);
  camera_params_set_ = true;
}

void CameraExtractor::SetTargetParams(std::shared_ptr<vicon_calibration::TargetParams> &target_params) {
  target_params_ = target_params;
  target_params_set_ = true;
}

void CameraExtractor::SetShowMeasurements(bool show_measurements) {
  show_measurements_ = show_measurements;
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

std::pair<double, double> CameraExtractor::GetErrors() {
  if (measurement_complete_) {
    return std::make_pair(dist_err_, rot_err_);
  } else {
    throw std::runtime_error{"Measurement incomplete. Please run "
                             "ExtractMeasurement() before getting the "
                             "measurement error values."};
  }
  measurement_complete_ = false;
}

} // namespace vicon_calibration
