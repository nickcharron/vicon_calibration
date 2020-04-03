#include "vicon_calibration/measurement_extractors/IsolateTargetPoints.h"

namespace vicon_calibration {

// TODO: move this to json tools object
void IsolateTargetPoints::LoadConfig() {
  std::string config_path =
      utils::GetFilePathConfig("IsolateTargetPointsConfig.json");
  nlohmann::json J;
  std::ifstream file(config_path);
  file >> J;
  crop_scan_ = J.at("crop_scan");
  isolator_volume_weight_ = J.at("isolator_volume_weight");
  isolator_distance_weight_ = J.at("isolator_distance_weight");
}

void IsolateTargetPoints::SetTransformEstimate(
    const Eigen::Matrix4d &T_TARGET_LIDAR) {
  T_TARGET_LIDAR_ = T_TARGET_LIDAR;
}

void IsolateTargetPoints::SetScan(const PointCloud::Ptr &scan_in) {
  scan_in_ = scan_in;
}

void IsolateTargetPoints::SetTargetParams(const TargetParams &target_params) {
  target_params_ = target_params;
}

PointCloud::Ptr IsolateTargetPoints::GetPoints() {
  if (!CheckInputs()) {
    throw std::runtime_error{"Invalid inputs to IsolateTargetPoints."};
  }
  PointCloud::Ptr scan_isolated_ = boost::make_shared<PointCloud>();
  // TODO: implement these
  // this->ClusterPoints();
  // this->GetTargetCluster();
  return scan_isolated_;
}

bool IsolateTargetPoints::CheckInputs() {
  if (scan_in_ == nullptr) {
    LOG_ERROR("Input scan not set.");
    return false;
  } else if (target_params_ == NULL) {
    LOG_ERROR("Target params not set.");
    return false;
  } else if (T_TARGET_LIDAR_ == NULL) {
    LOG_ERROR("Transformation estimate not set.");
    return false;
  } else {
    return true;
  }
}

void IsolateTargetPoints::CropScan() {
  scan_cropped_ = boost::make_shared<PointCloud>();
  beam_filtering::CropBox cropper;
  Eigen::Vector3f min_vector, max_vector;
  max_vector = target_params_->crop_scan.cast<float>();
  min_vector = -max_vector;
  cropper.SetMinVector(min_vector);
  cropper.SetMaxVector(max_vector);
  cropper.SetRemoveOutsidePoints(true);
  cropper.SetTransform(T_TARGET_LIDAR_.cast<float>());
  cropper.Filter(*scan_in_, *scan_cropped_);
}

} // namespace vicon_calibration
