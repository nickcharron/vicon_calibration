#include "vicon_calibration/measurement_extractors/LidarExtractor.h"
#include <beam_filtering/CropBox.h>
#include <boost/make_shared.hpp>
#include <chrono>
#include <thread>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

LidarExtractor::LidarExtractor() {
  keypoints_measured_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
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

void LidarExtractor::SetShowMeasurements(const bool &show_measurements) {
  show_measurements_ = show_measurements;
}

bool LidarExtractor::GetShowMeasurements() { return show_measurements_; }

bool LidarExtractor::GetMeasurementValid() {
  if (!measurement_complete_) {
    throw std::invalid_argument{
        "Cannot retrieve measurement, please run ExtractKeypoints before "
        "attempting to retrieve measurement."};
  }
  return measurement_valid_;
}

// TODO: move this to json tools object
void LidarExtractor::LoadConfig() {
  std::string config_path =
      utils::GetFilePathConfig("LidarExtractorConfig.json");
  nlohmann::json J;
  std::ifstream file(config_path);
  file >> J;
  crop_scan_ = J.at("crop_scan");
  max_keypoint_distance_ = J.at("max_keypoint_distance");
  dist_acceptance_criteria_ = J.at("dist_acceptance_criteria");
  icp_transform_epsilon_ = J.at("icp_transform_epsilon");
  icp_euclidean_epsilon_ = J.at("icp_euclidean_epsilon");
  icp_max_iterations_ = J.at("icp_max_iterations");
  icp_max_correspondence_dist_ = J.at("icp_max_correspondence_dist");
}

void LidarExtractor::ProcessMeasurement(
    const Eigen::Matrix4d &T_LIDAR_TARGET_EST, const PointCloud::Ptr &cloud_in) {
  // initialize member variables
  this->LoadConfig();
  scan_in_ = boost::make_shared<PointCloud>();
  scan_cropped_ = boost::make_shared<PointCloud>();
  if (show_measurements_) {
    pcl_viewer_ = boost::make_shared<pcl::visualization::PCLVisualizer>();
  }
  scan_in_ = cloud_in;
  T_LIDAR_TARGET_EST_ = T_LIDAR_TARGET_EST;
  measurement_valid_ = true;
  measurement_complete_ = false;

  this->CheckInputs();
  this->CropScan();
  this->GetKeypoints();
  measurement_complete_ = true;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr LidarExtractor::GetMeasurement() {
  if (!measurement_complete_) {
    throw std::invalid_argument{
        "Cannot retrieve measurement, please run ExtractKeypoints before "
        "attempting to retrieve measurement."};
  }
  return keypoints_measured_;
}

void LidarExtractor::CheckInputs() {
  if (target_params_->template_cloud == nullptr ||
      target_params_->template_cloud->size() == 0) {
    throw std::runtime_error{"Template cloud is empty"};
  }

  if (scan_in_ == nullptr || scan_in_->size() == 0) {
    throw std::runtime_error{"Input scan is empty"};
  }

  if (!utils::IsTransformationMatrix(T_LIDAR_TARGET_EST_)) {
    throw std::runtime_error{
        "Estimated transform from target to lidar is invalid"};
  }
}

void LidarExtractor::CropScan() {

  Eigen::Affine3f T_TARGET_EST_SCAN;
  T_TARGET_EST_SCAN.matrix() = T_LIDAR_TARGET_EST_.inverse().cast<float>();
  beam_filtering::CropBox cropper;
  Eigen::Vector3f min_vector, max_vector;
  max_vector = target_params_->crop_scan.cast<float>();
  min_vector = -max_vector;
  cropper.SetMinVector(min_vector);
  cropper.SetMaxVector(max_vector);
  cropper.SetRemoveOutsidePoints(true);
  cropper.SetTransform(T_TARGET_EST_SCAN);
  cropper.Filter(*scan_in_, *scan_cropped_);
}

void LidarExtractor::AddColouredPointCloudToViewer(
    const PointCloudColor::Ptr &cloud, const std::string &cloud_name,
    boost::optional<Eigen::MatrixXd &> T = boost::none, int point_size) {
  if (cloud_name == "blue_cloud") {
    blue_cloud_ = cloud;
    blue_cloud_on_ = true;
  } else if (cloud_name == "green_cloud") {
    green_cloud_ = cloud;
    green_cloud_on_ = true;
  }
  if (T) {
    Eigen::Affine3f TA;
    TA.matrix() = (*T).cast<float>();
    pcl_viewer_->addCoordinateSystem(1, TA, cloud_name + "frame");
    pcl::PointXYZ point;
    point.x = (*T)(0, 3);
    point.y = (*T)(1, 3);
    point.z = (*T)(2, 3);
    pcl_viewer_->addText3D(cloud_name + " ", point, 0.05, 0.05, 0.05);
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
      cloud);
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, cloud_name);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, cloud_name);
}

void LidarExtractor::AddPointCloudToViewer(const PointCloud::Ptr &cloud,
                                           const std::string &cloud_name,
                                           const Eigen::Matrix4d &T,
                                           int point_size) {
  if (cloud_name == "white_cloud") {
    white_cloud_ = cloud;
    white_cloud_on_ = true;
  }
  Eigen::Affine3f TA;
  TA.matrix() = T.cast<float>();
  pcl_viewer_->addPointCloud<pcl::PointXYZ>(cloud, cloud_name);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, cloud_name);
  pcl_viewer_->addCoordinateSystem(1, TA, cloud_name + "frame");
  pcl::PointXYZ point;
  point.x = T(0, 3);
  point.y = T(1, 3);
  point.z = T(2, 3);
  pcl_viewer_->addText3D(cloud_name + " ", point, 0.05, 0.05, 0.05);
}

void LidarExtractor::ConfirmMeasurementKeyboardCallback(
    const pcl::visualization::KeyboardEvent &event, void *viewer_void) {
  // check if key has been down for two consecutive spins
  if(viewer_key_down_ && event.keyDown()) {
    return;
  } else if (viewer_key_down_ && !event.keyDown()){
    viewer_key_down_ = false;
    return;
  } else if (!viewer_key_down_ && event.keyDown()){
    viewer_key_down_ = true;
  } else if (!viewer_key_down_ && !event.keyDown()) {
    return;
  }
  if (measurement_failed_) {
    if (event.getKeySym() == "c") {
      measurement_failed_ = false;
      close_viewer_ = true;
    }
  } else {
    if (event.getKeySym() == "y") {
      measurement_valid_ = true;
      close_viewer_ = true;
    } else if (event.getKeySym() == "n") {
      measurement_valid_ = false;
      close_viewer_ = true;
    } else if (event.getKeySym() == "c") {
      close_viewer_ = true;
    } else if (event.getKeySym() == "s") {
      this->SetShowMeasurements(false);
      close_viewer_ = true;
    } else if (event.getKeySym() == "KP_1") {
      std::string cloud_id = "white_cloud";
      if (white_cloud_on_) {
        white_cloud_on_ = false;
        pcl_viewer_->removePointCloud(cloud_id);
      } else {
        white_cloud_on_ = true;
        pcl_viewer_->addPointCloud(white_cloud_, cloud_id);
      }
    } else if (event.getKeySym() == "KP_2") {
      std::string cloud_id = "blue_cloud";
      if (blue_cloud_on_) {
        blue_cloud_on_ = false;
        pcl_viewer_->removePointCloud(cloud_id);
      } else {
        blue_cloud_on_ = true;
        pcl_viewer_->addPointCloud(blue_cloud_, cloud_id);
      }
    } else if (event.getKeySym() == "KP_3") {
      std::string cloud_id = "green_cloud";
      if (green_cloud_on_) {
        green_cloud_on_ = false;
        pcl_viewer_->removePointCloud(cloud_id);
      } else {
        green_cloud_on_ = true;
        pcl_viewer_->addPointCloud(green_cloud_, cloud_id);
      }
    }
  }
}

void LidarExtractor::ShowFailedMeasurement() {
  // calculate cropbox cube
  Eigen::Affine3f T;
  T.matrix() = T_LIDAR_TARGET_EST_.cast<float>();
  Eigen::Vector3f translation = T.translation();
  Eigen::Quaternionf rotation(T.rotation());
  double width = target_params_->crop_scan[0];
  double height = target_params_->crop_scan[1];
  double depth = target_params_->crop_scan[2];
  pcl_viewer_->addCube(translation, rotation, width, height, depth);
  pcl_viewer_->setRepresentationToWireframeForAllActors();
  std::cout << "\nViewer Legend:\n"
            << "  Red   -> cropped scan\n"
            << "  White -> original scan\n"
            << "Press [c] to continue with other measurements\n"
            << "Press [s] to stop showing future measurements\n";
  while (!pcl_viewer_->wasStopped() && !close_viewer_) {
    pcl_viewer_->spinOnce(10);
    pcl_viewer_->registerKeyboardCallback(
        &LidarExtractor::ConfirmMeasurementKeyboardCallback, *this);
    std::this_thread::sleep_for(10ms);
  }
  close_viewer_ = false;
  pcl_viewer_->removeAllPointClouds();
  pcl_viewer_->close();
  pcl_viewer_->resetStoppedFlag();
  if (measurement_failed_) {
    std::cout << "Continuing with taking measurements" << std::endl;
  } else if (measurement_valid_) {
    std::cout << "Accepting measurement" << std::endl;
  } else {
    std::cout << "Rejecting measurement" << std::endl;
  }
}

void LidarExtractor::ShowFinalTransformation() {
  std::cout << "\nViewer Legend:\n"
            << "  White -> scan (press 1 to toggle on/off)\n"
            << "  Blue  -> target initial guess (press 2 to toggle on/off)\n"
            << "  Green -> target aligned (press 3 to toggle on/off)\n"
            << "Accept measurement? [y/n]\n"
            << "Press [c] to accept default\n"
            << "Press [s] to stop showing future measurements\n";
  while (!pcl_viewer_->wasStopped() && !close_viewer_) {
    pcl_viewer_->spinOnce(10);
    pcl_viewer_->registerKeyboardCallback(
        &LidarExtractor::ConfirmMeasurementKeyboardCallback, *this);
    std::this_thread::sleep_for(10ms);
  }
  close_viewer_ = false;
  pcl_viewer_->removeAllPointClouds();
  pcl_viewer_->removeAllCoordinateSystems();
  pcl_viewer_->removeAllShapes();
  pcl_viewer_->close();
  pcl_viewer_->resetStoppedFlag();
  if (measurement_failed_) {
    std::cout << "Continuing with taking measurements" << std::endl;
  } else if (measurement_valid_) {
    std::cout << "Accepting measurement" << std::endl;
  } else {
    std::cout << "Rejecting measurement" << std::endl;
  }
}

} // namespace vicon_calibration
