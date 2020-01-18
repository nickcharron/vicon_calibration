#include "vicon_calibration/measurement_extractors/LidarExtractor.h"
#include <boost/make_shared.hpp>
#include <beam_filtering/CropBox.h>
#include <chrono>
#include <thread>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

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

void LidarExtractor::ProcessMeasurement(
    const Eigen::Matrix4d &T_LIDAR_TARGET_EST, PointCloud::Ptr &cloud_in) {
  // initialize member variables
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

PointCloudColor::Ptr
LidarExtractor::ColourPointCloud(PointCloud::Ptr &cloud, int r, int g,
                                         int b) {
  PointCloudColor::Ptr coloured_cloud;
  coloured_cloud = boost::make_shared<PointCloudColor>();
  uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                  static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
  pcl::PointXYZRGB point;
  for (PointCloud::iterator it = cloud->begin(); it != cloud->end(); ++it) {
    point.x = it->x;
    point.y = it->y;
    point.z = it->z;
    point.rgb = *reinterpret_cast<float *>(&rgb);
    coloured_cloud->push_back(point);
  }
  return coloured_cloud;
}

void LidarExtractor::AddColouredPointCloudToViewer(
    PointCloudColor::Ptr &cloud, const std::string &cloud_name,
    boost::optional<Eigen::MatrixXd &> T = boost::none) {
  if(T){
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
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

void LidarExtractor::AddPointCloudToViewer(
    PointCloud::Ptr &cloud, const std::string &cloud_name,
    const Eigen::Matrix4d &T) {
  Eigen::Affine3f TA;
  TA.matrix() = T.cast<float>();
  pcl_viewer_->addPointCloud<pcl::PointXYZ>(cloud, cloud_name);
  pcl_viewer_->addCoordinateSystem(1, TA, cloud_name + "frame");
  pcl::PointXYZ point;
  point.x = T(0, 3);
  point.y = T(1, 3);
  point.z = T(2, 3);
  pcl_viewer_->addText3D(cloud_name + " ", point, 0.05, 0.05, 0.05);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

void LidarExtractor::ConfirmMeasurementKeyboardCallback(
    const pcl::visualization::KeyboardEvent &event, void *viewer_void) {

  if (measurement_failed_) {
    if (event.getKeySym() == "c" && event.keyDown()) {
      measurement_failed_ = false;
      close_viewer_ = true;
    }
  } else {
    if (event.getKeySym() == "y" && event.keyDown()) {
      measurement_valid_ = true;
      close_viewer_ = true;
    } else if (event.getKeySym() == "n" && event.keyDown()) {
      measurement_valid_ = false;
      close_viewer_ = true;
    }
  }
}

void LidarExtractor::ShowFailedMeasurement() {
  std::cout << "\nViewer Legend:\n"
            << "  Red   -> cropped scan\n"
            << "  White -> original scan\n"
            << "Press [c] to continue with other measurements\n";
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
            << "  White -> scan\n"
            << "  Blue  -> target initial guess\n"
            << "  Green -> target aligned\n"
            << "Accept measurement? [y/n]\n";
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
