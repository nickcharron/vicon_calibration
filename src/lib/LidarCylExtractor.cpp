#include "vicon_calibration/LidarCylExtractor.h"

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

bool LidarCylExtractor::measurement_valid_;
bool LidarCylExtractor::measurement_failed_;

LidarCylExtractor::LidarCylExtractor() { ModifyICPConfig(); }

LidarCylExtractor::LidarCylExtractor(PointCloud::Ptr &template_cloud,
                                     PointCloud::Ptr &scan)
    : template_cloud_(template_cloud), scan_(scan) {}

void LidarCylExtractor::SetScanTransform(Eigen::Affine3d &T_LIDAR_SCAN) {
  if (!beam::IsTransformationMatrix(T_LIDAR_SCAN.matrix())) {
    throw std::runtime_error{
        "Passed in scan transform (scan to lidar) is invalid"};
  }
  T_LIDAR_SCAN_ = T_LIDAR_SCAN;
  pcl::transformPointCloud(*scan_, *scan_, T_LIDAR_SCAN_);
}

void LidarCylExtractor::SetShowTransformation(bool show_measurements) {
  if (show_measurements) {
    pcl_viewer_ = pcl::visualization::PCLVisualizer::Ptr(
        new pcl::visualization::PCLVisualizer("Cloud viewer"));
    pcl_viewer_->setBackgroundColor(0, 0, 0);
    pcl_viewer_->addCoordinateSystem(1.0);
    pcl_viewer_->initCameraParameters();
    pcl_viewer_->registerKeyboardCallback(ConfirmMeasurementKeyboardCallback,
                                          (void *)pcl_viewer_.get());
  }
  show_measurements_ = show_measurements;
}

std::pair<Eigen::Vector4d, bool> LidarCylExtractor::GetMeasurementInfo() {
  if (measurement_complete_) {
    measurement_complete_ = false;
    return std::make_pair(measurement_, measurement_valid_);
  } else {
    throw std::runtime_error{"Measurement not complete. Please run "
                             "LidarCylExtractor::ExtractCylinder() before "
                             "getting the measurement information."};
  }
}

void LidarCylExtractor::ExtractCylinder(Eigen::Affine3d &T_SCAN_TARGET_EST,
                                        int measurement_num) {
  if (template_cloud_ == nullptr) {
    throw std::runtime_error{"Template cloud is empty"};
  }

  if (!beam::IsTransformationMatrix(T_SCAN_TARGET_EST.matrix())) {
    throw std::runtime_error{"Passed in target to lidar transform is invalid"};
  }

  // Crop the scan before performing ICP registration
  auto cropped_cloud = CropPointCloud(T_SCAN_TARGET_EST);

  // Perform ICP Registration
  icp_.setInputSource(cropped_cloud);
  icp_.setInputTarget(template_cloud_);
  PointCloud::Ptr final_cloud(new PointCloud);
  icp_.align(*final_cloud, T_SCAN_TARGET_EST.inverse().matrix().cast<float>());

  if (!icp_.hasConverged()) {
    if (show_measurements_) {
      measurement_failed_ = true;
      std::cout << "ICP failed. Displaying cropped scan." << std::endl;
      auto coloured_cropped_cloud = ColourPointCloud(cropped_cloud, 255, 0, 0);
      AddColouredPointCloudToViewer(coloured_cropped_cloud,
                                    "coloured cropped cloud " +
                                        std::to_string(measurement_num));
      AddPointCloudToViewer(scan_, "scan " + std::to_string(measurement_num));
      ShowFailedMeasurement();
      pcl_viewer_->resetStoppedFlag();
    }

    measurement_valid_ = false;

    measurement_ = INVALID_MEASUREMENT;
  } else {
    Eigen::Affine3d T_SCAN_TARGET_OPT;
    T_SCAN_TARGET_OPT.matrix() =
        icp_.getFinalTransformation().inverse().cast<double>();

    // Get x,y,r,p data
    measurement_ = ExtractRelevantMeasurements(T_SCAN_TARGET_OPT);

    if (show_measurements_) {
      // Display clouds for testing
      // transform template cloud from target to lidar
      auto estimated_template_cloud =
          ColourPointCloud(template_cloud_, 0, 0, 255);
      pcl::transformPointCloud(*estimated_template_cloud,
                               *estimated_template_cloud, T_SCAN_TARGET_EST);
      AddColouredPointCloudToViewer(estimated_template_cloud,
                                    "estimated template cloud " +
                                        std::to_string(measurement_num));

      auto measured_template_cloud =
          ColourPointCloud(template_cloud_, 0, 255, 0);
      pcl::transformPointCloud(*measured_template_cloud,
                               *measured_template_cloud, T_SCAN_TARGET_OPT);
      AddColouredPointCloudToViewer(measured_template_cloud,
                                    "measured template cloud " +
                                        std::to_string(measurement_num));

      AddPointCloudToViewer(cropped_cloud,
                            "cropped scan " + std::to_string(measurement_num));

      ShowFinalTransformation();
      pcl_viewer_->resetStoppedFlag();

    } else {
      auto est_measurement = ExtractRelevantMeasurements(T_SCAN_TARGET_EST);
      Eigen::Vector2d dist_diff(measurement_(0) - est_measurement(0),
                                measurement_(1) - est_measurement(1));
      Eigen::Vector2d rot_diff(measurement_(2) - est_measurement(2),
                               measurement_(3) - est_measurement(3));

      double dist_err = std::round(dist_diff.norm() * 10000) / 10000;
      double rot_err = std::round(rot_diff.norm() * 10000) / 10000;
      if (dist_err >= dist_err_criteria_ || rot_err >= rot_err_criteria_) {
        // if error between estimated measurement and optimized measurement
        // is greater than 5 cm or than 30 deg, don't accept the measurement
        measurement_valid_ = false;
      } else {
        measurement_valid_ = true;
      }
    }
  }
  measurement_complete_ = true;
}

PointCloud::Ptr
LidarCylExtractor::CropPointCloud(Eigen::Affine3d &T_SCAN_TARGET_EST) {
  if (scan_ == nullptr) {
    throw std::runtime_error{"Scan is empty"};
  }

  if (radius_ == 0 || height_ == 0) {
    throw std::runtime_error{"Can't crop a cylinder with radius of " +
                             std::to_string(radius_) + " and height of " +
                             std::to_string(height_)};
  }

  if (threshold_ == 0) {
    std::cout << "WARNING: Using threshold of 0 for cropping" << std::endl;
  }

  Eigen::Vector3f min_vector(-threshold_, -radius_ - threshold_,
                             -radius_ - threshold_);
  Eigen::Vector3f max_vector(height_ + threshold_, radius_ + threshold_,
                             radius_ + threshold_);

  cropper_.SetMinVector(min_vector);
  cropper_.SetMaxVector(max_vector);
  Eigen::Affine3f T_TARGET_EST_SCAN = T_SCAN_TARGET_EST.inverse().cast<float>();
  cropper_.SetTransform(T_TARGET_EST_SCAN);

  PointCloud::Ptr cropped_cloud(new PointCloud);
  cropper_.Filter(*scan_, *cropped_cloud);

  return cropped_cloud;
}

Eigen::Vector4d
LidarCylExtractor::ExtractRelevantMeasurements(Eigen::Affine3d &T_SCAN_TARGET) {

  // Extract x,y,r,p values
  auto translation_vector = T_SCAN_TARGET.translation();
  auto rpy_vector = beam::RToLieAlgebra(T_SCAN_TARGET.rotation());
  Eigen::Vector4d measurement(translation_vector(0), translation_vector(1),
                              rpy_vector(0), rpy_vector(1));

  return measurement;
}

void LidarCylExtractor::AddColouredPointCloudToViewer(
    PointCloudColor::Ptr cloud, std::string cloud_name) {
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
      cloud);
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, cloud_name);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

void LidarCylExtractor::AddPointCloudToViewer(PointCloud::Ptr cloud,
                                              std::string cloud_name) {
  pcl_viewer_->addPointCloud<pcl::PointXYZ>(cloud, cloud_name);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

PointCloudColor::Ptr LidarCylExtractor::ColourPointCloud(PointCloud::Ptr &cloud,
                                                         int r, int g, int b) {
  PointCloudColor::Ptr coloured_cloud(new PointCloudColor);
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

void LidarCylExtractor::ShowFinalTransformation() {
  std::cout << "-------------------------------" << std::endl;
  std::cout << "Legend:" << std::endl;
  std::cout << "  White -> scan" << std::endl;
  std::cout << "  Blue  -> target initial guess" << std::endl;
  std::cout << "  Green -> target aligned" << std::endl;
  std::cout << "Accept measurement? [y/n]" << std::endl;
  while (!pcl_viewer_->wasStopped()) {
    pcl_viewer_->spinOnce(10);
    std::this_thread::sleep_for(10ms);
  }
}

void LidarCylExtractor::ShowFailedMeasurement() {
  std::cout << "-------------------------------" << std::endl;
  std::cout << "Legend:" << std::endl;
  std::cout << "  Red   -> cropped scan" << std::endl;
  std::cout << "  White -> original scan" << std::endl;
  std::cout << "Press [c] to continue with other measurements" << std::endl;
  while (!pcl_viewer_->wasStopped()) {
    pcl_viewer_->spinOnce(10);
    std::this_thread::sleep_for(10ms);
  }
}

void LidarCylExtractor::ConfirmMeasurementKeyboardCallback(
    const pcl::visualization::KeyboardEvent &event, void *viewer_void) {
  pcl::visualization::PCLVisualizer *viewer =
      static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);

  if (measurement_failed_) {
    if (event.getKeySym() == "c" && event.keyDown()) {
      std::cout << "Continuing with taking measurements" << std::endl;
      measurement_failed_ = false;
      viewer->removeAllPointClouds();
      viewer->close();
    }
  } else {
    if (event.getKeySym() == "y" && event.keyDown()) {
      std::cout << "Accepting measurement" << std::endl;
      measurement_valid_ = true;
      viewer->removeAllPointClouds();
      viewer->close();

    } else if (event.getKeySym() == "n" && event.keyDown()) {
      std::cout << "Rejecting measurement" << std::endl;
      measurement_valid_ = false;
      viewer->removeAllPointClouds();
      viewer->close();
    }
  }
}

void LidarCylExtractor::SetICPParameters(double t_eps, double fit_eps,
                                         double max_corr, int max_iter) {
  t_eps_ = t_eps;
  fit_eps_ = fit_eps;
  max_corr_ = max_corr;
  max_iter_ = max_iter;

  ModifyICPConfig();
}

void LidarCylExtractor::ModifyICPConfig() {
  icp_.setTransformationEpsilon(t_eps_);
  icp_.setEuclideanFitnessEpsilon(fit_eps_);
  icp_.setMaximumIterations(max_corr_);
  icp_.setMaxCorrespondenceDistance(max_iter_);
}

} // end namespace vicon_calibration
