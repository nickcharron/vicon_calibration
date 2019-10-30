#include "vicon_calibration/CylinderLidarExtractor.h"
#include <beam_filtering/CropBox.h>
#include <chrono>
#include <thread>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

void CylinderLidarExtractor::ExtractKeypoints(
    Eigen::Matrix4d &T_LIDAR_TARGET_EST, PointCloud::Ptr &cloud_in) {
  // initialize member variables
  scan_in_ = boost::make_shared<PointCloud>();
  scan_cropped_ = boost::make_shared<PointCloud>();
  scan_best_points_ = boost::make_shared<PointCloud>();
  pcl_viewer_ = boost::make_shared<pcl::visualization::PCLVisualizer>();
  scan_in_ = cloud_in;
  T_LIDAR_TARGET_EST_ = T_LIDAR_TARGET_EST;
  Eigen::Affine3d T_identity;
  T_identity.setIdentity();
  measurement_valid_ = true;
  measurement_complete_ = false;

  this->CheckInputs();
  this->CropScan();
  this->RegisterScan();
  this->SaveMeasurement();
  measurement_complete_ = true;
}

void CylinderLidarExtractor::CheckInputs() {
  if (target_params_.template_cloud == nullptr ||
      target_params_.template_cloud->size() == 0) {
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

void CylinderLidarExtractor::CropScan() {

  Eigen::Affine3f T_TARGET_EST_SCAN;
  T_TARGET_EST_SCAN.matrix() = T_LIDAR_TARGET_EST_.inverse().cast<float>();
  beam_filtering::CropBox cropper;
  Eigen::Vector3f min_vector, max_vector;
  max_vector = target_params_.crop_scan.cast<float>();
  min_vector = -max_vector;
  cropper.SetMinVector(min_vector);
  cropper.SetMaxVector(max_vector);
  cropper.SetRemoveOutsidePoints(true);
  cropper.SetTransform(T_TARGET_EST_SCAN);
  cropper.Filter(*scan_in_, *scan_cropped_);
}

void CylinderLidarExtractor::RegisterScan() {
  if (!test_registration_) {
    return;
  }
  measurement_failed_ = false;
  boost::shared_ptr<PointCloud> scan_registered;
  scan_registered = boost::make_shared<PointCloud>();
  IterativeClosestPointCustom<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setTransformationEpsilon(icp_transform_epsilon_);
  icp.setEuclideanFitnessEpsilon(icp_euclidean_epsilon_);
  icp.setMaximumIterations(icp_max_iterations_);
  icp.setMaxCorrespondenceDistance(icp_max_correspondence_dist_);
  if (crop_scan_) {
    icp.setInputSource(scan_cropped_);
  } else {
    icp.setInputSource(scan_in_);
  }
  icp.setInputTarget(target_params_.template_cloud);
  icp.align(*scan_registered, T_LIDAR_TARGET_EST_.inverse().cast<float>());

  if (!icp.hasConverged()) {
    measurement_failed_ = true;
    measurement_valid_ = false;
    measurement_complete_ = true;
    if (show_measurements_) {
      std::cout << "ICP failed. Displaying cropped scan." << std::endl;
      boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>
          scan_cropped_coloured;
      scan_cropped_coloured =
          boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
      Eigen::Affine3d T_identity;
      T_identity.setIdentity();
      scan_cropped_coloured = this->ColourPointCloud(scan_cropped_, 255, 0, 0);
      this->AddColouredPointCloudToViewer(scan_cropped_coloured,
                                          "Coloured Cropped Scan ", T_identity);
      this->AddPointCloudToViewer(scan_in_, "Input Scan", T_identity);
      this->ShowFailedMeasurement();
      pcl_viewer_->resetStoppedFlag();
    }
    return;
  }

  Eigen::Affine3d TA_LIDAR_TARGET_OPT, TA_LIDAR_TARGET_EST;
  TA_LIDAR_TARGET_OPT.matrix() =
      icp.getFinalTransformation().inverse().cast<double>();
  TA_LIDAR_TARGET_EST.matrix() = T_LIDAR_TARGET_EST_;

  Eigen::Vector2d dist_diff(
      TA_LIDAR_TARGET_OPT.matrix()(0, 3) - T_LIDAR_TARGET_EST_.matrix()(0, 3),
      TA_LIDAR_TARGET_OPT.matrix()(1, 3) - T_LIDAR_TARGET_EST_.matrix()(1, 3));
  double dist_err = std::round(dist_diff.norm() * 10000) / 10000;

  if (dist_err >= dist_acceptance_criteria_) {
    measurement_valid_ = false;
    std::cout << "-----------------------------\n"
              << "Measurement Invalid\n"
              << "Distance error norm: " << dist_err << "\n"
              << "Distance acceptance criteria: " << dist_acceptance_criteria_
              << "\n";
  } else {
    measurement_valid_ = true;
    std::cout << "-----------------------------\n"
              << "Measurement Valid\n"
              << "Distance error norm: " << dist_err << "\n"
              << "Distance acceptance criteria: " << dist_acceptance_criteria_
              << "\n";
  }

  if (show_measurements_) {
    // output error information
    if (!measurement_valid_) {
      std::cout << "-----------------------------\n"
                << "Measurement Invalid\n"
                << "Distance error norm: " << dist_err << "\n"
                << "Distance acceptance criteria: " << dist_acceptance_criteria_
                << "\n";
    } else {
      std::cout << "-----------------------------\n"
                << "Measurement Valid\n"
                << "Distance error norm: " << dist_err << "\n"
                << "Distance acceptance criteria: " << dist_acceptance_criteria_
                << "\n";
    }
    // Display clouds for testing
    // transform template cloud from target to lidar
    PointCloudColor::Ptr estimated_template_cloud =
        boost::make_shared<PointCloudColor>();
    estimated_template_cloud =
        this->ColourPointCloud(target_params_.template_cloud, 0, 0, 255);
    pcl::transformPointCloud(*estimated_template_cloud,
                             *estimated_template_cloud, TA_LIDAR_TARGET_EST);
    this->AddColouredPointCloudToViewer(estimated_template_cloud,
                                        "estimated template cloud ",
                                        TA_LIDAR_TARGET_EST);

    PointCloudColor::Ptr measured_template_cloud =
        boost::make_shared<PointCloudColor>();
    measured_template_cloud =
        this->ColourPointCloud(target_params_.template_cloud, 0, 255, 0);
    pcl::transformPointCloud(*measured_template_cloud, *measured_template_cloud,
                             TA_LIDAR_TARGET_OPT);
    this->AddColouredPointCloudToViewer(measured_template_cloud,
                                        "measured template cloud",
                                        TA_LIDAR_TARGET_OPT);
    Eigen::Affine3d T_identity;
    T_identity.setIdentity();
    this->AddPointCloudToViewer(scan_cropped_, "cropped scan", T_identity);
    this->ShowFinalTransformation();
  }

  pcl::CorrespondencesPtr correspondences = icp.getCorrespondencesPtr();
  for (uint32_t i = 0; i < correspondences->size(); i++) {
    pcl::Correspondence corr_i = (*correspondences)[i];
    int source_index_i = corr_i.index_query;
    if(corr_i.distance < max_keypoint_distance_){
      pcl::PointXYZ point;
      if (crop_scan_) {
        point = scan_cropped_->points[source_index_i];
      } else {
        point = scan_in_->points[source_index_i];
      }
      scan_best_points_->points.push_back(point);
    }
  }
}

PointCloudColor::Ptr
CylinderLidarExtractor::ColourPointCloud(PointCloud::Ptr &cloud, int r, int g,
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

void CylinderLidarExtractor::AddColouredPointCloudToViewer(
    PointCloudColor::Ptr cloud, std::string cloud_name, Eigen::Affine3d &T) {
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
      cloud);
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, cloud_name);
  pcl_viewer_->addCoordinateSystem(1, T.cast<float>(), cloud_name + "frame");
  pcl::PointXYZ point;
  point.x = T.translation()(0);
  point.y = T.translation()(1);
  point.z = T.translation()(2);
  pcl_viewer_->addText3D(cloud_name + " ", point, 0.05, 0.05, 0.05);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

void CylinderLidarExtractor::AddPointCloudToViewer(PointCloud::Ptr cloud,
                                                   std::string cloud_name,
                                                   Eigen::Affine3d &T) {
  pcl_viewer_->addPointCloud<pcl::PointXYZ>(cloud, cloud_name);
  pcl_viewer_->addCoordinateSystem(1, T.cast<float>(), cloud_name + "frame");
  pcl::PointXYZ point;
  point.x = T.translation()(0);
  point.y = T.translation()(1);
  point.z = T.translation()(2);
  pcl_viewer_->addText3D(cloud_name + " ", point, 0.05, 0.05, 0.05);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

void CylinderLidarExtractor::ShowFailedMeasurement() {
  std::cout << "\nViewer Legend:\n"
            << "  Red   -> cropped scan\n"
            << "  White -> original scan\n"
            << "Press [c] to continue with other measurements\n";
  while (!pcl_viewer_->wasStopped() && !close_viewer_) {
    pcl_viewer_->spinOnce(10);
    pcl_viewer_->registerKeyboardCallback(
        &CylinderLidarExtractor::ConfirmMeasurementKeyboardCallback, *this);
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

void CylinderLidarExtractor::ConfirmMeasurementKeyboardCallback(
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

void CylinderLidarExtractor::ShowFinalTransformation() {
  std::cout << "\nViewer Legend:\n"
            << "  White -> scan\n"
            << "  Blue  -> target initial guess\n"
            << "  Green -> target aligned\n"
            << "Accept measurement? [y/n]\n";
  while (!pcl_viewer_->wasStopped() && !close_viewer_) {
    pcl_viewer_->spinOnce(10);
    pcl_viewer_->registerKeyboardCallback(
        &CylinderLidarExtractor::ConfirmMeasurementKeyboardCallback, *this);
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

void CylinderLidarExtractor::SaveMeasurement() {
  if (!measurement_valid_) {
    return;
  }
  if (!crop_scan_) {
    keypoints_measured_ = scan_in_;
  } else {
    keypoints_measured_ = scan_best_points_;
  }
}

} // namespace vicon_calibration
