#include "vicon_calibration/measurement_extractors/DiamondLidarExtractor.h"
#include <beam_filtering/CropBox.h>
#include <chrono>
#include <thread>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

// edited - this should be changed to GetMeasurement() and be implemented
// in base class. Edits are shown here
void DiamondLidarExtractor::ExtractKeypoints(
    Eigen::Matrix4d &T_LIDAR_TARGET_EST, PointCloud::Ptr &cloud_in) {
  // initialize member variables
  scan_in_ = boost::make_shared<PointCloud>();
  scan_cropped_ = boost::make_shared<PointCloud>();
  // add this somewhere specific to DiamondLidarExtractor:
  // scan_best_points_ = boost::make_shared<PointCloud>();
  if (show_measurements_) {
    pcl_viewer_ = boost::make_shared<pcl::visualization::PCLVisualizer>();
  }
  scan_in_ = cloud_in;
  T_LIDAR_TARGET_EST_ = T_LIDAR_TARGET_EST;
  Eigen::Affine3d T_identity;
  T_identity.setIdentity();
  measurement_valid_ = true;
  measurement_complete_ = false;

  this->CheckInputs();
  this->CropScan();
  this->RegisterScan();
  // this->SaveMeasurement();
  measurement_complete_ = true;
}

// unedited
void DiamondLidarExtractor::CheckInputs() {
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

// unedited
void DiamondLidarExtractor::CropScan() {

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

// edited - this should be changed to GetKeypoints()
void DiamondLidarExtractor::RegisterScan() {
  // if (!test_registration_) {
  //   return;
  // }
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
  icp.setInputTarget(target_params_->template_cloud);
  icp.align(*scan_registered, utils::InvertTransform(T_LIDAR_TARGET_EST_).cast<float>());

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
      Eigen::MatrixXd T_identity = Eigen::MatrixXd(4,4);
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

  Eigen::MatrixXd T_LIDAR_TARGET_OPT = Eigen::MatrixXd(4,4);
  T_LIDAR_TARGET_OPT =
      utils::InvertTransform(icp.getFinalTransformation().cast<double>());
  measurement_valid_ = true;

  // TODO: create error metric that depends on distance to sensor

  // transform keypoints from json using opt. transform and store
  Eigen::Vector4d keypoint_homo;
  Eigen::Vector4d keypoint_trans_homo;
  Eigen::Vector3d keypoint_trans;
  keypoints_measured_->clear();
  for (Eigen::Vector3d keypoint : target_params_->keypoints_lidar){
    keypoint_homo = utils::PointToHomoPoint(keypoint);
    keypoint_trans_homo = T_LIDAR_TARGET_OPT * keypoint_homo;
    keypoint_trans = utils::HomoPointToPoint(keypoint_trans_homo);
    keypoints_measured_->push_back(utils::EigenPointToPCL(keypoint_trans));
  }

  if (show_measurements_) {
    if (!measurement_valid_) {
      std::cout << "Measurement Invalid\n"
                << "Showing detected keypoints in Yellow\n";
    } else {
      std::cout << "Measurement Valid\n"
                << "Showing detected keypoints in Yellow\n";
    }

    // add estimated template cloud
    PointCloudColor::Ptr estimated_template_cloud =
        this->ColourPointCloud(target_params_->template_cloud, 0, 0, 255);
    pcl::transformPointCloud(*estimated_template_cloud,
                             *estimated_template_cloud,
                             T_LIDAR_TARGET_EST_.cast<float>());
    this->AddColouredPointCloudToViewer(estimated_template_cloud,
                                        "estimated template cloud ",
                                        T_LIDAR_TARGET_EST_);

    // add measured template cloud
    PointCloudColor::Ptr measured_template_cloud =
        this->ColourPointCloud(target_params_->template_cloud, 0, 255, 0);
    pcl::transformPointCloud(*measured_template_cloud, *measured_template_cloud,
                             T_LIDAR_TARGET_OPT.cast<float>());
    this->AddColouredPointCloudToViewer(
        measured_template_cloud, "measured template cloud", T_LIDAR_TARGET_OPT);

    // add keypoints
    PointCloudColor::Ptr measured_keypoints =
        this->ColourPointCloud(keypoints_measured_, 255, 255, 0);
    this->AddColouredPointCloudToViewer(
        measured_keypoints, "measured keypoints", T_LIDAR_TARGET_OPT);

    Eigen::Matrix4d T_identity;
    T_identity.setIdentity();
    this->AddPointCloudToViewer(scan_cropped_, "cropped scan", T_identity);
    this->ShowFinalTransformation();
  }
}

// unedited
PointCloudColor::Ptr
DiamondLidarExtractor::ColourPointCloud(PointCloud::Ptr &cloud, int r, int g,
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

// edited but should work for both
void DiamondLidarExtractor::AddColouredPointCloudToViewer(
    PointCloudColor::Ptr cloud, const std::string &cloud_name,
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

// unedited
void DiamondLidarExtractor::AddPointCloudToViewer(
    PointCloud::Ptr cloud, const std::string &cloud_name,
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

// unedited
void DiamondLidarExtractor::ShowFailedMeasurement() {
  std::cout << "\nViewer Legend:\n"
            << "  Red   -> cropped scan\n"
            << "  White -> original scan\n"
            << "Press [c] to continue with other measurements\n";
  while (!pcl_viewer_->wasStopped() && !close_viewer_) {
    pcl_viewer_->spinOnce(10);
    pcl_viewer_->registerKeyboardCallback(
        &DiamondLidarExtractor::ConfirmMeasurementKeyboardCallback, *this);
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

// unedited
void DiamondLidarExtractor::ConfirmMeasurementKeyboardCallback(
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

// unedited
void DiamondLidarExtractor::ShowFinalTransformation() {
  std::cout << "\nViewer Legend:\n"
            << "  White -> scan\n"
            << "  Blue  -> target initial guess\n"
            << "  Green -> target aligned\n"
            << "Accept measurement? [y/n]\n";
  while (!pcl_viewer_->wasStopped() && !close_viewer_) {
    pcl_viewer_->spinOnce(10);
    pcl_viewer_->registerKeyboardCallback(
        &DiamondLidarExtractor::ConfirmMeasurementKeyboardCallback, *this);
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

// removed - this can be moved to ExtractKeypoints
// void DiamondLidarExtractor::SaveMeasurement()

} // namespace vicon_calibration
