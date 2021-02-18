#include <vicon_calibration/measurement_extractors/LidarExtractor.h>

#include <pcl/io/pcd_io.h>

#include <beam_filtering/CropBox.h>

namespace vicon_calibration {

LidarExtractor::LidarExtractor(
    const std::shared_ptr<vicon_calibration::LidarParams>& lidar_params,
    const std::shared_ptr<vicon_calibration::TargetParams>& target_params,
    const bool& show_measurements, const std::shared_ptr<Visualizer> pcl_viewer)
    : lidar_params_(lidar_params),
      target_params_(target_params),
      pcl_viewer_(pcl_viewer),
      show_measurements_(show_measurements) {
  keypoints_measured_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  LOG_ERROR("Initialized LidarExtractor");
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
    const Eigen::Matrix4d& T_LIDAR_TARGET_EST, const PointCloud::Ptr& cloud_in,
    bool& show_measurements) {
  LOG_ERROR("ProcessMeasurement called");
  scan_in_ = cloud_in;
  T_LIDAR_TARGET_EST_ = T_LIDAR_TARGET_EST;
  LOG_ERROR("Setting up variables");
  this->SetupVariables();
  LOG_ERROR("Checking inputs");
  this->CheckInputs();
  LOG_ERROR("Isolating points");
  this->IsolatePoints();
  LOG_ERROR("Getting keypoints");
  this->GetKeypoints();          // implemented in derived class
  LOG_ERROR("Checking measurement valid");
  this->CheckMeasurementValid(); // implemented in derived class
  LOG_ERROR("Getting user input");
  this->GetUserInput();
  LOG_ERROR("Outputting scans");
  this->OutputScans();
  LOG_ERROR("Setting flags");
  measurement_complete_ = true;
  show_measurements = show_measurements_;
  LOG_ERROR("Done processing measurement");
}

void LidarExtractor::SetupVariables() {
  if (icp_enable_debug_) {
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
  }
  measurement_valid_ = true;
  measurement_complete_ = false;
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

void LidarExtractor::IsolatePoints() {
  target_isolator_ = IsolateTargetPoints();
  target_isolator_.SetScan(scan_in_);
  target_isolator_.SetTransformEstimate(
      utils::InvertTransform(T_LIDAR_TARGET_EST_));
  target_isolator_.SetTargetParams(target_params_);
  target_isolator_.SetLidarParams(lidar_params_);
  scan_isolated_ = target_isolator_.GetPoints();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr LidarExtractor::GetMeasurement() {
  if (!measurement_complete_) {
    throw std::invalid_argument{
        "Cannot retrieve measurement, please run ExtractKeypoints before "
        "attempting to retrieve measurement."};
  }
  return keypoints_measured_;
}

void LidarExtractor::GetUserInput() {
  std::cout << "TEST: show_measurements_: " << show_measurements_ << "\n";
  if (!show_measurements_) { return; }

  estimated_template_cloud_ = boost::make_shared<PointCloud>();
  pcl::transformPointCloud(*target_params_->template_cloud,
                           *estimated_template_cloud_,
                           T_LIDAR_TARGET_EST_.cast<float>());
  
  pcl_viewer_->ClearPointClouds();

  if (measurement_valid_) {
    std::cout << "Measurement Valid\n";
    measured_template_cloud_ = boost::make_shared<PointCloud>();
    pcl::transformPointCloud(*target_params_->template_cloud,
                             *measured_template_cloud_,
                             T_LIDAR_TARGET_OPT_.cast<float>());

    // add estimated template cloud
    pcl_viewer_->AddPointCloudToViewer(estimated_template_cloud_, "blue_cloud",
                                       Eigen::Vector3i(0, 0, 255), 5,
                                       T_LIDAR_TARGET_EST_.cast<float>());

    // add measured template cloud
    pcl_viewer_->AddPointCloudToViewer(measured_template_cloud_, "green_cloud",
                                       Eigen::Vector3i(0, 255, 0), 5,
                                       T_LIDAR_TARGET_OPT_.cast<float>());

    // add the isolated scan
    pcl_viewer_->AddPointCloudToViewer(scan_isolated_, "white_cloud",
                                       Eigen::Vector3i(255, 255, 255), 5);

    // add keypoints if discrete keypoints are specified
    if (target_params_->keypoints_lidar.size() > 0) {
      std::cout << "Showing measured keypoints in yellow.\n";
      pcl_viewer_->AddPointCloudToViewer(keypoints_measured_, "keypoints",
                                         Eigen::Vector3i(255, 255, 0), 5);
    }

    std::cout << "\nViewer Legend:\n"
              << "  White -> scan (press 1 to toggle on/off)\n"
              << "  Blue  -> target initial guess (press 2 to toggle on/off)\n"
              << "  Green -> target aligned (press 3 to toggle on/off)\n"
              << "Accept measurement? [y/n]\n"
              << "Press [c] to accept default\n"
              << "Press [s] to stop showing future measurements\n";

    measurement_valid_ = pcl_viewer_->DisplayClouds();
    if (measurement_valid_) {
      std::cout << "Accepting measurement" << std::endl;
    } else {
      std::cout << "Rejecting measurement" << std::endl;
    }

  } else {
    std::cout << "Measurement Invalid\n";

    pcl_viewer_->AddPointCloudToViewer(scan_isolated_, "red_cloud",
                                       Eigen::Vector3i(255, 0, 0), 3);

    std::cout << "\nViewer Legend:\n"
              << "  Red   -> isolated scan\n"
              << "  White -> original scan\n"
              << "Accept measurement? [y/n]\n"
              << "Press [c] to continue with other measurements\n"
              << "Press [s] to stop showing future measurements\n";
    measurement_valid_ = pcl_viewer_->DisplayClouds();
    if (measurement_valid_) {
      std::cout << "Accepting measurement" << std::endl;
    } else {
      std::cout << "Rejecting measurement" << std::endl;
    }
  }
}

void LidarExtractor::OutputScans() {
  if (!output_scans_) { return; }
  std::string date_and_time =
      utils::ConvertTimeToDate(std::chrono::system_clock::now());
  if (!boost::filesystem::exists(output_directory_)) {
    boost::filesystem::create_directory(output_directory_);
  }
  std::string save_dir = output_directory_ + date_and_time + "/";
  boost::filesystem::create_directory(save_dir);
  pcl::PCDWriter writer;
  // crop scan in
  PointCloud scan_in2;
  beam_filtering::CropBox cropper;
  Eigen::Vector3f min_vector(-7, -7, -7), max_vector(7, 7, 7);
  cropper.SetMinVector(min_vector);
  cropper.SetMaxVector(max_vector);
  cropper.SetRemoveOutsidePoints(true);
  cropper.Filter(*scan_in_, scan_in2);
  if (scan_in_ != nullptr && scan_in2.size() > 0) {
    writer.write(save_dir + "scan_in.pcd", scan_in2);
  }
  if (scan_isolated_ != nullptr && scan_isolated_->size() > 0) {
    writer.write(save_dir + "scan_isolated.pcd", *scan_isolated_);
  }
  if (keypoints_measured_ != nullptr && keypoints_measured_->size() > 0) {
    writer.write(save_dir + "keypoints_measured.pcd", *keypoints_measured_);
  }
  PointCloud::Ptr scan_cropped = target_isolator_.GetCroppedScan();
  if (scan_cropped != nullptr && scan_cropped->size() > 0) {
    writer.write(save_dir + "scan_cropped.pcd", *scan_cropped);
  }
  if (estimated_template_cloud_ != nullptr &&
      estimated_template_cloud_->size() > 0) {
    writer.write(save_dir + "estimated_template_cloud.pcd",
                 *estimated_template_cloud_);
  }
  if (measured_template_cloud_ != nullptr &&
      measured_template_cloud_->size() > 0) {
    writer.write(save_dir + "measured_template_cloud.pcd",
                 *measured_template_cloud_);
  }
  std::vector<PointCloud::Ptr> clusters = target_isolator_.GetClusters();
  for (int i = 0; i < clusters.size(); i++) {
    writer.write(save_dir + "cluster" + std::to_string(i) + ".pcd",
                 *clusters[i]);
  }
  LOG_INFO("Saved %d clusters.", clusters.size());
}

} // namespace vicon_calibration
