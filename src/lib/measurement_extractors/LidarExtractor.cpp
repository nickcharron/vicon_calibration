#include <vicon_calibration/measurement_extractors/LidarExtractor.h>

#include <chrono>
#include <thread>

#include <pcl/io/pcd_io.h>

#include <beam_filtering/CropBox.h>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

LidarExtractor::LidarExtractor() {
  keypoints_measured_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
}

void LidarExtractor::SetLidarParams(
    std::shared_ptr<vicon_calibration::LidarParams>& lidar_params) {
  lidar_params_ = lidar_params;
  lidar_params_set_ = true;
}

void LidarExtractor::SetTargetParams(
    std::shared_ptr<vicon_calibration::TargetParams>& target_params) {
  target_params_ = target_params;
  target_params_set_ = true;
}

void LidarExtractor::SetShowMeasurements(const bool& show_measurements) {
  show_measurements_ = show_measurements;
}

bool LidarExtractor::GetShowMeasurements() {
  return show_measurements_;
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
    const Eigen::Matrix4d& T_LIDAR_TARGET_EST,
    const PointCloud::Ptr& cloud_in) {
  scan_in_ = cloud_in;
  T_LIDAR_TARGET_EST_ = T_LIDAR_TARGET_EST;
  this->SetupVariables();
  this->CheckInputs();
  this->IsolatePoints();
  this->GetKeypoints();          // implemented in derived class
  this->CheckMeasurementValid(); // implemented in derived class
  this->GetUserInput();
  this->OutputScans();
  measurement_complete_ = true;
}

void LidarExtractor::SetupVariables() {
  if (show_measurements_) {
    pcl_viewer_ = boost::make_shared<pcl::visualization::PCLVisualizer>();
    int hor_res, vert_res;
    utils::GetScreenResolution(hor_res, vert_res);
    pcl_viewer_->setSize(hor_res, vert_res);
  }
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
  if (!show_measurements_) { return; }
  estimated_template_cloud_ = boost::make_shared<PointCloud>();
  pcl::transformPointCloud(*target_params_->template_cloud,
                           *estimated_template_cloud_,
                           T_LIDAR_TARGET_EST_.cast<float>());

  if (measurement_valid_) {
    std::cout << "Measurement Valid\n";

    PointCloudColor::Ptr estimated_template_cloud_col =
        utils::ColorPointCloud(estimated_template_cloud_, 0, 0, 255);
    measured_template_cloud_ = boost::make_shared<PointCloud>();
    pcl::transformPointCloud(*target_params_->template_cloud,
                             *measured_template_cloud_,
                             T_LIDAR_TARGET_OPT_.cast<float>());
    PointCloudColor::Ptr measured_template_cloud_col =
        utils::ColorPointCloud(measured_template_cloud_, 0, 255, 0);

    // add estimated template cloud
    this->AddColouredPointCloudToViewer(estimated_template_cloud_col,
                                        "blue_cloud", T_LIDAR_TARGET_EST_);

    // add measured template cloud
    this->AddColouredPointCloudToViewer(measured_template_cloud_col,
                                        "green_cloud", T_LIDAR_TARGET_OPT_);

    // add keypoints if discrete keypoints are specified
    if (target_params_->keypoints_lidar.size() > 0) {
      std::cout << "Showing measured keypoints in yellow.\n";
      PointCloudColor::Ptr measured_keypoints =
          utils::ColorPointCloud(keypoints_measured_, 255, 255, 0);
      this->AddColouredPointCloudToViewer(measured_keypoints, "keypoints",
                                          T_LIDAR_TARGET_OPT_, 5);
    }

    // add the isolated scan
    Eigen::Matrix4d T_identity;
    T_identity.setIdentity();
    this->AddPointCloudToViewer(scan_isolated_, "white_cloud", T_identity);
    this->ShowPassedMeasurement();
  } else {
    std::cout << "Measurement Invalid\n";
    PointCloudColor::Ptr scan_isolated_coloured =
        boost::make_shared<PointCloudColor>();
    Eigen::MatrixXd T_identity = Eigen::MatrixXd(4, 4);
    T_identity.setIdentity();
    scan_isolated_coloured = utils::ColorPointCloud(scan_isolated_, 255, 0, 0);
    this->AddColouredPointCloudToViewer(scan_isolated_coloured, "red_cloud",
                                        T_identity, 3);
    this->AddPointCloudToViewer(scan_in_, "white_cloud", T_identity);
    this->ShowFailedMeasurement();
  }
  return;
}

void LidarExtractor::AddColouredPointCloudToViewer(
    const PointCloudColor::Ptr& cloud, const std::string& cloud_name,
    boost::optional<Eigen::MatrixXd&> T = boost::none, int point_size) {
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
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
      cloud);
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, cloud_name);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, cloud_name);
}

void LidarExtractor::AddPointCloudToViewer(const PointCloud::Ptr& cloud,
                                           const std::string& cloud_name,
                                           const Eigen::Matrix4d& T,
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
}

void LidarExtractor::ConfirmMeasurementKeyboardCallback(
    const pcl::visualization::KeyboardEvent& event, void* viewer_void) {
  // check if key has been down for two consecutive spins
  if (viewer_key_down_ && event.keyDown()) {
    return;
  } else if (viewer_key_down_ && !event.keyDown()) {
    viewer_key_down_ = false;
    return;
  } else if (!viewer_key_down_ && event.keyDown()) {
    viewer_key_down_ = true;
  } else if (!viewer_key_down_ && !event.keyDown()) {
    return;
  }

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

void LidarExtractor::ShowFailedMeasurement() {
  // // calculate cropbox cube
  // Eigen::Matrix3d R = T_LIDAR_TARGET_EST_.block(0,0,3,3);
  // Eigen::Vector3d t = T_LIDAR_TARGET_EST_.block(0,3,3,1);
  // Eigen::Quaterniond q(R);
  // double width = target_params_->crop_scan[1] - target_params_->crop_scan[0];
  // double height = target_params_->crop_scan[3] - target_params_->crop_scan[2];
  // double depth = target_params_->crop_scan[5] - target_params_->crop_scan[4];
  // pcl_viewer_->addCube(t.cast<float>(), q.cast<float>(), width, height, depth,
  //                      "bounding_box");
  // pcl_viewer_->setShapeRenderingProperties(
  //     pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
  //     pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
  //     "bounding_box");
  std::cout << "\nViewer Legend:\n"
            << "  Red   -> isolated scan\n"
            << "  White -> original scan\n"
            // << "  Box: cropbox input\n"
            << "Accept measurement? [y/n]\n"
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
  if (measurement_valid_) {
    std::cout << "Accepting measurement" << std::endl;
  } else {
    std::cout << "Rejecting measurement" << std::endl;
  }
}

void LidarExtractor::ShowPassedMeasurement() {
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
  if (measurement_valid_) {
    std::cout << "Accepting measurement" << std::endl;
  } else {
    std::cout << "Rejecting measurement" << std::endl;
  }
}

void LidarExtractor::OutputScans() {
  if (!output_scans_) { return; }
  std::string date_and_time =
      utils::ConvertTimeToDate(std::chrono::system_clock::now());
  if(!boost::filesystem::exists(output_directory_)){
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
