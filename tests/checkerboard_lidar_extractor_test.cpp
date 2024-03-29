#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_msgs/TFMessage.h>

#include <vicon_calibration/JsonTools.h>
#include <vicon_calibration/PclConversions.h>
#include <vicon_calibration/TfTree.h>
#include <vicon_calibration/Utils.h>
#include <vicon_calibration/measurement_extractors/CheckerboardLidarExtractor.h>

using namespace vicon_calibration;

// Global variables for testing
std::string bag_path, template_cloud, template_cloud_rot, target_config;
Eigen::Matrix4d T_SCAN_TARGET1_TRUE, T_SCAN_TARGET1_EST_CONV,
    T_SCAN_TARGET1_EST_DIV, T_SCAN_TARGET2_TRUE, T_SCAN_TARGET2_EST_CONV;
ros::Time transform_lookup_time;
std::shared_ptr<TargetParams> target_params;
std::shared_ptr<LidarParams> lidar_params;
PointCloud::Ptr temp_cloud;
PointCloud::Ptr sim_cloud;
bool close_viewer{false};
bool show_measurements{false};
bool setup_called{false};
Eigen::VectorXd T_perturb_small = Eigen::VectorXd(6);
Eigen::VectorXd T_perturb_large = Eigen::VectorXd(6);
std::string m3d_link_frame = "m3d_link";
std::string target1_frame = "vicon/target11/target11";
std::string target2_frame = "vicon/target12/target12";
std::shared_ptr<Visualizer> pcl_viewer;

using namespace std::literals::chrono_literals;

void FileSetup() {
  bag_path = utils::GetFilePathTestBags("roben_simulation.bag");
  target_config = utils::GetFilePathTestData("CheckerboardTargetSim.json");
  template_cloud =
      utils::GetFilePathTestClouds("checkerboard_target_simulation.pcd");
}

void LoadSimulatedCloud() {
  sim_cloud = std::make_shared<PointCloud>();
  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);
  rosbag::TopicQuery query = rosbag::TopicQuery("/m3d/aggregator/cloud");
  rosbag::View cloud_bag_view = {bag, query};
  for (const auto& msg_instance : cloud_bag_view) {
    auto cloud_message = msg_instance.instantiate<sensor_msgs::PointCloud2>();
    if (cloud_message != nullptr) {
      pcl_conversions::toPCL(*cloud_message, *sim_cloud);
      transform_lookup_time = cloud_message->header.stamp;
    }
  }
}

void LoadTransforms() {
  TfTree tf_tree;
  std::string calibration_json;
  calibration_json = utils::GetFilePathTestData("roben_extrinsics.json");
  tf_tree.LoadJSON(calibration_json);
  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);
  rosbag::View tf_bag_view = {bag, rosbag::TopicQuery("/tf")};

  // Iterate over all message instances in our tf bag view
  std::cout << "Adding transforms" << std::endl;
  for (const auto& msg_instance : tf_bag_view) {
    auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
    if (tf_message != nullptr) {
      for (geometry_msgs::TransformStamped tf : tf_message->transforms) {
        tf_tree.AddTransform(tf);
      }
    }
  }
  // Get transforms between targets and lidar and world and targets
  Eigen::Affine3d TA_SCAN_TARGET1_TRUE, TA_SCAN_TARGET2_TRUE;
  TA_SCAN_TARGET1_TRUE = tf_tree.GetTransformEigen(
      m3d_link_frame, target1_frame, transform_lookup_time);
  TA_SCAN_TARGET2_TRUE = tf_tree.GetTransformEigen(
      m3d_link_frame, target2_frame, transform_lookup_time);
  T_SCAN_TARGET1_TRUE = TA_SCAN_TARGET1_TRUE.matrix();
  T_SCAN_TARGET2_TRUE = TA_SCAN_TARGET2_TRUE.matrix();

  // Apply perturbation to estimate
  T_perturb_small << 0.01, -0.01, 0, 0.1, 0.1, 0.2;
  T_perturb_large << 0.1, -0.1, 0, 0.5, -1, 0.5;
  T_SCAN_TARGET1_EST_CONV =
      utils::PerturbTransformDegM(T_SCAN_TARGET1_TRUE, T_perturb_small);
  T_SCAN_TARGET2_EST_CONV =
      utils::PerturbTransformDegM(T_SCAN_TARGET2_TRUE, T_perturb_small);
  T_SCAN_TARGET1_EST_DIV =
      utils::PerturbTransformDegM(T_SCAN_TARGET1_TRUE, T_perturb_large);
}

void LoadTargetParams() {
  CalibratorInputs inputs;
  inputs.target_config_path = target_config;
  JsonTools json_loader(inputs);
  target_params = json_loader.LoadTargetParams(target_config);

  // replace template with template from test data
  temp_cloud = std::make_shared<PointCloud>();
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud, *temp_cloud) == -1) {
    PCL_ERROR("Couldn't read file %s \n", template_cloud.c_str());
  }
  target_params->template_cloud = temp_cloud;
}

void LoadLidarParams() {
  lidar_params = std::make_shared<LidarParams>();
  lidar_params->topic = "null_topic";
  lidar_params->frame = "lidar";
}

void LoadTargetParamsRotated() {
  CalibratorInputs inputs;
  inputs.target_config_path = target_config;
  JsonTools json_loader(inputs);
  target_params = json_loader.LoadTargetParams(target_config);
  PointCloud::Ptr temp_cloud;
  temp_cloud = std::make_shared<PointCloud>();
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud_rot, *temp_cloud) ==
      -1) {
    PCL_ERROR("Couldn't read file %s \n", template_cloud_rot.c_str());
  }
  target_params->template_cloud = temp_cloud;
}

void Setup() {
  if (setup_called) { return; }
  if (show_measurements) {
    pcl_viewer = std::make_shared<Visualizer>("CylTestVis");
  }
  FileSetup();
  LoadTransforms();
  LoadSimulatedCloud();
  LoadTargetParams();
  LoadLidarParams();
  setup_called = true;
}

TEST_CASE(
    "Test checkerboard extractor with empty template cloud && empty scan") {
  Setup();
  std::shared_ptr<TargetParams> invalid_target_params =
      std::make_shared<TargetParams>(*target_params);
  std::shared_ptr<PointCloud> null_cloud;
  std::shared_ptr<PointCloud> empty_cloud = std::make_shared<PointCloud>();
  invalid_target_params->template_cloud = null_cloud;

  std::shared_ptr<LidarExtractor> checkerboard_extractor =
      std::make_shared<CheckerboardLidarExtractor>(
          lidar_params, invalid_target_params, show_measurements, pcl_viewer);
  REQUIRE_THROWS(checkerboard_extractor->ProcessMeasurement(T_SCAN_TARGET1_TRUE,
                                                            sim_cloud));

  invalid_target_params->template_cloud = empty_cloud;
  checkerboard_extractor = std::make_shared<CheckerboardLidarExtractor>(
      lidar_params, invalid_target_params, show_measurements, pcl_viewer);
  REQUIRE_THROWS(checkerboard_extractor->ProcessMeasurement(T_SCAN_TARGET1_TRUE,
                                                            sim_cloud));

  checkerboard_extractor = std::make_shared<CheckerboardLidarExtractor>(
      lidar_params, target_params, show_measurements, pcl_viewer);
  REQUIRE_THROWS(checkerboard_extractor->ProcessMeasurement(T_SCAN_TARGET1_TRUE,
                                                            null_cloud));
  REQUIRE_THROWS(checkerboard_extractor->ProcessMeasurement(T_SCAN_TARGET1_TRUE,
                                                            empty_cloud));
}

TEST_CASE("Test extracting checkerboard with invalid transformation matrix") {
  Setup();
  Eigen::Affine3d TA_INVALID1;
  Eigen::Matrix4d T_INVALID2, T_INVALID3;
  T_INVALID3.setIdentity();
  T_INVALID3(3, 0) = 1;
  std::shared_ptr<LidarExtractor> checkerboard_extractor =
      std::make_shared<CheckerboardLidarExtractor>(
          lidar_params, target_params, show_measurements, pcl_viewer);
  REQUIRE_THROWS(checkerboard_extractor->ProcessMeasurement(
      TA_INVALID1.matrix(), sim_cloud));
  REQUIRE_THROWS(
      checkerboard_extractor->ProcessMeasurement(T_INVALID2, sim_cloud));
  REQUIRE_THROWS(
      checkerboard_extractor->ProcessMeasurement(T_INVALID3, sim_cloud));
}

TEST_CASE("Test extracting checkerboard target with and without diverged ICP "
          "registration") {
  Setup();
  std::shared_ptr<TargetParams> div_target_params =
      std::make_shared<TargetParams>(*target_params);
  std::shared_ptr<TargetParams> conv_target_params =
      std::make_shared<TargetParams>(*target_params);
  Eigen::VectorXf div_crop1(6);
  div_crop1 << 0.3, -0.3, 0.3, -0.3, 0.3, -0.3;
  Eigen::VectorXf good_crop(6);
  good_crop << -0.4, 0.4, -0.4, 0.4, -0.4, 0.4;
  Eigen::VectorXf good_crop2(6);
  good_crop2 << -1, 1, -1, 1, -1, 1;
  div_target_params->crop_scan = div_crop1;

  std::shared_ptr<LidarExtractor> checkerboard_extractor =
      std::make_shared<CheckerboardLidarExtractor>(
          lidar_params, div_target_params, show_measurements, pcl_viewer);
  REQUIRE_THROWS(checkerboard_extractor->GetMeasurementValid());
  checkerboard_extractor->ProcessMeasurement(T_SCAN_TARGET1_TRUE, sim_cloud);
  REQUIRE(checkerboard_extractor->GetMeasurementValid() == false);
  div_target_params->crop_scan = good_crop;
  checkerboard_extractor->ProcessMeasurement(T_SCAN_TARGET1_TRUE, sim_cloud);
  REQUIRE(checkerboard_extractor->GetMeasurementValid() == true);
  checkerboard_extractor->ProcessMeasurement(T_SCAN_TARGET1_EST_CONV,
                                             sim_cloud);
  REQUIRE(checkerboard_extractor->GetMeasurementValid() == true);
  checkerboard_extractor->ProcessMeasurement(T_SCAN_TARGET1_EST_DIV, sim_cloud);
  REQUIRE(checkerboard_extractor->GetMeasurementValid() == false);
  checkerboard_extractor->ProcessMeasurement(T_SCAN_TARGET2_TRUE, sim_cloud);
  REQUIRE(checkerboard_extractor->GetMeasurementValid() == true);

  conv_target_params->crop_scan = good_crop2;
  checkerboard_extractor = std::make_shared<CheckerboardLidarExtractor>(
      lidar_params, conv_target_params, show_measurements, pcl_viewer);
  checkerboard_extractor->ProcessMeasurement(T_SCAN_TARGET2_EST_CONV,
                                             sim_cloud);
  REQUIRE(checkerboard_extractor->GetMeasurementValid() == true);
}

// need to view these results, can't automate the test
TEST_CASE("Test best correspondence estimation") {
  if (!show_measurements) { return; }
  Setup();

  Eigen::Affine3d T_identity;
  T_identity.setIdentity();
  std::shared_ptr<TargetParams> target_params2 =
      std::make_shared<TargetParams>(*target_params);
  Eigen::VectorXf div_crop(6);
  div_crop << -0.5, 0.5, -0.2, 0.2, -0.2, 0.2;
  target_params2->crop_scan = div_crop;

  std::shared_ptr<LidarExtractor> checkerboard_extractor =
      std::make_shared<CheckerboardLidarExtractor>(
          lidar_params, target_params2, show_measurements, pcl_viewer);

  // view keypoints 1
  checkerboard_extractor->ProcessMeasurement(T_SCAN_TARGET1_TRUE, sim_cloud);
  std::shared_ptr<PointCloud> keypoints1 =
      checkerboard_extractor->GetMeasurement();
  pcl_viewer->AddPointCloudToViewer(keypoints1, "keypoints1",
                                    Eigen::Vector3i(255, 0, 0), 5);

  // view keypoints 2
  // can't run these both at the same time
  Eigen::VectorXf div_crop2(6);
  div_crop2 << -0.7, 0.7, -0.5, 0.5, -0.5, 0.5;
  target_params2->crop_scan = div_crop2;
  checkerboard_extractor->ProcessMeasurement(T_SCAN_TARGET1_EST_CONV,
                                             sim_cloud);
  std::shared_ptr<PointCloud> keypoints2 =
      checkerboard_extractor->GetMeasurement();
  pcl_viewer->AddPointCloudToViewer(keypoints2, "keypoints2",
                                    Eigen::Vector3i(0, 255, 0), 5);
}
//
