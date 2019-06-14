#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "vicon_calibration/LidarCylExtractor.h"
#include "vicon_calibration/utils.hpp"

#include <beam_calibration/TfTree.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_msgs/TFMessage.h>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

// Global variables for testing
std::string current_file = "lidar_cyl_extract_test.cpp";
std::string test_path = __FILE__;
std::string bag_path;
std::string template_cloud_path;
std::string config_file;

double max_corr, t_eps, fit_eps, threshold, x, y, z;
int max_iter;
std::string bag_file;
std::string template_cloud_file;
bool set_show_transform;

vicon_calibration::LidarCylExtractor cyl_extractor;
Eigen::Affine3d TA_SCAN_TARGET_EST1, TA_SCAN_TARGET_EST2;
ros::Time transform_lookup_time;
vicon_calibration::PointCloud::Ptr
    temp_cloud(new vicon_calibration::PointCloud);
vicon_calibration::PointCloud::Ptr sim_cloud(new vicon_calibration::PointCloud);


void LoadJson(std::string file_name) {
  nlohmann::json J;
  std::ifstream file(file_name);
  file >> J;

  bag_file = J["bag_file"];
  template_cloud_file = J["template_cloud_name"];

  max_corr = J["max_corr"];
  max_iter = J["max_iter"];
  t_eps = J["t_eps"];
  fit_eps = J["fit_eps"];
  threshold = J["threshold"];
  x = J["x"];
  y = J["y"];
  z = J["z"];

  set_show_transform = J["set_show_transform"];
}

void FileSetup() {
  test_path.erase(test_path.end() - current_file.size(), test_path.end());
  bag_path = test_path + "test_bags/2019-06-14-15-13-46.bag";
  template_cloud_path = test_path + "template_pointclouds/cylinder_target_rotated.pcd";
}

void LoadTemplateCloud() {
  // Load template cloud from pcd file
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud_file, *temp_cloud) ==
      -1) {
    PCL_ERROR("Couldn't read file cylinder_target.pcd \n");
  }
}

void LoadSimulatedCloud() {
  // Rosbag for accessing bag data
  rosbag::Bag bag;
  bag.open(bag_file, rosbag::bagmode::Read);
  // Load the simulated pointcloud from the bag
  rosbag::View cloud_bag_view = {bag,
                                 rosbag::TopicQuery("/m3d/aggregator/cloud")};

  for (const auto &msg_instance : cloud_bag_view) {
    auto cloud_message = msg_instance.instantiate<sensor_msgs::PointCloud2>();
    if (cloud_message != nullptr) {
      pcl::fromROSMsg(*cloud_message, *sim_cloud);
      transform_lookup_time = cloud_message->header.stamp;
    }
  }
}

void LoadTransforms() {
  // TfTree for storing all static / dynamic transforms
  beam_calibration::TfTree tf_tree;

  // Rosbag for accessing bag data
  rosbag::Bag bag;
  bag.open(bag_file, rosbag::bagmode::Read);
  rosbag::View tf_bag_view = {bag, rosbag::TopicQuery("/tf")};

  tf_tree.start_time_ = tf_bag_view.getBeginTime();

  // Iterate over all message instances in our tf bag view
  std::cout << "Adding transforms" << std::endl;
  for (const auto &msg_instance : tf_bag_view) {
    auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
    if (tf_message != nullptr) {
      for (const geometry_msgs::TransformStamped &tf : tf_message->transforms) {
        tf_tree.AddTransform(tf);
      }
    }
  }
  // Load calibration json and add transform between m3d_link and base_link
  std::string calibration_json;
  calibration_json = test_path + "calibration/roben_extrinsic.json";
  tf_tree.LoadJSON(calibration_json);

  // Get transforms between targets and lidar and world and targets
  std::string m3d_link_frame = "m3d_link";
  std::string target1_frame = "vicon/cylinder_target1/cylinder_target1";
  std::string target2_frame = "vicon/cylinder_target2/cylinder_target2";

  auto T_SCAN_TARGET_EST1_msg = tf_tree.GetTransform(
      m3d_link_frame, target1_frame, transform_lookup_time);

  auto T_SCAN_TARGET_EST2_msg = tf_tree.GetTransform(
      m3d_link_frame, target2_frame, transform_lookup_time);

  TA_SCAN_TARGET_EST1 = tf2::transformToEigen(T_SCAN_TARGET_EST1_msg);
  TA_SCAN_TARGET_EST2 = tf2::transformToEigen(T_SCAN_TARGET_EST2_msg);
}
/*
TEST_CASE("Test cylinder extractor with empty template cloud (nullptr)") {
  bool accept_measurement;

  FileSetup();
  LoadJson(test_path + "test.json");
  LoadTransforms();

  REQUIRE_THROWS(
      cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1, accept_measurement));
}

TEST_CASE("Test extracting cylinder from empty scan (nullptr)") {
  bool accept_measurement;

  LoadTemplateCloud();

  cyl_extractor.SetTemplateCloud(temp_cloud);
  REQUIRE_THROWS(
      cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1, accept_measurement));
}

TEST_CASE("Test extracting cylinder with invalid transformation matrix") {
  Eigen::Affine3d TA_INVALID;
  bool accept_measurement;

  LoadSimulatedCloud();

  cyl_extractor.SetScan(sim_cloud);
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(TA_INVALID, accept_measurement));
}

TEST_CASE("Test extracting cylinder target with diverged ICP registration") {
  double default_threshold = 0.001;
  bool accept_measurement;

  cyl_extractor.SetThreshold(-0.05);
  cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1, accept_measurement);

  REQUIRE(accept_measurement == false);
  cyl_extractor.SetThreshold(default_threshold);
}

TEST_CASE("Test extracting cylinder with invalid parameters") {
  double default_radius = 0.0635;
  double default_height = 0.5;
  bool accept_measurement;

  cyl_extractor.SetRadius(0);
  REQUIRE_THROWS(
      cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1, accept_measurement));
  cyl_extractor.SetRadius(default_radius);

  cyl_extractor.SetHeight(0);
  REQUIRE_THROWS(
      cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1, accept_measurement));
  cyl_extractor.SetHeight(default_height);
}*/

TEST_CASE("Test cylinder extractor") {
  bool accept_measurement1, accept_measurement2;

  FileSetup();
  LoadJson(test_path + "test.json");
  LoadTransforms();
  LoadTemplateCloud();
  LoadSimulatedCloud();
  cyl_extractor.SetTemplateCloud(temp_cloud);
  cyl_extractor.SetScan(sim_cloud);

  cyl_extractor.SetICPConfigs(t_eps, fit_eps, max_corr, max_iter);
  cyl_extractor.SetThreshold(threshold);
  cyl_extractor.SetXYZ(x, y, z);
  cyl_extractor.SetShowTransformation(set_show_transform);

  auto measured_transform1 = cyl_extractor.ExtractCylinder(
      TA_SCAN_TARGET_EST1, accept_measurement1, 1);
  auto measured_transform2 = cyl_extractor.ExtractCylinder(
      TA_SCAN_TARGET_EST2, accept_measurement2, 2);
  auto true_transform1 =
      cyl_extractor.ExtractRelevantMeasurements(TA_SCAN_TARGET_EST1);
  auto true_transform2 =
      cyl_extractor.ExtractRelevantMeasurements(TA_SCAN_TARGET_EST2);

  std::cout << "Measured transform 1" << std::endl << measured_transform1 << std::endl;
  std::cout << "True transform 1" << std::endl << true_transform1 << std::endl;
  std::cout << "Measured transform 2" << std::endl << measured_transform2 << std::endl;
  std::cout << "True transform 2" << std::endl << true_transform2 << std::endl;

  Eigen::Vector2d dist_diff1(measured_transform1(0)-true_transform1(0), measured_transform1(1)-true_transform1(1));
  Eigen::Vector2d rot_diff1(measured_transform1(2)-true_transform1(2), measured_transform1(3)-true_transform1(3));

  double dist_err1 = dist_diff1.norm();
  double rot_err1 = rot_diff1.norm();

  std::cout << "DIST ERROR 1: " << dist_err1 << std::endl;
  std::cout << "ROT ERROR 1: " << rot_err1 << std::endl;

  Eigen::Vector2d dist_diff2(measured_transform2(0)-true_transform2(0), measured_transform2(1)-true_transform2(1));
  Eigen::Vector2d rot_diff2(measured_transform2(2)-true_transform2(2), measured_transform2(3)-true_transform2(3));

  double dist_err2 = dist_diff2.norm();
  double rot_err2 = rot_diff2.norm();

  std::cout << "DIST ERROR 2: " << dist_err2 << std::endl;
  std::cout << "ROT ERROR 2: " << rot_err2 << std::endl;

  double error1 =
      std::round((measured_transform1 - true_transform1).norm() * 100) / 100;
  double error2 =
      std::round((measured_transform2 - true_transform2).norm() * 100) / 100;
  REQUIRE(error1 <= 0.01);
  REQUIRE(error2 <= 0.01);
}
