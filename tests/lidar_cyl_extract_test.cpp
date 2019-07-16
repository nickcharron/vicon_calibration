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
std::string rotated_template_cloud_path;

vicon_calibration::LidarCylExtractor cyl_extractor;
Eigen::Affine3d TA_SCAN_TARGET_EST1, TA_SCAN_TARGET_EST2;
ros::Time transform_lookup_time;
vicon_calibration::PointCloud::Ptr
    temp_cloud(new vicon_calibration::PointCloud);
vicon_calibration::PointCloud::Ptr sim_cloud(new vicon_calibration::PointCloud);

void FileSetup() {
  test_path.erase(test_path.end() - current_file.size(), test_path.end());
  bag_path = test_path + "test_bags/roben_simulation.bag";
  template_cloud_path = test_path + "template_pointclouds/cylinder_target.pcd";
  rotated_template_cloud_path =
      test_path + "template_pointclouds/cylinder_target_rotated.pcd";
}

void LoadTemplateCloud(std::string temp_cloud_path) {
  // Load template cloud from pcd file
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(temp_cloud_path, *temp_cloud) == -1) {
    PCL_ERROR("Couldn't read file %s \n", temp_cloud_path);
  }
}

void LoadSimulatedCloud() {
  // Rosbag for accessing bag data
  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);
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
  bag.open(bag_path, rosbag::bagmode::Read);
  rosbag::View tf_bag_view = {bag, rosbag::TopicQuery("/tf")};

  tf_tree.start_time = tf_bag_view.getBeginTime();

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
  calibration_json = test_path + "calibration/roben_extrinsics.json";
  tf_tree.LoadJSON(calibration_json);

  // Get transforms between targets and lidar and world and targets
  std::string m3d_link_frame = "m3d_link";
  std::string target1_frame = "vicon/cylinder_target1/cylinder_target1";
  std::string target2_frame = "vicon/cylinder_target2/cylinder_target2";

  auto T_SCAN_TARGET_EST1_msg = tf_tree.GetTransformROS(
      m3d_link_frame, target1_frame, transform_lookup_time);

  auto T_SCAN_TARGET_EST2_msg = tf_tree.GetTransformROS(
      m3d_link_frame, target2_frame, transform_lookup_time);

  TA_SCAN_TARGET_EST1 = tf2::transformToEigen(T_SCAN_TARGET_EST1_msg);
  TA_SCAN_TARGET_EST2 = tf2::transformToEigen(T_SCAN_TARGET_EST2_msg);
}

/* TEST CASES */

TEST_CASE("Test cylinder extractor with empty template cloud (nullptr)") {
  FileSetup();
  LoadTransforms();

  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1));
}

TEST_CASE("Test extracting cylinder from empty scan (nullptr)") {
  LoadTemplateCloud(template_cloud_path);

  cyl_extractor.SetTemplateCloud(temp_cloud);
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1));
}

TEST_CASE("Test extracting cylinder with invalid transformation matrix") {
  Eigen::Affine3d TA_INVALID;

  LoadSimulatedCloud();

  cyl_extractor.SetScan(sim_cloud);
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(TA_INVALID));
}

TEST_CASE("Test extracting cylinder target with diverged ICP registration") {
  double default_threshold = 0.01;

  cyl_extractor.SetThreshold(-0.05);
  cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1);
  auto measurement_info = cyl_extractor.GetMeasurementInfo();

  REQUIRE(measurement_info.second == false);
  cyl_extractor.SetThreshold(default_threshold);
}

TEST_CASE("Test extracting cylinder with invalid parameters") {
  double default_radius = 0.0635;
  double default_height = 0.5;

  cyl_extractor.SetRadius(0);
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1));
  cyl_extractor.SetRadius(default_radius);

  cyl_extractor.SetHeight(0);
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1));
  cyl_extractor.SetHeight(default_height);
}

TEST_CASE("Test getting measurement information before extracting cylinder") {
  REQUIRE_THROWS(cyl_extractor.GetMeasurementInfo());
}

TEST_CASE("Test cylinder extractor") {
  std::pair<Eigen::Vector4d, bool> measurement_info1, measurement_info2;

  auto true_transform1 =
      cyl_extractor.ExtractRelevantMeasurements(TA_SCAN_TARGET_EST1);
  cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1, 1);
  measurement_info1 = cyl_extractor.GetMeasurementInfo();

  auto true_transform2 =
      cyl_extractor.ExtractRelevantMeasurements(TA_SCAN_TARGET_EST2);
  cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST2, 2);
  measurement_info2 = cyl_extractor.GetMeasurementInfo();

  Eigen::Vector2d dist_diff1(measurement_info1.first(0) - true_transform1(0),
                             measurement_info1.first(1) - true_transform1(1));
  Eigen::Vector2d rot_diff1(measurement_info1.first(2) - true_transform1(2),
                            measurement_info1.first(3) - true_transform1(3));

  double dist_err1 = std::round(dist_diff1.norm() * 10000) / 10000;
  double rot_err1 = std::round(rot_diff1.norm() * 10000) / 10000;

  Eigen::Vector2d dist_diff2(measurement_info2.first(0) - true_transform2(0),
                             measurement_info2.first(1) - true_transform2(1));
  Eigen::Vector2d rot_diff2(measurement_info2.first(2) - true_transform2(2),
                            measurement_info2.first(3) - true_transform2(3));

  double dist_err2 = std::round(dist_diff2.norm() * 1000) / 1000;
  double rot_err2 = std::round(rot_diff2.norm() * 1000) / 1000;

  REQUIRE(dist_err1 <= 0.02); // less than 2 cm
  REQUIRE(dist_err2 <= 0.02);
  REQUIRE(rot_err1 <= 0.2); // less than 0.2 rad
  REQUIRE(rot_err2 <= 0.2);
  REQUIRE(measurement_info1.second == true);
  REQUIRE(measurement_info2.second == true);
}

TEST_CASE("Test auto rejection") {
  std::pair<Eigen::Vector4d, bool> measurement_info;

  LoadTemplateCloud(rotated_template_cloud_path);

  cyl_extractor.SetTemplateCloud(temp_cloud);
  cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1, 3);
  measurement_info = cyl_extractor.GetMeasurementInfo();

  REQUIRE(measurement_info.second == false);
}
