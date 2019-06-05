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
std::string bag_name = "tests/test_bags/2019-06-04-16-46-51.bag";
vicon_calibration::LidarCylExtractor cyl_extractor;
Eigen::Affine3d TA_SCAN_TARGET_EST1, TA_SCAN_TARGET_EST2;
ros::Time transform_lookup_time;
vicon_calibration::PointCloud::Ptr
    temp_cloud(new vicon_calibration::PointCloud);
vicon_calibration::PointCloud::Ptr sim_cloud(new vicon_calibration::PointCloud);

void LoadTemplateCloud() {
  // Load template cloud from pcd file
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(
          "tests/template_pointclouds/cylinder_target.pcd", *temp_cloud) ==
      -1) {
    PCL_ERROR("Couldn't read file cylinder_target.pcd \n");
  }
}

void LoadSimulatedCloud() {
  // Rosbag for accessing bag data
  rosbag::Bag bag;
  bag.open(bag_name, rosbag::bagmode::Read);
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
  bag.open(bag_name, rosbag::bagmode::Read);
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
  std::string calibration_json_name = "tests/calibration/roben_extrinsic.json";
  tf_tree.LoadJSON(calibration_json_name);

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

TEST_CASE("Test cylinder extractor with empty template cloud (nullptr)") {
  bool accept_measurement;

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
  double default_threshold = 0.015;
  bool accept_measurement;

  cyl_extractor.SetThreshold(-0.025);
  REQUIRE_THROWS(
      cyl_extractor.ExtractCylinder(TA_SCAN_TARGET_EST1, accept_measurement));
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
}

TEST_CASE("Test cylinder extractor") {
  bool accept_measurement1, accept_measurement2;

  auto measured_transform1 = cyl_extractor.ExtractCylinder(
      TA_SCAN_TARGET_EST1, accept_measurement1, 1);
  auto measured_transform2 = cyl_extractor.ExtractCylinder(
      TA_SCAN_TARGET_EST2, accept_measurement2, 2);
  auto true_transform1 =
      cyl_extractor.ExtractRelevantMeasurements(TA_SCAN_TARGET_EST1);
  auto true_transform2 =
      cyl_extractor.ExtractRelevantMeasurements(TA_SCAN_TARGET_EST2);

  double error1 =
      std::round((measured_transform1 - true_transform1).norm() * 100) / 100;
  double error2 =
      std::round((measured_transform2 - true_transform2).norm() * 100) / 100;
  REQUIRE(error1 <= 0.01);
  REQUIRE(error2 <= 0.01);
}
