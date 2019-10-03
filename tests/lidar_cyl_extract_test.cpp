#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "vicon_calibration/LidarCylExtractor.h"
#include "vicon_calibration/utils.h"

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
Eigen::Affine3d T_SCAN_TARGET1_EST, T_SCAN_TARGET2_EST, T_SCAN_TARGET1_TRUE,
    T_SCAN_TARGET2_TRUE;
ros::Time transform_lookup_time;
vicon_calibration::RegistrationParams registration_params;
vicon_calibration::CylinderTgtParams target_params;
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

  if (pcl::io::loadPCDFile<pcl::PointXYZ>(temp_cloud_path, *temp_cloud) == -1) {
    PCL_ERROR("Couldn't read file %s \n", temp_cloud_path);
  }
}

void LoadSimulatedCloud() {

  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);
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

  beam_calibration::TfTree tf_tree;
  std::string calibration_json;
  calibration_json = test_path + "data/roben_extrinsics.json";
  tf_tree.LoadJSON(calibration_json);
  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);
  rosbag::View tf_bag_view = {bag, rosbag::TopicQuery("/tf")};

  tf_tree.start_time = tf_bag_view.getBeginTime();

  // Iterate over all message instances in our tf bag view
  std::cout << "Adding transforms" << std::endl;
  for (const auto &msg_instance : tf_bag_view) {
    auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
    if (tf_message != nullptr) {
      for (geometry_msgs::TransformStamped tf : tf_message->transforms) {
        tf_tree.AddTransform(tf);
      }
    }
  }

  // Get transforms between targets and lidar and world and targets
  std::string m3d_link_frame = "m3d_link";
  std::string target1_frame = "vicon/cylinder_target1/cylinder_target1";
  std::string target2_frame = "vicon/cylinder_target2/cylinder_target2";

  T_SCAN_TARGET1_TRUE = tf_tree.GetTransformEigen(m3d_link_frame, target1_frame,
                                                  transform_lookup_time);

  T_SCAN_TARGET2_TRUE = tf_tree.GetTransformEigen(m3d_link_frame, target2_frame,
                                                  transform_lookup_time);

  // Apply translation to estimate
  Eigen::Vector3d t_perturb(0.1, 0.1, 0.2);
  T_SCAN_TARGET1_EST = T_SCAN_TARGET1_TRUE;
  T_SCAN_TARGET2_EST = T_SCAN_TARGET2_TRUE;
  T_SCAN_TARGET1_EST.translation() = T_SCAN_TARGET1_EST.translation() + t_perturb;
  T_SCAN_TARGET2_EST.translation() = T_SCAN_TARGET2_EST.translation() + t_perturb;
  // std::cout << "T_SCAN_TARGET1_TRUE:\n" << T_SCAN_TARGET1_TRUE.matrix() << "\n";
  // std::cout << "T_SCAN_TARGET2_TRUE:\n" << T_SCAN_TARGET2_TRUE.matrix() << "\n";
  // std::cout << "T_SCAN_TARGET1_EST:\n" << T_SCAN_TARGET1_EST.matrix() << "\n";
  // std::cout << "T_SCAN_TARGET2_EST:\n" << T_SCAN_TARGET2_EST.matrix() << "\n";
}

void LoadRegistrationParams() {
  registration_params.max_correspondance_distance = 1;
  registration_params.max_iterations = 10;
  registration_params.transform_epsilon = 1e-8;
  registration_params.euclidean_epsilon = 1e-4;
  registration_params.show_transform = false;
  registration_params.dist_acceptance_criteria = 0.2;
  registration_params.rot_acceptance_criteria = 0.5;
  registration_params.crop_threshold_x = 0.0;
  registration_params.crop_threshold_y = 0.3;
  registration_params.crop_threshold_z = 0.3;
}

void LoadTargetParams(){
  target_params.radius = 0.0635;
  target_params.height = 0.5;
  target_params.template_cloud = template_cloud_path;
}

void LoadTargetParamsRotated(){
  target_params.radius = 0.0635;
  target_params.height = 0.5;
  target_params.template_cloud = rotated_template_cloud_path;
}

TEST_CASE("Test cylinder extractor with empty template cloud (nullptr)") {
  FileSetup();
  LoadTransforms();
  LoadTemplateCloud(template_cloud_path);
  LoadSimulatedCloud();
  LoadRegistrationParams();
  LoadTargetParams();
  vicon_calibration::LidarCylExtractor cyl_extractor;
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(T_SCAN_TARGET1_EST));
}

TEST_CASE("Test extracting cylinder from empty scan (nullptr)") {
  // FileSetup();
  // LoadTransforms();
  // LoadTemplateCloud(template_cloud_path);
  // LoadSimulatedCloud();
  LoadRegistrationParams();
  LoadTargetParams();
  vicon_calibration::LidarCylExtractor cyl_extractor;
  cyl_extractor.SetRegistrationParams(registration_params);
  cyl_extractor.SetTargetParams(target_params);
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(T_SCAN_TARGET1_EST));
}

TEST_CASE("Test extracting cylinder with invalid transformation matrix") {
  // FileSetup(); // only run this command once
  // LoadTransforms();
  // LoadTemplateCloud(template_cloud_path);
  // LoadSimulatedCloud();
  LoadRegistrationParams();
  LoadTargetParams();
  Eigen::Affine3d T_INVALID;
  vicon_calibration::LidarCylExtractor cyl_extractor;
  cyl_extractor.SetRegistrationParams(registration_params);
  cyl_extractor.SetTargetParams(target_params);
  cyl_extractor.SetScan(sim_cloud);
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(T_INVALID));
}

TEST_CASE("Test extracting cylinder target with diverged ICP registration") {
  // FileSetup();
  // LoadTransforms();
  // LoadTemplateCloud(template_cloud_path);
  // LoadSimulatedCloud();
  LoadRegistrationParams();
  LoadTargetParams();
  registration_params.crop_threshold_x = -0.05;
  registration_params.crop_threshold_y = -0.05;
  registration_params.crop_threshold_z = -0.05;
  vicon_calibration::LidarCylExtractor cyl_extractor;
  cyl_extractor.SetRegistrationParams(registration_params);
  cyl_extractor.SetTargetParams(target_params);
  cyl_extractor.SetScan(sim_cloud);
  cyl_extractor.ExtractCylinder(T_SCAN_TARGET1_TRUE);
  auto measurement_info = cyl_extractor.GetMeasurementInfo();
  REQUIRE(measurement_info.second == false);
}

TEST_CASE("Test extracting cylinder with invalid parameters") {
  // FileSetup();
  // LoadTransforms();
  // LoadTemplateCloud(template_cloud_path);
  // LoadSimulatedCloud();
  LoadRegistrationParams();
  LoadTargetParams();
  target_params.radius = 0;
  vicon_calibration::LidarCylExtractor cyl_extractor;
  cyl_extractor.SetRegistrationParams(registration_params);
  REQUIRE_THROWS(cyl_extractor.SetTargetParams(target_params));
  target_params.radius = 0.0635;
  target_params.height = 0;
  REQUIRE_THROWS(cyl_extractor.SetTargetParams(target_params));
}

TEST_CASE("Test getting measurement information before extracting cylinder") {
  // FileSetup();
  // LoadTransforms();
  // LoadTemplateCloud(template_cloud_path);
  // LoadSimulatedCloud();
  LoadRegistrationParams();
  LoadTargetParams();
  vicon_calibration::LidarCylExtractor cyl_extractor;
  cyl_extractor.SetRegistrationParams(registration_params);
  cyl_extractor.SetTargetParams(target_params);
  cyl_extractor.SetScan(sim_cloud);
  REQUIRE_THROWS(cyl_extractor.GetMeasurementInfo());
}

TEST_CASE("Test cylinder extractor") {
  // FileSetup();
  // LoadTransforms();
  // LoadTemplateCloud(template_cloud_path);
  // LoadSimulatedCloud();
  LoadRegistrationParams();
  LoadTargetParams();
  vicon_calibration::LidarCylExtractor cyl_extractor;
  // registration_params.show_transform = true;
  cyl_extractor.SetRegistrationParams(registration_params);
  cyl_extractor.SetTargetParams(target_params);
  cyl_extractor.SetScan(sim_cloud);

  std::pair<Eigen::Affine3d, bool> measurement_info1, measurement_info2;
  cyl_extractor.ExtractCylinder(T_SCAN_TARGET1_EST, 1);
  measurement_info1 = cyl_extractor.GetMeasurementInfo();
  Eigen::Affine3d measurement1 = measurement_info1.first;
  bool measurement1_valid = measurement_info1.second;

  cyl_extractor.ExtractCylinder(T_SCAN_TARGET2_EST, 2);
  measurement_info2 = cyl_extractor.GetMeasurementInfo();
  Eigen::Affine3d measurement2 = measurement_info2.first;
  bool measurement2_valid = measurement_info2.second;

  Eigen::Vector2d dist_diff1(
      measurement1.matrix()(0, 3) - T_SCAN_TARGET1_TRUE.matrix()(0, 3),
      measurement1.matrix()(1, 3) - T_SCAN_TARGET1_TRUE.matrix()(1, 3));
  Eigen::Vector2d dist_diff2(
      measurement2.matrix()(0, 3) - T_SCAN_TARGET2_TRUE.matrix()(0, 3),
      measurement2.matrix()(1, 3) - T_SCAN_TARGET2_TRUE.matrix()(1, 3));
  double dist_err1 = std::round(dist_diff1.norm() * 10000) / 10000;
  double dist_err2 = std::round(dist_diff2.norm() * 1000) / 1000;
  REQUIRE(dist_err1 <= 0.03); // less than 3 cm
  REQUIRE(dist_err2 <= 0.03);
  REQUIRE(measurement1_valid == true);
  REQUIRE(measurement2_valid == true);

  // ------------------------------
  // Below is some test to calculate the difference betwwen true rotation and
  // estimated transform. The issue is the yaw rotation will be off since it's
  // not constrained

  // Eigen::Vector3d ypr_measured1, ypr_true1;
  // ypr_measured1 = measurement1.rotation().eulerAngles(2, 1, 0);
  // ypr_true1 = T_SCAN_TARGET1_TRUE.rotation().eulerAngles(2, 1, 0);
  //
  // Eigen::Vector2d rot_diff1(ypr_measured1(0) - ypr_true1(0),
  //                           ypr_measured1(1) - ypr_true1(1));
  // std::cout << "ypr_measured1:\n " << ypr_measured1 << "\n"
  //           << "ypr_true1:\n " << ypr_true1 << "\n";

  // Eigen::Vector3d ypr_measured2, ypr_true2;
  // ypr_measured2 = measurement2.rotation().eulerAngles(2, 1, 0);
  // ypr_true2 = T_SCAN_TARGET2_TRUE.rotation().eulerAngles(2, 1, 0);
  // Eigen::Vector2d rot_diff2(ypr_measured2(1) - ypr_true2(1),
  //                           ypr_measured2(2) - ypr_true2(2));
  // double rot_err1 = std::round(rot_diff1.norm() * 10000) / 10000;
  // double rot_err2 = std::round(rot_diff2.norm() * 1000) / 1000;

  // std::cout << "\ndist_err1: " << dist_err1
  //           << "\ndist_err2: " << dist_err2
  //           << "\nrot_err1: " << rot_err1
  //           << "\nrot_err2: " << rot_err2 << "\n";

  // REQUIRE(rot_err1 <= 0.3); // less than 0.3 rad
  // REQUIRE(rot_err2 <= 0.3);
  // -----------------------------
}

TEST_CASE("Test auto rejection") {
  // FileSetup();
  // LoadTransforms();
  // LoadTemplateCloud(template_cloud_path);
  // LoadSimulatedCloud();
  LoadRegistrationParams();
  LoadTargetParamsRotated();
  // registration_params.show_transform = true;
  registration_params.crop_threshold_x = 1;
  registration_params.crop_threshold_y = 1;
  registration_params.crop_threshold_z = 1;
  std::pair<Eigen::Affine3d, bool> measurement_info;
  vicon_calibration::LidarCylExtractor cyl_extractor;
  cyl_extractor.SetRegistrationParams(registration_params);
  cyl_extractor.SetTargetParams(target_params);
  cyl_extractor.SetScan(sim_cloud);
  cyl_extractor.ExtractCylinder(T_SCAN_TARGET1_EST, 3);
  measurement_info = cyl_extractor.GetMeasurementInfo();
  REQUIRE(measurement_info.second == false);
}
