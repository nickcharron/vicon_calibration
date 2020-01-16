#define CATCH_CONFIG_MAIN

#include "vicon_calibration/measurement_extractors/CylinderCameraExtractor.h"

#include <catch2/catch.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>

std::string image_path;
std::string intrinsic_path;
cv::Mat image;
std::shared_ptr<vicon_calibration::CameraExtractor> cyl_extractor;
std::shared_ptr<vicon_calibration::TargetParams> target_params;
std::shared_ptr<vicon_calibration::CameraParams> camera_params;

Eigen::Matrix4d T_SENSOR_TARGET;
Eigen::Affine3d TA_SENSOR_TARGET;

void SetUp() {
  cyl_extractor = std::make_shared<vicon_calibration::CylinderCameraExtractor>();
  std::string current_file = "tests/cylinder_camera_extractor_test.cpp";
  std::string test_path = __FILE__;
  test_path.erase(test_path.end() - current_file.size(), test_path.end());
  image_path = test_path + "tests/data/ig_f1_sim_cyl.jpg";
  intrinsic_path = test_path + "tests/data/F1_sim.json";
  std::string template_cloud_path =
      test_path + "tests/template_pointclouds/cylinder_target.pcd";
  std::string target_config_path = test_path + "tests/data/CylinderTarget.json";
  image = cv::imread(image_path);

  Eigen::Vector3d t_SENSOR_TARGET(0, -0.395, 2.570);
  T_SENSOR_TARGET.setIdentity();
  T_SENSOR_TARGET.block(0, 3, 3, 1) = t_SENSOR_TARGET;
  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(1.570796, Eigen::Vector3d::UnitZ());
  T_SENSOR_TARGET.block(0, 0, 3, 3) = R;
  TA_SENSOR_TARGET.matrix() = T_SENSOR_TARGET;

  Eigen::Vector2d crop_image(600, 400);
  pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud_path, *template_cloud);

  target_params = std::make_shared<vicon_calibration::TargetParams>();
  target_params->extractor_type = "CYLINDER";
  target_params->target_config_path = target_config_path;
  target_params->crop_image = crop_image;
  target_params->template_cloud = template_cloud;

  camera_params = std::make_shared<vicon_calibration::CameraParams>();
  camera_params->intrinsics = intrinsic_path;
  camera_params->images_distorted = true;

  cyl_extractor->SetTargetParams(target_params);
  cyl_extractor->SetShowMeasurements(false);
  cyl_extractor->SetCameraParams(camera_params);
}

TEST_CASE("Test extracting cylinder with invalid image") {
  SetUp();
  cv::Mat invalid_image;
  REQUIRE_THROWS(
      cyl_extractor->ExtractKeypoints(TA_SENSOR_TARGET.matrix(), invalid_image));
}

TEST_CASE("Test extracting cylinder with invalid transformation matrix") {
  Eigen::Affine3d TA_INVALID;
  REQUIRE_THROWS(cyl_extractor->ExtractKeypoints(TA_INVALID.matrix(), image));
}

TEST_CASE("Test extracting from a black image") {
  cv::Mat black_image(image.rows, image.cols, image.type(), cv::Scalar(0,0,0));
  cyl_extractor->ExtractKeypoints(TA_SENSOR_TARGET.matrix(), black_image);
  REQUIRE(cyl_extractor->GetMeasurementValid() == false);
}

TEST_CASE("Test extracting cylinder") {
  //SetUp();
  // cyl_extractor->SetShowMeasurements(true);
  cyl_extractor->ExtractKeypoints(TA_SENSOR_TARGET.matrix(), image);
  REQUIRE(cyl_extractor->GetMeasurementValid() == true);
  auto errors = cyl_extractor->GetErrors();
  REQUIRE(errors.first <= 300);
  REQUIRE(errors.second <= 0.05);
}

// TEST_CASE("Test extracting cylinder with invalid cropping") {
//   // SetUp();
//   Eigen::Vector2d crop_image_invalid(-100, -100);
//   target_params->crop_image = crop_image_invalid;
//   cyl_extractor->SetTargetParams(target_params);
//   //cyl_extractor->SetShowMeasurements(true);
//   cyl_extractor->ExtractKeypoints(TA_SENSOR_TARGET.matrix(), image);
//   bool measurement_valid = cyl_extractor->GetMeasurementValid();
//   Eigen::Vector2d crop_image_valid(600, 400);
//   target_params->crop_image = crop_image_valid;
//   cyl_extractor->SetTargetParams(target_params);
//   REQUIRE(measurement_valid == false);
// }
