#define CATCH_CONFIG_MAIN

#include "vicon_calibration/JsonTools.h"
#include "vicon_calibration/measurement_extractors/DiamondCameraExtractor.h"
#include "vicon_calibration/utils.h"

#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace vicon_calibration;

std::string image_path;
std::string intrinsic_path;
cv::Mat image;
std::shared_ptr<CameraExtractor> diamond_extractor;
std::shared_ptr<TargetParams> target_params;
std::shared_ptr<CameraParams> camera_params;

Eigen::Matrix4d T_SENSOR_TARGET;
Eigen::Affine3d TA_SENSOR_TARGET;

void SetUp() {
  diamond_extractor = std::make_shared<DiamondCameraExtractor>();
  image_path = utils::GetFilePathTestData("ig_f1_sim_dia.jpg");
  intrinsic_path = utils::GetFilePathTestData("F1_sim.json");
  std::string template_cloud_path =
      utils::GetFilePathTestClouds("diamond_target_simulation.pcd");
  std::string target_config_path =
      utils::GetFilePathTestData("DiamondTargetSim.json");
  image = cv::imread(image_path);

  Eigen::Vector3d t_SENSOR_TARGET(-0.357, -0.272, 2.2070);
  Eigen::Quaterniond q;
  q.x() = 0.033;
  q.y() = -0.054;
  q.z() = 0.002;
  q.w() = 0.998;
  T_SENSOR_TARGET.setIdentity();
  T_SENSOR_TARGET.block(0, 3, 3, 1) = t_SENSOR_TARGET;
  Eigen::Matrix3d R = q.toRotationMatrix();
  T_SENSOR_TARGET.block(0, 0, 3, 3) = R;
  TA_SENSOR_TARGET.matrix() = T_SENSOR_TARGET;

  Eigen::Vector2d crop_image(20, 20);
  pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud_path, *template_cloud);

  JsonTools json_loader;
  target_params = json_loader.LoadTargetParams(target_config_path);
  target_params->extractor_type = "DIAMOND";
  target_params->target_config_path = target_config_path;
  target_params->crop_image = crop_image;
  target_params->template_cloud = template_cloud;

  camera_params = std::make_shared<CameraParams>();
  camera_params->intrinsics = intrinsic_path;
  camera_params->images_distorted = true;

  diamond_extractor->SetTargetParams(target_params);
  diamond_extractor->SetShowMeasurements(false);
  diamond_extractor->SetCameraParams(camera_params);
}

TEST_CASE("Test extracting diamond with invalid image") {
  SetUp();
  cv::Mat invalid_image;
  REQUIRE_THROWS(
      diamond_extractor->ExtractKeypoints(T_SENSOR_TARGET, invalid_image));
}

TEST_CASE("Test extracting diamond with invalid transformation matrix") {
  Eigen::Affine3d TA_INVALID;
  REQUIRE_THROWS(
      diamond_extractor->ExtractKeypoints(TA_INVALID.matrix(), image));
}

TEST_CASE("Test extracting from a black image") {
  cv::Mat black_image(image.rows, image.cols, image.type(),
                      cv::Scalar(0, 0, 0));
  diamond_extractor->ExtractKeypoints(T_SENSOR_TARGET, black_image);
  REQUIRE(diamond_extractor->GetMeasurementValid() == false);
}

TEST_CASE("Test extracting diamond") {
  // SetUp();
  // diamond_extractor->SetShowMeasurements(true);
  diamond_extractor->ExtractKeypoints(T_SENSOR_TARGET, image);
  REQUIRE(diamond_extractor->GetMeasurementValid() == true);
  // auto errors = diamond_extractor->GetErrors();
  // REQUIRE(errors.first <= 300);
  // REQUIRE(errors.second <= 0.05);
}

TEST_CASE("Test extracting diamond with invalid cropping") {
  // SetUp();
  Eigen::Vector2d crop_image_invalid(400, 400);
  target_params->crop_image = crop_image_invalid;
  diamond_extractor->SetTargetParams(target_params);
  // diamond_extractor->SetShowMeasurements(true);
  diamond_extractor->ExtractKeypoints(T_SENSOR_TARGET, image);
  bool measurement_valid = diamond_extractor->GetMeasurementValid();
  Eigen::Vector2d crop_image_valid(20, 20);
  target_params->crop_image = crop_image_valid;
  diamond_extractor->SetTargetParams(target_params);
  REQUIRE(measurement_valid == false);
}
