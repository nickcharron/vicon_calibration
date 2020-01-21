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
bool show_measurements{true};

Eigen::Matrix4d T_SENSOR_TARGET, T_SENSOR_TARGET_PERT1, T_SENSOR_TARGET_PERT2;

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
  Eigen::VectorXd perturbation1(6);
  perturbation1 << 0.2, -0.3, 0, 0.07, 0.05, 0;
  T_SENSOR_TARGET_PERT1 =
      utils::PerturbTransform(T_SENSOR_TARGET, perturbation1);
  Eigen::VectorXd perturbation2(6);
  perturbation2 << 0, 0, 0, 0.06, 0.04, 0;
  T_SENSOR_TARGET_PERT2 =
      utils::PerturbTransform(T_SENSOR_TARGET, perturbation2);

  pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud_path, *template_cloud);

  JsonTools json_loader;
  target_params = json_loader.LoadTargetParams(target_config_path);
  target_params->template_cloud = template_cloud;

  camera_params = std::make_shared<CameraParams>();
  camera_params->intrinsics = intrinsic_path;
  camera_params->images_distorted = true;

  diamond_extractor->SetTargetParams(target_params);
  diamond_extractor->SetShowMeasurements(show_measurements);
  diamond_extractor->SetCameraParams(camera_params);
}

TEST_CASE("Test extracting diamond with invalid image") {
  SetUp();
  cv::Mat invalid_image;
  REQUIRE_THROWS(
      diamond_extractor->ProcessMeasurement(T_SENSOR_TARGET, invalid_image));
}

/*
TEST_CASE("Test extracting diamond with invalid transformation matrix") {
  Eigen::Affine3d TA_INVALID;
  REQUIRE_THROWS(
      diamond_extractor->ProcessMeasurement(TA_INVALID.matrix(), image));
}

TEST_CASE("Test extracting from a black image") {
  cv::Mat black_image(image.rows, image.cols, image.type(),
                      cv::Scalar(0, 0, 0));
  diamond_extractor->ProcessMeasurement(T_SENSOR_TARGET, black_image);
  REQUIRE(diamond_extractor->GetMeasurementValid() == false);
}


TEST_CASE("Test extracting diamond") {
  // SetUp();
  // diamond_extractor->SetShowMeasurements(true);
  diamond_extractor->ProcessMeasurement(T_SENSOR_TARGET, image);
  REQUIRE(diamond_extractor->GetMeasurementValid() == true);
  diamond_extractor->ProcessMeasurement(T_SENSOR_TARGET_PERT1, image);
  REQUIRE(diamond_extractor->GetMeasurementValid() == true);
  diamond_extractor->ProcessMeasurement(T_SENSOR_TARGET_PERT2, image);
  REQUIRE(diamond_extractor->GetMeasurementValid() == true);
}
*/

TEST_CASE("Test extracting diamond with invalid cropping") {
  // SetUp();
  std::shared_ptr<TargetParams> invalid_target_params =
      std::make_shared<TargetParams>();
  *invalid_target_params = *target_params;
  diamond_extractor->SetTargetParams(invalid_target_params);
  bool measurement_valid;

  // case 1: crop out 100% of target. Calculated area should be below min
  invalid_target_params->crop_image = Eigen::Vector2d(-100, -100);
  diamond_extractor->ProcessMeasurement(T_SENSOR_TARGET, image);
  measurement_valid = diamond_extractor->GetMeasurementValid();
  REQUIRE(measurement_valid == false);

  // case 2: this should create a cropbox that is outside the image plane
  invalid_target_params->crop_image = Eigen::Vector2d(500, 500);
  diamond_extractor->ProcessMeasurement(T_SENSOR_TARGET, image);
  measurement_valid = diamond_extractor->GetMeasurementValid();
  REQUIRE(measurement_valid == false);

  // case 3: this should create a cropbox that is inside the image plane but
  // smaller than the target
  invalid_target_params->crop_image = Eigen::Vector2d(-30, -30);
  diamond_extractor->ProcessMeasurement(T_SENSOR_TARGET, image);
  measurement_valid = diamond_extractor->GetMeasurementValid();
  REQUIRE(measurement_valid == true);

  // reset
  diamond_extractor->SetTargetParams(target_params);
}
