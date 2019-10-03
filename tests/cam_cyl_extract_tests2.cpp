#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "vicon_calibration/CamCylExtractor.h"
#include <string>

#include <opencv2/opencv.hpp>

std::string image_path;
std::string intrinsic_path;
cv::Mat image;
vicon_calibration::CamCylExtractor cyl_extractor;
vicon_calibration::CylinderTgtParams target_params;
vicon_calibration::ImageProcessingParams processing_params;
vicon_calibration::CameraParams camera_params;

Eigen::Matrix4d T_SENSOR_TARGET;
Eigen::Affine3d TA_SENSOR_TARGET;

void SetUp() {
  std::string current_file = "tests/cam_cyl_extract_tests.cpp";
  std::string test_path = __FILE__;
  test_path.erase(test_path.end() - current_file.size(), test_path.end());
  image_path = test_path + "tests/data/vicon1.png";
  intrinsic_path = test_path + "data/F2.json";
  image = cv::imread(image_path);

  Eigen::Vector3d t_SENSOR_TARGET(-0.380727, 0, 0.634802);
  T_SENSOR_TARGET.setIdentity();
  T_SENSOR_TARGET.block(0,3,3,1) = t_SENSOR_TARGET;
  TA_SENSOR_TARGET.matrix() = T_SENSOR_TARGET;

  target_params.radius = 0.0635;
  target_params.height = 0.5;

  processing_params.num_intersections = 300;
  processing_params.min_length_ratio = 0.5;
  processing_params.max_gap_ratio = 0.078;
  processing_params.canny_ratio = 0.1;
  processing_params.cropbox_offset = 0.15;
  processing_params.dist_criteria = 5;
  processing_params.rot_criteria = 0.1;
  processing_params.show_measurements = false;

  camera_params.intrinsics = intrinsic_path;
  camera_params.images_distorted = true;

  cyl_extractor.SetTargetParams(target_params);
  cyl_extractor.SetImageProcessingParams(processing_params);
  cyl_extractor.SetCameraParams(camera_params);
}

TEST_CASE("Test extracting cylinder with invalid image") {
  SetUp();
  cv::Mat invalid_image;
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(TA_SENSOR_TARGET, invalid_image));
}

TEST_CASE("Test extracting cylinder with invalid transformation matrix") {
  Eigen::Affine3d TA_INVALID;
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(TA_INVALID, image));
}

TEST_CASE("Test invalid cylinder parameters") {
  vicon_calibration::CylinderTgtParams target_params_invalid;
  target_params_invalid = target_params;
  target_params_invalid.radius = 0;
  REQUIRE_THROWS(cyl_extractor.SetTargetParams(target_params_invalid));
  target_params_invalid = target_params;
  target_params_invalid.height = 0;
  REQUIRE_THROWS(cyl_extractor.SetTargetParams(target_params_invalid));
}

TEST_CASE("Test extracting from a black image") {
  cv::Mat black_image(image.rows, image.cols, image.type(), cv::Scalar(0, 0, 0));
  cyl_extractor.ExtractCylinder(TA_SENSOR_TARGET, black_image);
  auto result = cyl_extractor.GetMeasurementInfo();
  REQUIRE(result.second == false);
}

TEST_CASE("Test extracting cylinder with small offset that edges which meet "
          "the validation criteria are not detected") {
  processing_params.cropbox_offset = 0;
  processing_params.show_measurements = true;
  cyl_extractor.SetImageProcessingParams(processing_params);
  cyl_extractor.ExtractCylinder(TA_SENSOR_TARGET, image);
  auto result = cyl_extractor.GetMeasurementInfo();
  REQUIRE(result.second == false);
  processing_params.cropbox_offset = 0.15;
  cyl_extractor.SetImageProcessingParams(processing_params);
}

TEST_CASE("Test extracting cylinder") {
  processing_params.dist_criteria = 100;
  processing_params.rot_criteria = 0.5;
  processing_params.show_measurements = true;
  cyl_extractor.SetImageProcessingParams(processing_params);
  cyl_extractor.ExtractCylinder(TA_SENSOR_TARGET, image);
  auto result = cyl_extractor.GetMeasurementInfo();
  REQUIRE(result.second == true);
  auto errors = cyl_extractor.GetErrors();
  REQUIRE(errors.first <= 100);
  REQUIRE(errors.second <= 0.5);
}
