#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "vicon_calibration/CamCylExtractor.h"
#include <string>

#include <opencv2/opencv.hpp>

std::string current_file = "tests/cam_cyl_extract_tests.cpp";
std::string test_path = __FILE__;
std::string image_path;
std::string intrinsic_path;
cv::Mat image;
vicon_calibration::CamCylExtractor cyl_extractor;

Eigen::Matrix4d TA_SENSOR_TARGET;
Eigen::Affine3d T_SENSOR_TARGET;

void SetUp() {
  test_path.erase(test_path.end() - current_file.size(), test_path.end());
  image_path = test_path + "tests/data/vicon1.png";
  image = cv::imread(image_path);

  TA_SENSOR_TARGET << 1, 0, 0, -0.380727, 0, 1, 0, 0, 0, 0, 1, 0.634802, 0, 0,
      0, 1;
  T_SENSOR_TARGET.matrix() = TA_SENSOR_TARGET;

  intrinsic_path = test_path + "data/F2.json";
  cyl_extractor.ConfigureCameraModel(intrinsic_path);
}

TEST_CASE("Test extracting cylinder with invalid image") {
  SetUp();
  cv::Mat invalid_image;

  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(T_SENSOR_TARGET, invalid_image));
}

TEST_CASE("Test extracting cylinder with invalid transformation matrix") {
  Eigen::Affine3d TA_INVALID;

  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(TA_INVALID, image));
}

TEST_CASE("Test invalid cylinder parameters") {
  double default_radius = 0.0635, default_height = 0.5, default_offset = 0.15;

  REQUIRE_THROWS(cyl_extractor.SetCylinderDimension(0, default_height));
  REQUIRE_THROWS(cyl_extractor.SetCylinderDimension(default_radius, 0));
  cyl_extractor.SetCylinderDimension(default_radius, default_height);

  REQUIRE_THROWS(cyl_extractor.SetOffset(-0.0635));
  cyl_extractor.SetOffset(default_offset);
}

TEST_CASE("Test extracting from a black image") {
  cv::Mat black_image(image.rows, image.cols, image.type(), cv::Scalar(0, 0, 0));
  cyl_extractor.ExtractCylinder(T_SENSOR_TARGET, black_image);
  auto result = cyl_extractor.GetMeasurementInfo();
  REQUIRE(result.second == false);
}

TEST_CASE("Test extracting cylinder with small offset that edges which meet "
          "the validation criteria are not detected") {
  double default_offset = 0.15;
  cyl_extractor.SetOffset(0);
  cyl_extractor.ExtractCylinder(T_SENSOR_TARGET, image);
  auto result = cyl_extractor.GetMeasurementInfo();
  REQUIRE(result.second == false);
  cyl_extractor.SetOffset(default_offset);
}

TEST_CASE("Test extracting cylinder") {
  cyl_extractor.SetAcceptanceCriteria(100, 0.5);
  cyl_extractor.ExtractCylinder(T_SENSOR_TARGET, image);
  auto result = cyl_extractor.GetMeasurementInfo();
  REQUIRE(result.second == true);
  auto errors = cyl_extractor.GetErrors();
  REQUIRE(errors.first <= 100);
  REQUIRE(errors.second <= 0.5);
}
