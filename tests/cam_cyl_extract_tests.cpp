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
  // image_path = test_path + "tests/data/vicon1.png";  // this is for the real
  // image (non-sim)
  // intrinsic_path = test_path + "tests/data/F2.json";
  image_path = test_path + "tests/data/ig_f1_sim.jpg";
  intrinsic_path = test_path + "tests/data/F1_sim.json";
  std::string template_cloud_path =
      test_path + "tests/template_pointclouds/cylinder_target.pcd";
  image = cv::imread(image_path);

  // Eigen::Vector3d t_SENSOR_TARGET(-0.380727, 0, 0.634802); // this is for the
  // real image (non-sim)
  Eigen::Vector3d t_SENSOR_TARGET(0, -0.395, 2.570);
  T_SENSOR_TARGET.setIdentity();
  T_SENSOR_TARGET.block(0, 3, 3, 1) = t_SENSOR_TARGET;
  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(1.570796, Eigen::Vector3d::UnitZ());
  T_SENSOR_TARGET.block(0, 0, 3, 3) = R;
  TA_SENSOR_TARGET.matrix() = T_SENSOR_TARGET;
  // std::vector<uint8_t> color_threshold_min{0, 0, 0};  // black target
  // std::vector<uint8_t> color_threshold_max{30, 30, 30}; // black target
  std::vector<uint8_t> color_threshold_min{0, 95, 0};
  std::vector<uint8_t> color_threshold_max{20, 255, 20};
  target_params.radius = 0.0635;
  target_params.height = 0.5;
  target_params.color_threshold_min = color_threshold_min;
  target_params.color_threshold_max = color_threshold_max;
  target_params.template_cloud = template_cloud_path;

  processing_params.dist_criteria = 100;
  processing_params.rot_criteria = 0.5;
  processing_params.show_measurements = true;
  processing_params.crop_threshold_u = 600;
  processing_params.crop_threshold_v = 400;

  camera_params.intrinsics = intrinsic_path;
  camera_params.images_distorted = false;

  cyl_extractor.SetTargetParams(target_params);
  cyl_extractor.SetImageProcessingParams(processing_params);
  cyl_extractor.SetCameraParams(camera_params);
}

TEST_CASE("Test extracting cylinder with invalid image") {
  SetUp();
  cv::Mat invalid_image;
  REQUIRE_THROWS(cyl_extractor.ExtractMeasurement(TA_SENSOR_TARGET,
invalid_image));
}

TEST_CASE("Test extracting cylinder with invalid transformation matrix") {
  Eigen::Affine3d TA_INVALID;
  REQUIRE_THROWS(cyl_extractor.ExtractMeasurement(TA_INVALID, image));
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
  cv::Mat black_image(image.rows, image.cols, image.type(), cv::Scalar(0,0,0));
  cyl_extractor.ExtractMeasurement(TA_SENSOR_TARGET, black_image);
  REQUIRE(cyl_extractor.GetMeasurementsValid() == false);
}

TEST_CASE("Test extracting cylinder with small offset that edges which meet "
          "the validation criteria are not detected") {
  // SetUp();
  processing_params.crop_threshold_u = 0;
  processing_params.crop_threshold_v = 0;
  cyl_extractor.SetImageProcessingParams(processing_params);
  cyl_extractor.SetTargetParams(target_params);
  cyl_extractor.ExtractMeasurement(TA_SENSOR_TARGET, image);
  processing_params.crop_threshold_u = 600;
  processing_params.crop_threshold_v = 400;
  cyl_extractor.SetImageProcessingParams(processing_params);
  REQUIRE(cyl_extractor.GetMeasurementsValid() == false);
}

TEST_CASE("Test extracting cylinder") {
  //SetUp();
  cyl_extractor.SetImageProcessingParams(processing_params);
  cyl_extractor.ExtractMeasurement(TA_SENSOR_TARGET, image);
  REQUIRE(cyl_extractor.GetMeasurementsValid() == true);
  auto errors = cyl_extractor.GetErrors();
  REQUIRE(errors.first <= 100);
  REQUIRE(errors.second <= 0.5);
}
