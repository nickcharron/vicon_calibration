#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <vicon_calibration/JsonTools.h>
#include <vicon_calibration/measurement_extractors/CheckerboardCameraExtractor.h>
#include <vicon_calibration/Utils.h>

using namespace vicon_calibration;

std::string image_path;
std::string intrinsic_path;
cv::Mat image;
std::shared_ptr<CameraExtractor> checkerboard_extractor;
std::shared_ptr<TargetParams> target_params;
std::shared_ptr<CameraParams> camera_params;
bool show_measurements{false};
bool is_setup{false};

Eigen::Matrix4d T_SENSOR_TARGET, T_SENSOR_TARGET_PERT1, T_SENSOR_TARGET_PERT2;

void SetUp() {
  if (is_setup) { return; }
  is_setup = true;
  checkerboard_extractor = std::make_shared<CheckerboardCameraExtractor>();
  image_path = utils::GetFilePathTestData("ig_f1_sim_dia.jpg");
  intrinsic_path = utils::GetFilePathTestData("F1_sim.json");
  std::string template_cloud_path =
      utils::GetFilePathTestClouds("checkerboard_target_simulation.pcd");
  std::string target_config_path =
      utils::GetFilePathTestData("CheckerboardTargetSim.json");
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
      utils::PerturbTransformDegM(T_SENSOR_TARGET, perturbation1);
  Eigen::VectorXd perturbation2(6);
  perturbation2 << 0, 0, 0, 0.06, 0.04, 0;
  T_SENSOR_TARGET_PERT2 =
      utils::PerturbTransformDegM(T_SENSOR_TARGET, perturbation2);

  PointCloud::Ptr template_cloud = std::make_shared<PointCloud>();
  pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud_path, *template_cloud);

  CalibratorInputs inputs;
  inputs.target_config_path = target_config_path;
  JsonTools json_loader(inputs);
  target_params = json_loader.LoadTargetParams(target_config_path);
  target_params->template_cloud = template_cloud;

  camera_params = std::make_shared<CameraParams>(intrinsic_path);

  checkerboard_extractor->SetTargetParams(target_params);
  checkerboard_extractor->SetShowMeasurements(show_measurements);
  checkerboard_extractor->SetCameraParams(camera_params);
}

TEST_CASE("Test extracting checkerboard with invalid image") {
  SetUp();
  cv::Mat invalid_image;
  REQUIRE_THROWS(
      checkerboard_extractor->ProcessMeasurement(T_SENSOR_TARGET, invalid_image));
}

TEST_CASE("Test extracting checkerboard with invalid transformation matrix") {
  SetUp();
  Eigen::Affine3d TA_INVALID;
  REQUIRE_THROWS(
      checkerboard_extractor->ProcessMeasurement(TA_INVALID.matrix(), image));
}

TEST_CASE("Test extracting from a black image") {
  SetUp();
  cv::Mat black_image(image.rows, image.cols, image.type(),
                      cv::Scalar(0, 0, 0));
  checkerboard_extractor->ProcessMeasurement(T_SENSOR_TARGET, black_image);
  REQUIRE(checkerboard_extractor->GetMeasurementValid() == false);
}

TEST_CASE("Test extracting checkerboard") {
  SetUp();
  // checkerboard_extractor->SetShowMeasurements(true);
  checkerboard_extractor->ProcessMeasurement(T_SENSOR_TARGET, image);
  REQUIRE(checkerboard_extractor->GetMeasurementValid() == true);
  image = cv::imread(image_path);
  checkerboard_extractor->ProcessMeasurement(T_SENSOR_TARGET_PERT1, image);
  REQUIRE(checkerboard_extractor->GetMeasurementValid() == true);
  image = cv::imread(image_path);
  checkerboard_extractor->ProcessMeasurement(T_SENSOR_TARGET_PERT2, image);
  REQUIRE(checkerboard_extractor->GetMeasurementValid() == true);
}

TEST_CASE("Test extracting checkerboard with invalid cropping") {
  SetUp();
  std::shared_ptr<TargetParams> invalid_target_params =
      std::make_shared<TargetParams>();
  *invalid_target_params = *target_params;
  checkerboard_extractor->SetTargetParams(invalid_target_params);
  bool measurement_valid;

  // case 1: crop out 100% of target. Calculated area should be below min
  invalid_target_params->crop_image = Eigen::Vector2d(-100, -100);
  image = cv::imread(image_path);
  checkerboard_extractor->ProcessMeasurement(T_SENSOR_TARGET, image);
  measurement_valid = checkerboard_extractor->GetMeasurementValid();
  REQUIRE(measurement_valid == false);

  // case 2: this should create a cropbox that is outside the image plane
  invalid_target_params->crop_image = Eigen::Vector2d(500, 500);
  image = cv::imread(image_path);
  checkerboard_extractor->ProcessMeasurement(T_SENSOR_TARGET, image);
  measurement_valid = checkerboard_extractor->GetMeasurementValid();
  REQUIRE(measurement_valid == false);

  // case 3: this should create a cropbox that is inside the image plane but
  // smaller than the target
  invalid_target_params->crop_image = Eigen::Vector2d(-30, -30);
  image = cv::imread(image_path);
  checkerboard_extractor->ProcessMeasurement(T_SENSOR_TARGET, image);
  measurement_valid = checkerboard_extractor->GetMeasurementValid();
  REQUIRE(measurement_valid == true);

  // reset
  checkerboard_extractor->SetTargetParams(target_params);
}
