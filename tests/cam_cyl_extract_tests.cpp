#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "vicon_calibration/CamCylExtractor.h"
#include <string>

#include <opencv2/opencv.hpp>

std::string current_file = "tests/cam_cyl_extract_tests.cpp";
std::string test_path = __FILE__;
std::string image_path;
cv::Mat image;
vicon_calibration::CamCylExtractor cyl_extractor;

Eigen::Affine3d T_SENSOR_TARGET;
T_SENSOR_TARGET << 1, 0, 0, -0.380727,
                   0, 1, 0, 0,
                   0, 0, 1, 0.634802,
                   0, 0, 0, 1;

void FileSetup() {
  test_path.erase(test_path.end() - current_file.size(), test_path.end());
  image_path = test_path + "data/vicon1.png";
  image = cv::imread(image_path, CV_LOAD_IMAGE_ANYDEPTH);
}

TEST_CASE("Test extracting cylinder with invalid image") {
  cv::Mat invalid_image;

  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(T_SENSOR_TARGET, invalid_image));
}

TEST_CASE("Test extracting cylinder with invalid transformation matrix") {
  Eigen::Affine3d TA_INVALID;

  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(TA_INVALID, image));
}

TEST_CASE("Test invalid cylinder parameters") {
  double default_radius = 0.0635, default_height = 0.5, default_offset = 0.15;
  cyl_extractor.SetCylinderDimension(0, default_height);
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(T_SENSOR_TARGET, image));

  cyl_extractor.SetCylinderDimension(default_radius, 0);
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(T_SENSOR_TARGET, image));

  cyl_extractor.SetCylinderDimension(default_radius, default_height);
  cyl_extractor.SetOffset(-0.1);
  REQUIRE_THROWS(cyl_extractor.ExtractCylinder(T_SENSOR_TARGET, image));
}

TEST_CASE("Test extracting cylinder with small offset that edges are not detected") {
  double default_offset = 0.15;
  cyl_extractor.SetOffset(0);
  cyl_extractor.ExtractCylinder(T_SENSOR_TARGET, image);
}

int main(int argc, char **argv) {

  // get configuration settings
  std::string config_file;
  config_file = GetJSONFileNameConfig("test_ig.json");
  try {
    LoadJson(config_file);
  } catch (nlohmann::detail::parse_error &ex) {
    LOG_ERROR("Unable to load json config file: %s", config_file.c_str());
  }

  // load bag file
  rosbag::Bag bag;
  try {
    LOG_INFO("Opening bag: %s", bag_file.c_str());
    bag.open(bag_file, rosbag::bagmode::Read);
  } catch (rosbag::BagException &ex) {
    LOG_ERROR("Bag exception : %s", ex.what());
  }

  for (intrinsic : intrinsics) {
    LOG_INFO("Configuring intrinsics");
    auto intrinsic_filename = GetJSONFileNameData(intrinsic);
    camera_extractor.ConfigureCameraModel(intrinsic_filename);
  }

  camera_extractor.SetCylinderDimension(target_radius, target_height);
  camera_extractor.SetOffset(target_crop_threshold);
  camera_extractor.SetEdgeDetectionParameters(
      num_intersections, min_length_percent, max_gap_percent, canny_percent);
  camera_extractor.SetShowMeasurement(true);
  // initialize tree with initial calibrations:
  beam_calibration::TfTree tree;
  std::string initial_calibration_file_dir;
  try {
    initial_calibration_file_dir =
        GetJSONFileNameData(initial_calibration_file);
    tree.LoadJSON(initial_calibration_file_dir);
  } catch (nlohmann::detail::parse_error &ex) {
    LOG_ERROR("Unable to load json calibration file: %s",
              initial_calibration_file_dir.c_str());
  }

  // main loop
  for (uint8_t k = 0; k < image_topics.size(); k++) {
    GetImageMeasurements(bag, image_topics[k], image_frames[k], tree);
    // GetImageMeasurements(bag, image_topics[k], image_frames[k]);
  }
  bag.close();

  return 0;
}
