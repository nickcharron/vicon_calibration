#include "vicon_calibration/CamCylExtractor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/TransformStamped.h>
#include <image_transport/image_transport.h>
#include <ros/time.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2/buffer_core.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_msgs/TFMessage.h>

#include <beam_calibration/TfTree.h>
#include <beam_utils/log.hpp>
#include <beam_utils/math.hpp>

std::string current_file = "tests/cam_cyl_extract_tests.cpp";

std::string bag_file, initial_calibration_file, vicon_baselink_frame, encoding;
double camera_time_steps, target_radius, target_height, target_crop_threshold;
bool show_camera_measurements;
std::vector<std::string> image_topics, image_frames, intrinsics,
    vicon_target_frames;

vicon_calibration::CamCylExtractor camera_extractor;

std::string GetJSONFileNameData(std::string file_name) {
  std::string file_location = __FILE__;
  file_location.erase(file_location.end() - current_file.size(),
                      file_location.end());
  file_location += "data/";
  file_location += file_name;
  return file_location;
}

std::string GetJSONFileNameConfig(std::string file_name) {
  std::string file_location = __FILE__;
  file_location.erase(file_location.end() - current_file.size(),
                      file_location.end());
  file_location += "config/";
  file_location += file_name;
  return file_location;
}

void LoadJson(std::string file_name) {
  nlohmann::json J;
  std::ifstream file(file_name);
  file >> J;

  bag_file = J["bag_file"];
  initial_calibration_file = J["initial_calibration"];
  vicon_baselink_frame = J["vicon_baselink_frame"];

  for (const auto &camera_info : J["camera_info"]) {
    show_camera_measurements = camera_info.at("show_camera_measurements");
    camera_time_steps = camera_info.at("camera_time_steps");
    for (const auto &topic : camera_info.at("image_topics")) {
      image_topics.push_back(topic.get<std::string>());
    }
    for (const auto &frame : camera_info.at("image_frames")) {
      image_frames.push_back(frame.get<std::string>());
    }
    for (const auto &intrinsic : camera_info.at("intrinsics")) {
      intrinsics.push_back(intrinsic.get<std::string>());
    }
    encoding = camera_info.at("encoding");
  }

  for (const auto &target_info : J["target_info"]) {
    target_radius = target_info.at("radius");
    target_height = target_info.at("height");
    target_crop_threshold = target_info.at("crop_threshold");
    for (const auto &frame : target_info.at("vicon_target_frames")) {
      vicon_target_frames.push_back(frame.get<std::string>());
    }
  }
}

std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
GetInitialGuess(rosbag::Bag &bag, ros::Time &time, std::string &sensor_frame) {
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      T_sensor_tgts_estimated;

  // check 2 second time window for vicon baselink transform
  ros::Duration time_window(1);
  ros::Time start_time = time - time_window;
  ros::Time time_zero(0, 0);
  if (start_time <= time_zero) {
    start_time = time_zero;
  }
  ros::Time end_time = time + time_window;
  rosbag::View view(bag, rosbag::TopicQuery("/tf"), start_time, end_time, true);

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

  // Add all vicon transforms in window
  for (const auto &msg_instance : view) {
    auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
    if (tf_message != nullptr) {
      for (const geometry_msgs::TransformStamped &tf : tf_message->transforms) {
        tree.AddTransform(tf);
      }
    }
  }

  geometry_msgs::TransformStamped T_SENSOR_TGTn_msg;
  // get transform to each of the targets at specified time
  for (uint8_t n; n < vicon_target_frames.size(); n++) {

    try {
      T_SENSOR_TGTn_msg =
          tree.GetTransform(sensor_frame, vicon_target_frames[n], time);
    } catch (const std::exception &e) {
      LOG_ERROR("Error Getting a transform, continue with the next transform: %s", e.what());
      continue;
    }
    //std::cout << T_SENSOR_TGTn_msg << std::endl;
    Eigen::Affine3d T_SENSOR_TGTn = tf2::transformToEigen(T_SENSOR_TGTn_msg);
    std::cout << T_SENSOR_TGTn.matrix() << std::endl;
    T_sensor_tgts_estimated.push_back(T_SENSOR_TGTn);
  }
  return T_sensor_tgts_estimated;
}

void GetImageMeasurements(rosbag::Bag &bag, std::string &topic,
                           std::string &frame) {
  rosbag::View view(bag, ros::TIME_MIN, ros::TIME_MAX, true);

  ros::Duration time_step(camera_time_steps);
  ros::Time time_last(0, 0);
  time_last = time_last + time_step;
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      T_camera_tgts_estimated;

  for (auto iter = view.begin(); iter != view.end(); iter++) {
    if (iter->getTopic() == topic) {
      boost::shared_ptr<sensor_msgs::Image> image_msg =
          iter->instantiate<sensor_msgs::Image>();
      cv_bridge::CvImagePtr cv_img_ptr;

      if (image_msg->header.stamp > time_last + time_step) {
        time_last = image_msg->header.stamp;
        cv_img_ptr = cv_bridge::toCvCopy(image_msg, encoding);

        try {
          T_camera_tgts_estimated =
              GetInitialGuess(bag, image_msg->header.stamp, frame);
        } catch (const std::exception &err) {
          LOG_ERROR("%s", err);
          std::cout
              << "Possible reasons for lookup error: \n"
              << "- Start or End of bag could have message timing issues\n"
              << "- Vicon messages not synchronized with robot's ROS time\n"
              << "- Invalid initial calibrations, i.e. input transformations "
                 "json has missing/invalid transforms\n";
          continue;
        }
        // here
        bool measurement_valid;
        Eigen::Vector3d measurement;
        for (uint8_t n = 0; n < T_camera_tgts_estimated.size(); n++) {
          camera_extractor.ExtractCylinder(T_camera_tgts_estimated[n],
                                           cv_img_ptr->image, n);
          //const auto measurement_info = camera_extractor.GetMeasurementInfo();
          //measurement = measurement_info.first;
          //measurement_valid = measurement_info.second;
        }
      }
    }
  }
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

  for(intrinsic : intrinsics) {
    auto intrinsic_filename = GetJSONFileNameData(intrinsic);
    camera_extractor.ConfigureCameraModel(intrinsic_filename);
  }

  // main loop
  for (uint8_t k = 0; k < image_topics.size(); k++) {
    GetImageMeasurements(bag, image_topics[k], image_frames[k]);
    // GetImageMeasurements(bag, image_topics[k], image_frames[k]);
  }
  bag.close();
  /*
  /// Load an image
  src = imread(argv[1]);

  if (!src.data) {
    return -1;
  }

  /// Create a matrix of the same type and size as src (for dst)
  dst.create(src.size(), src.type());

  /// Convert the image to grayscale
  cvtColor(src, src_gray, CV_BGR2GRAY);

  /// Create a window
  namedWindow(window_name, CV_WINDOW_AUTOSIZE);

  /// Create a Trackbar for user to enter threshold
  createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold,
                 CannyThreshold);

  /// Show the image
  CannyThreshold(0, 0);

  /// Wait until user exit program by pressing a key
  waitKey(0);*/

  return 0;
}
