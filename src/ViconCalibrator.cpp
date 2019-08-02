#include "beam_calibration/TfTree.h"
#include "vicon_calibration/CamCylExtractor.h"
#include "vicon_calibration/GTSAMGraph.h"
#include "vicon_calibration/LidarCylExtractor.h"
#include "vicon_calibration/utils.hpp"
#include <Eigen/StdVector>
#include <beam_utils/log.hpp>
#include <beam_utils/math.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <tf2/buffer_core.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_msgs/TFMessage.h>

// ROS specific headers
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/time.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

// PCL specific headers
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>

typedef vicon_calibration::PointCloud PointCloud;
typedef vicon_calibration::PointCloudColor PointCloudColor;

std::string bag_file, initial_calibration_file, vicon_baselink_frame,
    target_cloud_name;
double camera_time_steps, lidar_time_steps, target_radius, target_height,
    target_crop_threshold;
bool show_camera_measurements, show_lidar_measurements;
vicon_calibration::LidarCylExtractor lidar_extractor;

double max_corr, t_eps, fit_eps;
int max_iter;
bool set_show_transform;

std::vector<std::string> image_topics, image_frames, intrinsics, lidar_topics,
    lidar_frames, vicon_target_frames;

std::string GetJSONFileNameData(std::string file_name) {
  std::string file_location = __FILE__;
  file_location.erase(file_location.end() - 23, file_location.end());
  file_location += "data/";
  file_location += file_name;
  return file_location;
}

std::string GetJSONFileNameConfig(std::string file_name) {
  std::string file_location = __FILE__;
  file_location.erase(file_location.end() - 23, file_location.end());
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
  }

  for (const auto &lidar_info : J["lidar_info"]) {
    show_lidar_measurements = lidar_info.at("show_lidar_measurements");
    lidar_time_steps = lidar_info.at("lidar_time_steps");
    for (const auto &topic : lidar_info.at("lidar_topics")) {
      lidar_topics.push_back(topic.get<std::string>());
    }
    for (const auto &frame : lidar_info.at("lidar_frames")) {
      lidar_frames.push_back(frame.get<std::string>());
    }
  }

  for (const auto &target_info : J["target_info"]) {
    target_radius = target_info.at("radius");
    target_height = target_info.at("height");
    target_crop_threshold = target_info.at("crop_threshold");
    std::string target_cloud_file = target_info.at("template_cloud_name");
    target_cloud_name = GetJSONFileNameData(target_cloud_file);
    for (const auto &frame : target_info.at("vicon_target_frames")) {
      vicon_target_frames.push_back(frame.get<std::string>());
    }
  }

  for (const auto &icp_params : J["icp_params"]) {
    max_corr = icp_params.at("max_corr");
    max_iter = icp_params.at("max_iter");
    t_eps = icp_params.at("t_eps");
    fit_eps = icp_params.at("fit_eps");
    set_show_transform = icp_params.at("set_show_transform");
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
      for (geometry_msgs::TransformStamped tf : tf_message->transforms) {
        tree.AddTransform(tf);
      }
    }
  }

  // get transform to each of the targets at specified time
  for (uint8_t n; n < vicon_target_frames.size(); n++) {
    Eigen::Affine3d T_SENSOR_TGTn =
        tree.GetTransformEigen(sensor_frame, vicon_target_frames[n], time);
    T_sensor_tgts_estimated.push_back(T_SENSOR_TGTn);
  }
  return T_sensor_tgts_estimated;
}

void GetLidarMeasurements(rosbag::Bag &bag, std::string &topic,
                          std::string &frame) {
  rosbag::View view(bag, ros::TIME_MIN, ros::TIME_MAX, true);

  pcl::PCLPointCloud2::Ptr cloud_pc2 =
      boost::make_shared<pcl::PCLPointCloud2>();
  PointCloud::Ptr cloud = boost::make_shared<PointCloud>();
  ros::Duration time_step(lidar_time_steps);
  ros::Time time_last(0, 0);
  time_last = time_last + time_step;
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      T_lidar_tgts_estimated;

  for (auto iter = view.begin(); iter != view.end(); iter++) {
    if (iter->getTopic() == topic) {
      boost::shared_ptr<sensor_msgs::PointCloud2> lidar_msg =
          iter->instantiate<sensor_msgs::PointCloud2>();
      if (lidar_msg->header.stamp > time_last + time_step) {
        time_last = lidar_msg->header.stamp;
        pcl_conversions::toPCL(*lidar_msg, *cloud_pc2);
        pcl::fromPCLPointCloud2(*cloud_pc2, *cloud);

        try {
          T_lidar_tgts_estimated =
              GetInitialGuess(bag, lidar_msg->header.stamp, frame);
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
        bool measurement_valid;
        Eigen::Vector4d measurement;
        lidar_extractor.SetScan(cloud);
        for (uint8_t n = 0; n < T_lidar_tgts_estimated.size(); n++) {
          lidar_extractor.ExtractCylinder(T_lidar_tgts_estimated[n], n);
          const auto measurement_info = lidar_extractor.GetMeasurementInfo();
          measurement = measurement_info.first;
          measurement_valid = measurement_info.second;
        }
      }
    }
  }
}

void GetImageMeasurements(rosbag::Bag &bag, std::string &topic,
                          std::string &frame) {
  rosbag::View view(bag, ros::TIME_MIN, ros::TIME_MAX, true);
  for (auto iter = view.begin(); iter != view.end(); iter++) {
    if (iter->getTopic() == topic) {
      // Test
    }
  }
}

int main() {
  // get configuration settings
  std::string config_file;
  config_file = GetJSONFileNameConfig("ViconCalibrationConfigIG.json");
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

  // initialize lidar cylinder extractor object
  PointCloud::Ptr target_cloud(new PointCloud);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(target_cloud_name, *target_cloud) ==
      -1) {
    LOG_ERROR("Couldn't read template file: %s\n", target_cloud_name.c_str());
  }

  lidar_extractor.SetTemplateCloud(target_cloud);
  lidar_extractor.SetThreshold(target_crop_threshold); // Default: 0.01
  lidar_extractor.SetRadius(target_radius);            // Default: 0.0635
  lidar_extractor.SetHeight(target_height);            // Default: 0.5
  lidar_extractor.SetICPParameters(t_eps, fit_eps, max_corr, max_iter);
  lidar_extractor.SetShowTransformation(set_show_transform);

  // main loop
  for (uint8_t k = 0; k < lidar_topics.size(); k++) {
    GetLidarMeasurements(bag, lidar_topics[k], lidar_frames[k]);
    // GetImageMeasurements(bag, image_topics[k], image_frames[k]);
  }
  bag.close();

  return 0;
}
