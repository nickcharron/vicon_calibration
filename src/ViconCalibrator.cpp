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

std::string bag_file, initial_calibration_file, vicon_baselink_topic,
    target_cloud_name;
double camera_time_steps, lidar_time_steps;

beam_calibration::TfTree initial_calibrations;
vicon_calibration::LidarCylExtractor lidar_extractor;

std::vector<std::string> image_topics, image_frames, intrinsics, lidar_topics,
    lidar_frames, vicon_target_topics;

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

  for (const auto &topic : J["image_topics"]) {
    image_topics.push_back(topic.get<std::string>());
  }

  for (const auto &topic : J["image_frames"]) {
    image_frames.push_back(topic.get<std::string>());
  }

  camera_time_steps = J["camera_time_steps"];

  for (const auto &topic : J["intrinsics"]) {
    intrinsics.push_back(topic.get<std::string>());
  }

  for (const auto &topic : J["lidar_topics"]) {
    lidar_topics.push_back(topic.get<std::string>());
  }

  for (const auto &topic : J["lidar_frames"]) {
    lidar_frames.push_back(topic.get<std::string>());
  }

  lidar_time_steps = J["lidar_time_steps"];

  vicon_baselink_topic = J["vicon_baselink_topic"];

  for (const auto &topic : J["vicon_target_topics"]) {
    vicon_target_topics.push_back(topic.get<std::string>());
  }

  std::string target_cloud_file = J["target_cloud_name"];
  target_cloud_name = GetJSONFileNameData(target_cloud_file);
}

std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
GetInitialGuess(rosbag::Bag &bag, ros::Time &time, std::string &sensor_frame) {
  Eigen::Affine3d T_BASELINK_VICON, T_VICON_TGTn, T_BASELINK_SENSOR;
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      T_sensor_tgts_estimated;
  T_BASELINK_VICON.setIdentity();
  T_VICON_TGTn.setIdentity();
  std::string to_frame = "vicon_base";
  std::string from_frame = sensor_frame;
  T_BASELINK_SENSOR = initial_calibrations.GetTransform(to_frame, from_frame);

  // check 2 second time window for vicon baselink transform
  ros::Duration time_window(1);
  ros::Time start_time = time - time_window;
  ros::Time time_zero(0, 0);
  if (start_time <= time_zero) {
    start_time = time_zero;
  }
  ros::Time end_time = time + time_window;
  rosbag::View view(bag, start_time, end_time, true);

  // iterate through window and find transform from vicon base link
  for (auto iter = view.begin(); iter != view.end(); iter++) {
    if (iter->getTopic() == vicon_baselink_topic) {
      boost::shared_ptr<geometry_msgs::TransformStamped> vicon_msg =
          iter->instantiate<geometry_msgs::TransformStamped>();
      if (vicon_msg->header.stamp >= time) {
        T_BASELINK_VICON = tf2::transformToEigen(*vicon_msg);
        break;
      }
    }
  }
  if (T_BASELINK_VICON.matrix().isIdentity()) {
    LOG_ERROR("Cannot find baselink to vicon transform. Skipping timestep.");
    return T_sensor_tgts_estimated;
  }

  // get transforms for each target at given time
  for (uint8_t n; n < vicon_target_topics.size(); n++) {
    for (auto iter = view.begin(); iter != view.end(); iter++) {
      if (iter->getTopic() == vicon_target_topics[n]) {
        boost::shared_ptr<geometry_msgs::TransformStamped> vicon_msg =
            iter->instantiate<geometry_msgs::TransformStamped>();
        if (vicon_msg->header.stamp >= time) {
          T_VICON_TGTn = tf2::transformToEigen(*vicon_msg);
          Eigen::Affine3d T_SENSOR_TGTn =
              T_BASELINK_SENSOR.inverse() * T_BASELINK_VICON * T_VICON_TGTn;
          T_sensor_tgts_estimated.push_back(T_SENSOR_TGTn);
          break;
        }
      }
    }
  }
  // check that we got the right number of initial transforms
  if (T_sensor_tgts_estimated.size() != vicon_target_topics.size()) {
    LOG_ERROR("Cannot find valid transforms to all targets. Found %d "
              "transforms but %d topics were inputted.",
              T_sensor_tgts_estimated.size(), vicon_target_topics.size());
  }
  return T_sensor_tgts_estimated;
}

void GetLidarMeasurements(rosbag::Bag &bag, std::string &topic,
                          std::string &frame) {
  rosbag::View view(bag, ros::TIME_MIN, ros::TIME_MAX, true);
  pcl::PCLPointCloud2::Ptr cloud_pc2 =
      boost::make_shared<pcl::PCLPointCloud2>();
  PointCloud::Ptr cloud = boost::make_shared<PointCloud>();
  ros::Time time_last(0, 0);
  ros::Duration time_step(lidar_time_steps);
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

        T_lidar_tgts_estimated =
            GetInitialGuess(bag, lidar_msg->header.stamp, frame);
        bool measurement_valid;
        Eigen::Vector4d measurement;
        lidar_extractor.SetScan(cloud);
        for (uint8_t n = 0; n < vicon_target_topics.size(); n++){
          measurement = lidar_extractor.ExtractCylinder(T_lidar_tgts_estimated[n],
                                                        measurement_valid, 1);
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
  config_file = GetJSONFileNameConfig("ViconCalibrationConfig.json");
  try {
    LoadJson(config_file);
  } catch (nlohmann::detail::parse_error &ex) {
    LOG_ERROR("Unable to load json config file: %s", config_file.c_str());
  }

  // get initial calibration estimate
  std::string initial_calibration_file_dir;
  try {
    initial_calibration_file_dir =
        GetJSONFileNameData(initial_calibration_file);
    initial_calibrations.LoadJSON(initial_calibration_file_dir);
  } catch (nlohmann::detail::parse_error &ex) {
    LOG_ERROR("Unable to load json calibration file: %s",
              initial_calibration_file_dir.c_str());
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
  lidar_extractor.SetThreshold(3);   // Default: 0.015
  lidar_extractor.SetRadius(0.0635); // Default: 0.0635
  lidar_extractor.SetHeight(3);      // Default: 0.5
  lidar_extractor.SetShowTransformation(true);

  // main loop
  for (uint8_t k = 0; k < lidar_topics.size(); k++) {
    GetLidarMeasurements(bag, lidar_topics[k], lidar_frames[k]);
    // GetImageMeasurements(bag, image_topics[k], image_frames[k]);
  }
  bag.close();

  return 0;
}
