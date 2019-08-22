#include "vicon_calibration/ViconCalibrator.h"
#include "beam_calibration/TfTree.h"
#include "vicon_calibration/GTSAMGraph.h"
#include "vicon_calibration/utils.h"
#include <Eigen/StdVector>
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
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

// PCL specific headers
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>

namespace vicon_calibration {

std::string ViconCalibrator::GetJSONFileNameData(std::string file_name) {
  std::string file_location = __FILE__;
  file_location.erase(file_location.end() - 27, file_location.end());
  file_location += "data/";
  file_location += file_name;
  return file_location;
}

std::string ViconCalibrator::GetJSONFileNameConfig(std::string file_name) {
  std::string file_location = __FILE__;
  file_location.erase(file_location.end() - 27, file_location.end());
  file_location += "config/";
  file_location += file_name;
  return file_location;
}

// TODO: add try and catch blocks
void ViconCalibrator::LoadJSON(std::string file_name) {
  nlohmann::json J;
  std::ifstream file(file_name);
  file >> J;

  params_.bag_file = J["bag_file"];
  params_.initial_calibration_file = J["initial_calibration"];
  params_.lookup_tf_calibrations = J["lookup_tf_calibrations"];
  params_.vicon_baselink_frame = J["vicon_baselink_frame"];

  for (const auto &params : J["image_processing_params"]) {
    params_.image_processing_params.num_intersections =
        params.at("num_intersections");
    params_.image_processing_params.min_length_ratio =
        params.at("min_length_ratio");
    params_.image_processing_params.max_gap_ratio = params.at("max_gap_ratio");
    params_.image_processing_params.canny_ratio = params.at("canny_ratio");
    params_.image_processing_params.encoding = params.at("encoding");
    params_.image_processing_params.show_measurements =
        params.at("show_measurements");
  }

  for (const auto &params : J["registration_params"]) {
    params_.registration_params.max_correspondance_distance =
        params.at("max_correspondance_distance");
    params_.registration_params.max_iterations = params.at("max_iterations");
    params_.registration_params.transform_epsilon =
        params.at("transform_epsilon");
    params_.registration_params.euclidean_epsilon =
        params.at("euclidean_epsilon");
    params_.registration_params.show_transform = params.at("show_transform");
    params_.registration_params.dist_acceptance_criteria =
        params.at("dist_acceptance_criteria");
    params_.registration_params.rot_acceptance_criteria =
        params.at("rot_acceptance_criteria");
  }

  for (const auto &params : J["target_params"]) {
    params_.target_params.radius = params.at("radius");
    params_.target_params.height = params.at("height");
    params_.target_params.crop_threshold = params.at("crop_threshold");
    std::string template_cloud_name = params.at("template_cloud");
    params_.target_params.template_cloud =
        GetJSONFileNameData(template_cloud_name);
    for (const auto &frame : params.at("vicon_target_frames")) {
      params_.target_params.vicon_target_frames.push_back(
          frame.get<std::string>());
    }
  }

  for (const auto &camera : J["camera_params"]) {
    vicon_calibration::CameraParams cam_params;
    cam_params.topic = camera.at("topic");
    cam_params.frame = camera.at("frame");
    cam_params.intrinsics = camera.at("intrinsics");
    cam_params.time_steps = camera.at("time_steps");
    params_.camera_params.push_back(cam_params);
  }

  for (const auto &lidar : J["lidar_params"]) {
    vicon_calibration::LidarParams lid_params;
    lid_params.topic = lidar.at("topic");
    lid_params.frame = lidar.at("frame");
    lid_params.time_steps = lidar.at("time_steps");
    params_.lidar_params.push_back(lid_params);
  }
}

std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
ViconCalibrator::GetInitialGuess(ros::Time &time, std::string &sensor_frame) {
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      T_sensor_tgts_estimated;

  // check 2 second time window for vicon baselink transform
  ros::Duration time_window_half(1);
  ros::Time start_time = time - time_window_half;
  ros::Time time_zero(0, 0);
  if (start_time <= time_zero) {
    start_time = time_zero;
  }
  ros::Time end_time = time + time_window_half;
  rosbag::View view(bag_, rosbag::TopicQuery("/tf"), start_time, end_time,
                    true);

  // initialize tree with initial calibrations:
  beam_calibration::TfTree tree;

  if (!params_.lookup_tf_calibrations) {
    std::string initial_calibration_file_dir;
    try {
      initial_calibration_file_dir =
          GetJSONFileNameData(params_.initial_calibration_file);
      tree.LoadJSON(initial_calibration_file_dir);
    } catch (nlohmann::detail::parse_error &ex) {
      LOG_ERROR("Unable to load json calibration file: %s",
                initial_calibration_file_dir.c_str());
    }
  }

  // TODO: Do we need to add the static transform for calibrations, or will it
  // automatically be added based on what's below?

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
  Eigen::Affine3d T_SENSOR_TGTn;
  for (uint8_t n; n < params_.target_params.vicon_target_frames.size(); n++) {
    T_SENSOR_TGTn = tree.GetTransformEigen(
        sensor_frame, params_.target_params.vicon_target_frames[n], time);
    T_sensor_tgts_estimated.push_back(T_SENSOR_TGTn);
  }
  return T_sensor_tgts_estimated;
}

void ViconCalibrator::GetLidarMeasurements(uint8_t &lidar_iter) {
  std::string topic = params_.lidar_params[lidar_iter].topic;
  rosbag::View view(bag_, rosbag::TopicQuery(topic), ros::TIME_MIN,
                    ros::TIME_MAX, true);
  pcl::PCLPointCloud2::Ptr cloud_pc2 =
      boost::make_shared<pcl::PCLPointCloud2>();
  PointCloud::Ptr cloud = boost::make_shared<PointCloud>();
  ros::Duration time_step(params_.lidar_params[lidar_iter].time_steps);
  ros::Time time_last(0, 0);

  for (auto iter = view.begin(); iter != view.end(); iter++) {
    boost::shared_ptr<sensor_msgs::PointCloud2> lidar_msg =
        iter->instantiate<sensor_msgs::PointCloud2>();
    ros::Time time_current = lidar_msg->header.stamp;
    if (time_current > time_last + time_step) {
      time_last = time_current;
      pcl_conversions::toPCL(*lidar_msg, *cloud_pc2);
      pcl::fromPCLPointCloud2(*cloud_pc2, *cloud);
      std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
          T_lidar_tgts_estimated;
      try {
        T_lidar_tgts_estimated = GetInitialGuess(
            time_current, params_.lidar_params[lidar_iter].frame);
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
      lidar_extractor_.SetScan(cloud);
      for (uint8_t n = 0; n < T_lidar_tgts_estimated.size(); n++) {
        lidar_extractor_.ExtractCylinder(T_lidar_tgts_estimated[n], n);
        const auto measurement_info = lidar_extractor_.GetMeasurementInfo();
        if (measurement_info.second) {
          vicon_calibration::LidarMeasurement lidar_measurement;
          lidar_measurement.measurement = measurement_info.first;
          lidar_measurement.lidar_id = lidar_iter;
          lidar_measurement.target_id = n;
          lidar_measurements_.push_back(lidar_measurement);
        }
      }
    }
  }
}

void ViconCalibrator::GetCameraMeasurements(rosbag::Bag &bag,
                                            std::string &topic,
                                            std::string &frame) {
  rosbag::View view(bag, ros::TIME_MIN, ros::TIME_MAX, true);
  for (auto iter = view.begin(); iter != view.end(); iter++) {
    if (iter->getTopic() == topic) {
      // Test
    }
  }
}

void ViconCalibrator::RunCalibration() {

  // get configuration settings
  std::string config_file;
  config_file = GetJSONFileNameConfig("ViconCalibrationConfigIG.json");
  try {
    LoadJSON(config_file);
  } catch (nlohmann::detail::parse_error &ex) {
    LOG_ERROR("Unable to load json config file: %s", config_file.c_str());
  }

  // load bag file
  try {
    LOG_INFO("Opening bag: %s", params_.bag_file.c_str());
    bag_.open(params_.bag_file, rosbag::bagmode::Read);
  } catch (rosbag::BagException &ex) {
    LOG_ERROR("Bag exception : %s", ex.what());
  }

  // loop through each lidar, get measurements and solve graph
  lidar_extractor_.SetTargetParams(params_.target_params);
  lidar_extractor_.SetRegistrationParams(params_.registration_params);
  for (uint8_t lidar_iter = 0; lidar_iter < params_.lidar_params.size();
       lidar_iter++) {
    this->GetLidarMeasurements(lidar_iter);
  }

  // loop through each camera, get measurements and solve graph
  /*
  camera_extractor_.SetTargetParams(params_.target_params);
  camera_extractor_.SetImageProcessingParams(params_.image_processing_params);
  for (uint8_t cam_iter = 0; cam_iter < params_.camera_params.size();
  cam_iter++) {
    camera_extractor_.SetCameraParams(params_.camera_params[cam_iter]);
    this->GetCameraMeasurements(cam_iter);
  }
  */

  bag_.close();

  // this->BuildGraph(); // Or do we make this its own object?
  // this->SolveGraph();
  // this->OutputCalibrations();
  // this->OutputErrorMetrics();

  return;
}

} // end namespace vicon_calibration
