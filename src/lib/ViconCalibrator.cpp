#include "vicon_calibration/ViconCalibrator.h"
#include "vicon_calibration/utils.h"
#include <Eigen/StdVector>
#include <beam_utils/math.hpp>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
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
  std::vector<double> vect;
  for (const auto &value : J["initial_guess_perturb"]) {
    vect.push_back(value);
  }
  params_.initial_guess_perturbation = vect;
  params_.vicon_baselink_frame = J["vicon_baselink_frame"];
  params_.show_measurements = J["show_measurements"];

  for (const auto &target : J["targets"]) {
    vicon_calibration::TargetParams target_info;
    target_info.frame_id = target.at("frame_id");
    target_info.extractor_type = target.at("extractor_type");
    std::string target_config_path =
        GetJSONFileNameConfig(target.at("target_config"));
    target_info.target_config_path = target_config_path;
    nlohmann::json J_target;
    std::ifstream file(target_config_path);
    file >> J_target;
    std::vector<double> vect1, vect2;
    for (const auto &value : J_target["crop_scan"]) {
      vect1.push_back(value);
    }
    Eigen::Vector3d crop_scan;
    crop_scan << vect1[0], vect1[1], vect1[2];
    target_info.crop_scan = crop_scan;
    for (const auto &value : J_target["crop_image"]) {
      vect2.push_back(value);
    }
    Eigen::Vector2d crop_image;
    crop_image << vect2[0], vect2[1];
    target_info.crop_image = crop_image;
    std::string template_cloud_path =
        GetJSONFileNameConfig(target.at("template_cloud_path"));
    pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud_path,
                                            *template_cloud) == -1) {
      LOG_ERROR("Couldn't read template file: %s\n",
                template_cloud_path.c_str());
    }
    target_info.template_cloud = template_cloud;
    for (const auto &keypoint : J_target["keypoints_lidar"]) {
      Eigen::Vector3d point;
      point << keypoint.at("x"), keypoint.at("y"), keypoint.at("z");
      target_info.keypoints_lidar.push_back(point);
    }
    for (const auto &keypoint : J_target["keypoints_camera"]) {
      Eigen::Vector3d point;
      point << keypoint.at("x"), keypoint.at("y"), keypoint.at("z");
      target_info.keypoints_camera.push_back(point);
    }
    params_.target_params_list.push_back(target_info);
  }

  for (const auto &camera : J["camera_params"]) {
    vicon_calibration::CameraParams cam_params;
    cam_params.topic = camera.at("topic");
    cam_params.frame = camera.at("frame");
    std::string intrinsics_filename = camera.at("intrinsics");
    cam_params.intrinsics = GetJSONFileNameData(intrinsics_filename);
    cam_params.time_steps = camera.at("time_steps");
    cam_params.images_distorted = camera.at("images_distorted");
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

void ViconCalibrator::LoadEstimatedExtrinsics() {

  if (!params_.lookup_tf_calibrations) {
    std::string initial_calibration_file_dir;
    try {
      initial_calibration_file_dir =
          GetJSONFileNameData(params_.initial_calibration_file);
      estimate_extrinsics_.LoadJSON(initial_calibration_file_dir);
    } catch (nlohmann::detail::parse_error &ex) {
      LOG_ERROR("Unable to load json calibration file: %s",
                initial_calibration_file_dir.c_str());
    }
  } else {
    // Look up all transforms from /tf_static topic
    rosbag::View view(bag_, rosbag::TopicQuery("/tf_static"), ros::TIME_MIN,
                      ros::TIME_MAX, true);
    for (const auto &msg_instance : view) {
      auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
      if (tf_message != nullptr) {
        for (geometry_msgs::TransformStamped tf : tf_message->transforms) {
          estimate_extrinsics_.AddTransform(tf, true);
        }
      }
    }

    // check if transform from baselink to vicon base exists, if not get it
    // from /tf topic. Sometimes, this static transform is broadcasted on /tf
    std::string to_frame = params_.vicon_baselink_frame;
    std::string from_frame = "base_link";
    try {
      Eigen::Affine3d T_VICONBASE_BASELINK =
          estimate_extrinsics_.GetTransformEigen(to_frame, from_frame);
    } catch (std::runtime_error &error) {
      LOG_INFO("Transform from base_link to %s not available on topic "
               "/tf_static, looking at topic /tf.",
               params_.vicon_baselink_frame.c_str());
      rosbag::View view2(bag_, rosbag::TopicQuery("/tf"), ros::TIME_MIN,
                         ros::TIME_MAX, true);
      for (const auto &msg_instance : view2) {
        auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
        if (tf_message != nullptr) {
          for (geometry_msgs::TransformStamped tf : tf_message->transforms) {
            std::string child = tf.child_frame_id;
            std::string parent = tf.header.frame_id;
            if ((child == to_frame && parent == from_frame) ||
                (child == from_frame && parent == to_frame)) {
              estimate_extrinsics_.AddTransform(tf, true);
              LOG_INFO("Found transform from base_link to %s",
                       params_.vicon_baselink_frame.c_str());
              goto end_of_loop;
            }
          }
        }
      }
      LOG_ERROR("Transform from base_link to %s not available on topic /tf",
                params_.vicon_baselink_frame.c_str());
    end_of_loop:;
    }
  }
}

void ViconCalibrator::LoadLookupTree() {
  lookup_tree_.Clear();
  ros::Duration time_window_half(1); // Check two second time window
  ros::Time start_time = lookup_time_ - time_window_half;
  ros::Time time_zero(0, 0);
  if (start_time <= time_zero) {
    start_time = time_zero;
  }
  ros::Time end_time = lookup_time_ + time_window_half;
  rosbag::View view(bag_, rosbag::TopicQuery("/tf"), start_time, end_time,
                    true);
  bool first_msg = true;
  for (const auto &msg_instance : view) {
    auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
    if (tf_message != nullptr) {
      for (geometry_msgs::TransformStamped tf : tf_message->transforms) {
        lookup_tree_.AddTransform(tf);
      }
    }
  }
}

void ViconCalibrator::GetInitialCalibration(std::string &sensor_frame,
                                            std::string type) {
  T_SENSOR_VICONBASE_ = estimate_extrinsics_.GetTransformEigen(
      sensor_frame, params_.vicon_baselink_frame);
  vicon_calibration::CalibrationResult calib_initial;
  calib_initial.transform = T_SENSOR_VICONBASE_.matrix();
  calib_initial.type = type;
  calib_initial.to_frame = sensor_frame;
  calib_initial.from_frame = params_.vicon_baselink_frame;
  calibrations_initial_.push_back(calib_initial);
}

void ViconCalibrator::GetInitialCalibrationPerturbed(std::string &sensor_frame,
                                                     std::string type) {
  T_SENSOR_pert_VICONBASE_ = utils::PerturbTransform(
      T_SENSOR_VICONBASE_, params_.initial_guess_perturbation);
  vicon_calibration::CalibrationResult calib_perturbed;
  calib_perturbed.transform = T_SENSOR_pert_VICONBASE_.matrix();
  calib_perturbed.type = type;
  calib_perturbed.to_frame = sensor_frame;
  calib_perturbed.from_frame = params_.vicon_baselink_frame;
  calibrations_perturbed_.push_back(calib_perturbed);
}

std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
ViconCalibrator::GetInitialGuess(std::string &sensor_frame) {
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      T_sensor_tgts_estimated;
  for (uint8_t n; n < params_.target_params_list.size(); n++) {
    // get transform from sensor to target
    Eigen::Affine3d T_VICONBASE_TGTn = lookup_tree_.GetTransformEigen(
        params_.vicon_baselink_frame, params_.target_params_list[n].frame_id,
        lookup_time_);
    // perturb  for simulation testing ONLY
    Eigen::Affine3d T_SENSOR_pert_TGTn =
        T_SENSOR_pert_VICONBASE_ * T_VICONBASE_TGTn;
    T_sensor_tgts_estimated.push_back(T_SENSOR_pert_TGTn);
  }
  return T_sensor_tgts_estimated;
}

std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
ViconCalibrator::GetTargetLocation() {
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      T_viconbase_tgts;
  for (uint8_t n; n < params_.target_params_list.size(); n++) {
    Eigen::Affine3d T_viconbase_tgt = lookup_tree_.GetTransformEigen(
        params_.vicon_baselink_frame, params_.target_params_list[n].frame_id,
        lookup_time_);
    T_viconbase_tgts.push_back(T_viconbase_tgt);
  }
  return T_viconbase_tgts;
}

void ViconCalibrator::GetLidarMeasurements(uint8_t &lidar_iter) {
  std::string topic = params_.lidar_params[lidar_iter].topic;
  std::string sensor_frame = params_.lidar_params[lidar_iter].frame;
  LOG_INFO("Getting lidar measurements for frame id: %s and topic: %s .",
           sensor_frame.c_str(), topic.c_str());
  rosbag::View view(bag_, rosbag::TopicQuery(topic), ros::TIME_MIN,
                    ros::TIME_MAX, true);

  if (view.size() == 0) {
    throw std::invalid_argument{
        "No lidar messages read. Check your topics in config file."};
  }

  pcl::PCLPointCloud2::Ptr cloud_pc2 =
      boost::make_shared<pcl::PCLPointCloud2>();
  PointCloud::Ptr cloud = boost::make_shared<PointCloud>();
  ros::Duration time_step(params_.lidar_params[lidar_iter].time_steps);
  ros::Time time_last(0, 0);

  this->GetInitialCalibration(sensor_frame, "LIDAR");
  this->GetInitialCalibrationPerturbed(sensor_frame, "LIDAR");

  for (auto iter = view.begin(); iter != view.end(); iter++) {
    boost::shared_ptr<sensor_msgs::PointCloud2> lidar_msg =
        iter->instantiate<sensor_msgs::PointCloud2>();
    ros::Time time_current = lidar_msg->header.stamp;
    if (time_current > time_last + time_step) {
      lookup_time_ = time_current;
      this->LoadLookupTree();
      time_last = time_current;
      pcl_conversions::toPCL(*lidar_msg, *cloud_pc2);
      pcl::fromPCLPointCloud2(*cloud_pc2, *cloud);
      std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
          T_lidar_tgts_estimated, T_viconbase_tgts;
      try {
        T_lidar_tgts_estimated = GetInitialGuess(sensor_frame);
        T_viconbase_tgts = GetTargetLocation();
      } catch (const std::exception err) {
        LOG_ERROR("%s", err.what());
        std::cout
            << "Possible reasons for lookup error: \n"
            << "- Start or End of bag could have message timing issues\n"
            << "- Vicon messages not synchronized with robot's ROS time\n"
            << "- Invalid initial calibrations, i.e. input transformations "
               "json has missing/invalid transforms\n";
        continue;
      }
      for (uint8_t n = 0; n < T_lidar_tgts_estimated.size(); n++) {
        std::string extractor_type =
            params_.target_params_list[n].extractor_type;
        if (extractor_type == "CYLINDER") {
          lidar_extractor_ =
              std::make_shared<vicon_calibration::CylinderLidarExtractor>();
        } else if (extractor_type == "DIAMOND") {
          // TODO: uncomment this when implementing the extractor
          // lidar_extractor_ =
          //    std::make_shared<vicon_calibration::DiamondLidarExtractor>();
        } else {
          throw std::invalid_argument{
              "Invalid extractor type. Options: CYLINDER, DIAMOND"};
        }

        lidar_extractor_->SetLidarParams(params_.lidar_params[lidar_iter]);
        lidar_extractor_->SetTargetParams(params_.target_params_list[n]);
        lidar_extractor_->SetShowMeasurements(params_.show_measurements);
        lidar_extractor_->ExtractKeypoints(T_lidar_tgts_estimated[n].matrix(),
                                           cloud);
        if (lidar_extractor_->GetMeasurementValid()) {
          vicon_calibration::LidarMeasurement lidar_measurement;
          lidar_measurement.keypoints = lidar_extractor_->GetMeasurement();
          lidar_measurement.T_VICONBASE_TARGET = T_viconbase_tgts[n].matrix();
          lidar_measurement.lidar_id = lidar_iter;
          lidar_measurement.target_id = n;
          lidar_measurement.lidar_frame =
              params_.lidar_params[lidar_iter].frame;
          lidar_measurement.target_frame =
              params_.target_params_list[n].frame_id;
          lidar_measurements_.push_back(lidar_measurement);
        }
      }
    }
  }
}

void ViconCalibrator::GetCameraMeasurements(uint8_t &cam_iter) {
  std::string topic = params_.camera_params[cam_iter].topic;
  std::string sensor_frame = params_.camera_params[cam_iter].frame;
  LOG_INFO("Getting camera measurements for frame id: %s and topic: %s .",
           sensor_frame.c_str(), topic.c_str());

  rosbag::View view(bag_, rosbag::TopicQuery(topic), ros::TIME_MIN,
                    ros::TIME_MAX, true);
  if (view.size() == 0) {
    throw std::invalid_argument{
        "No image messages read. Check your topics in config file."};
  }

  ros::Duration time_step(params_.camera_params[cam_iter].time_steps);
  ros::Time time_last(0, 0);
  ros::Time time_zero(0, 0);
  this->GetInitialCalibration(sensor_frame, "CAMERA");
  this->GetInitialCalibrationPerturbed(sensor_frame, "CAMERA");

  for (auto iter = view.begin(); iter != view.end(); iter++) {
    sensor_msgs::ImageConstPtr img_msg =
        iter->instantiate<sensor_msgs::Image>();
    ros::Time time_current = img_msg->header.stamp;
    // skip first instance to avoid errors at beginning of bag
    if (time_last == time_zero) {
      time_last = time_current;
    }
    cv::Mat current_image =
        cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8)->image;
    if (time_current > time_last + time_step) {
      lookup_time_ = time_current;
      this->LoadLookupTree();
      time_last = time_current;
      std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
          T_cam_tgts_estimated, T_viconbase_tgts;
      try {
        T_cam_tgts_estimated = GetInitialGuess(sensor_frame);
        T_viconbase_tgts = GetTargetLocation();
      } catch (const std::exception err) {
        LOG_ERROR("%s", err.what());
        std::cout
            << "Possible reasons for lookup error: \n"
            << "- Start or End of bag could have message timing issues\n"
            << "- Vicon messages not synchronized with robot's ROS time\n"
            << "- Invalid initial calibrations, i.e. input transformations "
               "json has missing/invalid transforms\n";
        continue;
      }
      for (uint8_t n = 0; n < T_cam_tgts_estimated.size(); n++) {
        std::string extractor_type =
            params_.target_params_list[n].extractor_type;
        if (extractor_type == "CYLINDER") {
          camera_extractor_ =
              std::make_shared<vicon_calibration::CylinderCameraExtractor>();
        } else if (extractor_type == "DIAMOND") {
          // TODO: uncomment this when implementing the extractor
          // camera_extractor_ =
          //     std::make_shared<vicon_calibration::DiamondCameraExtractor>();
        } else {
          throw std::invalid_argument{
              "Invalid extractor type. Options: CYLINDER, DIAMOND"};
        }

        camera_extractor_->SetCameraParams(params_.camera_params[cam_iter]);
        camera_extractor_->SetTargetParams(params_.target_params_list[n]);
        camera_extractor_->SetShowMeasurements(params_.show_measurements);
        camera_extractor_->ExtractKeypoints(T_cam_tgts_estimated[n].matrix(),
                                            current_image);
        if (camera_extractor_->GetMeasurementValid()) {
          vicon_calibration::CameraMeasurement camera_measurement;
          camera_measurement.keypoints = camera_extractor_->GetMeasurement();
          camera_measurement.T_VICONBASE_TARGET = T_viconbase_tgts[n].matrix();
          camera_measurement.camera_id = cam_iter;
          camera_measurement.target_id = n;
          camera_measurement.camera_frame =
              params_.camera_params[cam_iter].frame;
          camera_measurement.target_frame =
              params_.target_params_list[n].frame_id;
          camera_measurements_.push_back(camera_measurement);
        }
      }
    }
  }
}

void ViconCalibrator::RunCalibration(std::string config_file) {

  // get configuration settings
  std::string config_file_path;
  config_file_path = GetJSONFileNameConfig(config_file);
  try {
    LoadJSON(config_file_path);
  } catch (nlohmann::detail::parse_error &ex) {
    LOG_ERROR("Unable to load json config file: %s", config_file_path.c_str());
    LOG_ERROR("%s", ex.what());
  }

  // load bag file
  try {
    LOG_INFO("Opening bag: %s", params_.bag_file.c_str());
    bag_.open(params_.bag_file, rosbag::bagmode::Read);
  } catch (rosbag::BagException &ex) {
    LOG_ERROR("Bag exception : %s", ex.what());
  }

  // Load extrinsics
  this->LoadEstimatedExtrinsics();

  // loop through each lidar, get measurements and solve graph
  LOG_INFO("Loading lidar measurements.");
  for (uint8_t lidar_iter = 0; lidar_iter < params_.lidar_params.size();
       lidar_iter++) {
    this->GetLidarMeasurements(lidar_iter);
  }

  // loop through each camera, get measurements and solve graph
  LOG_INFO("Loading camera measurements.");
  for (uint8_t cam_iter = 0; cam_iter < params_.camera_params.size();
       cam_iter++) {
    this->GetCameraMeasurements(cam_iter);
  }

  bag_.close();

  // Build and solve graph
  // graph_.SetLidarMeasurements(lidar_measurements_);
  // graph.LoadTargetPoints(params_.target_params.template_cloud);
  // graph_.SetCameraMeasurements(camera_measurements_);
  // TODO: Change this to calibrations_initial_
  // graph_.SetInitialGuess(calibrations_perturbed_);
  // graph_.SolveGraph();
  // calibrations_result_ = graph_.GetResults();
  // std::string file_name_print = "file.dot";
  // graph_.Print(file_name_print, true);
  // utils::OutputLidarMeasurements(lidar_measurements_);
  utils::OutputCalibrations(calibrations_initial_,
                            "Initial Calibration Estimates:");
  utils::OutputCalibrations(calibrations_perturbed_, "Perturbed Calibrations:");
  utils::OutputCalibrations(calibrations_result_, "Optimized Calibrations:");
  // this->OutputErrorMetrics();

  return;
}

} // end namespace vicon_calibration
