#include "vicon_calibration/ViconCalibrator.h"
#include "vicon_calibration/CalibrationVerification.h"
#include "vicon_calibration/utils.h"
#include "vicon_calibration/JsonTools.h"
#include <Eigen/StdVector>
#include <beam_utils/math.hpp>
#include <cv_bridge/cv_bridge.h>
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
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

// PCL specific headers
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

namespace vicon_calibration {

std::string ViconCalibrator::GetJSONFileNameData(const std::string &file_name) {
  std::string file_location = __FILE__;
  file_location.erase(file_location.end() - 27, file_location.end());
  file_location += "data/";
  file_location += file_name;
  return file_location;
}

std::string
ViconCalibrator::GetJSONFileNameConfig(const std::string &file_name) {
  std::string file_location = __FILE__;
  file_location.erase(file_location.end() - 27, file_location.end());
  file_location += "config/";
  file_location += file_name;
  return file_location;
}

void ViconCalibrator::LoadEstimatedExtrinsics() {
  LOG_INFO("Loading estimated extrinsics");
  if (!params_->lookup_tf_calibrations) {
    // Look up transforms from json file
    std::string initial_calibration_file_dir;
    try {
      initial_calibration_file_dir =
          GetJSONFileNameData(params_->initial_calibration_file);
      estimate_extrinsics_->LoadJSON(initial_calibration_file_dir);
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
          try {
            estimate_extrinsics_->AddTransform(tf, true);
          } catch (...) {
            // Nothing
          }
        }
      }
    }

    // check if transform from baselink to vicon base exists, if not get it
    // from /tf topic. Sometimes, this static transform is broadcasted on /tf
    std::string to_frame = params_->vicon_baselink_frame;
    std::string from_frame = "base_link";
    try {
      Eigen::Affine3d T_VICONBASE_BASELINK =
          estimate_extrinsics_->GetTransformEigen(to_frame, from_frame);
    } catch (std::runtime_error &error) {
      LOG_INFO("Transform from base_link to %s not available on topic "
               "/tf_static, looking at topic /tf.",
               params_->vicon_baselink_frame.c_str());
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
              estimate_extrinsics_->AddTransform(tf, true);
              LOG_INFO("Found transform from base_link to %s",
                       params_->vicon_baselink_frame.c_str());
              goto end_of_loop;
            }
          }
        }
      }
      LOG_ERROR("Transform from base_link to %s not available on topic /tf",
                params_->vicon_baselink_frame.c_str());
    end_of_loop:;
    }
  }
}

void ViconCalibrator::LoadLookupTree() {
  lookup_tree_->Clear();
  ros::Duration time_window_half(1); // Check two second time window
  ros::Time start_time = lookup_time_ - time_window_half;
  ros::Time time_zero(0, 0);
  if (start_time <= time_zero) {
    start_time = time_zero;
  }
  ros::Time end_time = lookup_time_ + time_window_half;
  rosbag::View view(bag_, rosbag::TopicQuery("/tf"), start_time, end_time,true);
  bool first_msg = true;
  for (const auto &msg_instance : view) {
    auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
    if (tf_message != nullptr) {
      for (geometry_msgs::TransformStamped tf : tf_message->transforms) {
        lookup_tree_->AddTransform(tf);
      }
    }
  }
}

void ViconCalibrator::GetInitialCalibration(std::string &sensor_frame,
                                            SensorType type,
                                            uint8_t &sensor_id) {
  T_VICONBASE_SENSOR_ =
      estimate_extrinsics_->GetTransformEigen(params_->vicon_baselink_frame, sensor_frame)
          .matrix();
  vicon_calibration::CalibrationResult calib_initial;
  calib_initial.transform = T_VICONBASE_SENSOR_;
  calib_initial.type = type;
  calib_initial.sensor_id = sensor_id;
  calib_initial.to_frame = params_->vicon_baselink_frame;
  calib_initial.from_frame = sensor_frame;
  calibrations_initial_.push_back(calib_initial);
}

void ViconCalibrator::GetInitialCalibrationPerturbed(std::string &sensor_frame,
                                                     SensorType type,
                                                     uint8_t &sensor_id) {
  T_VICONBASE_SENSOR_pert_ = utils::PerturbTransform(
      T_VICONBASE_SENSOR_, params_->initial_guess_perturbation);
  vicon_calibration::CalibrationResult calib_perturbed;
  calib_perturbed.transform = T_VICONBASE_SENSOR_pert_;
  calib_perturbed.type = type;
  calib_perturbed.sensor_id = sensor_id;
  calib_perturbed.to_frame = params_->vicon_baselink_frame;
  calib_perturbed.from_frame = sensor_frame;
  calibrations_perturbed_.push_back(calib_perturbed);
}

std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
ViconCalibrator::GetInitialGuess(std::string &sensor_frame) {
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      T_sensor_tgts_estimated;
  for (uint8_t n; n < params_->target_params.size(); n++) {
    // get transform from sensor to target
    Eigen::Affine3d T_VICONBASE_TGTn = lookup_tree_->GetTransformEigen(
        params_->vicon_baselink_frame, params_->target_params[n]->frame_id,
        lookup_time_);
    // perturb  for simulation testing ONLY
    Eigen::Affine3d TA_VICONBASE_SENSOR_pert_;
    TA_VICONBASE_SENSOR_pert_.matrix() = T_VICONBASE_SENSOR_pert_;
    Eigen::Affine3d T_SENSOR_pert_TGTn =
        TA_VICONBASE_SENSOR_pert_.inverse() * T_VICONBASE_TGTn;
    T_sensor_tgts_estimated.push_back(T_SENSOR_pert_TGTn);
  }
  return T_sensor_tgts_estimated;
}

void ViconCalibrator::GetLidarMeasurements(uint8_t &lidar_iter) {
  std::string topic = params_->lidar_params[lidar_iter]->topic;
  std::string sensor_frame = params_->lidar_params[lidar_iter]->frame;
  LOG_INFO("Getting lidar measurements for frame id: %s and topic: %s .",
           sensor_frame.c_str(), topic.c_str());
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      T_lidar_tgts_estimated_prev;
  rosbag::View view(bag_, rosbag::TopicQuery(topic), ros::TIME_MIN,
                    ros::TIME_MAX, true);

  if (view.size() == 0) {
    throw std::invalid_argument{
        "No lidar messages read. Check your topics in config file."};
  }

  pcl::PCLPointCloud2::Ptr cloud_pc2 =
      boost::make_shared<pcl::PCLPointCloud2>();
  PointCloud::Ptr cloud = boost::make_shared<PointCloud>();
  ros::Duration time_step(params_->lidar_params.at(lidar_iter)->time_steps);
  ros::Time time_last(0, 0);
  this->GetInitialCalibration(sensor_frame, SensorType::LIDAR, lidar_iter);
  this->GetInitialCalibrationPerturbed(sensor_frame, SensorType::LIDAR,
                                       lidar_iter);
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
        T_viconbase_tgts = utils::GetTargetLocation(
            params_->target_params, params_->vicon_baselink_frame,
            lookup_time_, lookup_tree_);
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
        // check minimal translation
        if (T_lidar_tgts_estimated_prev.size() > 0) {
          Eigen::Vector3d error = T_lidar_tgts_estimated[n].translation() -
                                  T_lidar_tgts_estimated_prev[n].translation();
          error[0] = sqrt(error[0] * error[0] * 1000000) / 1000;
          error[1] = sqrt(error[1] * error[1] * 1000000) / 1000;
          error[2] = sqrt(error[2] * error[2] * 1000000) / 1000;
          if (error[0] < params_->min_measurement_motion &&
              error[1] < params_->min_measurement_motion &&
              error[2] < params_->min_measurement_motion) {
            LOG_INFO("Target has not moved relative to base since last "
                     "measurement. Skipping.");
            continue;
          }
        }
        std::string extractor_type =
            params_->target_params[n]->extractor_type;
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
        lidar_extractor_->SetLidarParams(params_->lidar_params[lidar_iter]);
        lidar_extractor_->SetTargetParams(params_->target_params[n]);
        lidar_extractor_->SetShowMeasurements(params_->show_measurements);
        lidar_extractor_->ExtractKeypoints(T_lidar_tgts_estimated[n].matrix(),
                                           cloud);
        if (lidar_extractor_->GetMeasurementValid()) {
          vicon_calibration::LidarMeasurement lidar_measurement;
          lidar_measurement.keypoints = lidar_extractor_->GetMeasurement();
          lidar_measurement.T_VICONBASE_TARGET = T_viconbase_tgts[n].matrix();
          lidar_measurement.lidar_id = lidar_iter;
          lidar_measurement.target_id = n;
          lidar_measurement.lidar_frame =
              params_->lidar_params[lidar_iter]->frame;
          lidar_measurement.target_frame =
              params_->target_params[n]->frame_id;
          lidar_measurements_.push_back(lidar_measurement);
        }
      }
      T_lidar_tgts_estimated_prev = T_lidar_tgts_estimated;
    }
  }
}

void ViconCalibrator::GetCameraMeasurements(uint8_t &cam_iter) {
  std::string topic = params_->camera_params[cam_iter]->topic;
  std::string sensor_frame = params_->camera_params[cam_iter]->frame;
  LOG_INFO("Getting camera measurements for frame id: %s and topic: %s .",
           sensor_frame.c_str(), topic.c_str());
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      T_cam_tgts_estimated_prev;
  rosbag::View view(bag_, rosbag::TopicQuery(topic), ros::TIME_MIN,
                    ros::TIME_MAX, true);
  if (view.size() == 0) {
    throw std::invalid_argument{
        "No image messages read. Check your topics in config file."};
  }

  int counter = 0;
  ros::Duration time_step(params_->camera_params[cam_iter]->time_steps);
  ros::Time time_last(0, 0);
  ros::Time time_zero(0, 0);
  this->GetInitialCalibration(sensor_frame, SensorType::CAMERA, cam_iter);
  this->GetInitialCalibrationPerturbed(sensor_frame, SensorType::CAMERA,
                                       cam_iter);

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
        T_viconbase_tgts = utils::GetTargetLocation(
            params_->target_params, params_->vicon_baselink_frame,
            lookup_time_, lookup_tree_);
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
        // check minimal translation
        if (T_cam_tgts_estimated_prev.size() > 0) {
          Eigen::Vector3d error = T_cam_tgts_estimated[n].translation() -
                                  T_cam_tgts_estimated_prev[n].translation();
          error[0] = sqrt(error[0] * error[0] * 1000000) / 1000;
          error[1] = sqrt(error[1] * error[1] * 1000000) / 1000;
          error[2] = sqrt(error[2] * error[2] * 1000000) / 1000;
          if (error[0] < params_->min_measurement_motion &&
              error[1] < params_->min_measurement_motion &&
              error[2] < params_->min_measurement_motion) {
            LOG_INFO("Target has not moved relative to base since last "
                     "measurement. Skipping.");
            continue;
          }
        }
        std::string extractor_type =
            params_->target_params[n]->extractor_type;
        if (extractor_type == "CYLINDER") {
          camera_extractor_ =
              std::make_shared<vicon_calibration::CylinderCameraExtractor>();
        } else if (extractor_type == "DIAMOND") {
          camera_extractor_ =
              std::make_shared<vicon_calibration::DiamondCameraExtractor>();
        } else {
          throw std::invalid_argument{
              "Invalid extractor type. Options: CYLINDER, DIAMOND"};
        }

        camera_extractor_->SetCameraParams(params_->camera_params[cam_iter]);
        camera_extractor_->SetTargetParams(params_->target_params[n]);
        camera_extractor_->SetShowMeasurements(params_->show_measurements);
        camera_extractor_->ExtractKeypoints(T_cam_tgts_estimated[n].matrix(),
                                            current_image);

        if (camera_extractor_->GetMeasurementValid()) {
          counter++;
          vicon_calibration::CameraMeasurement camera_measurement;
          camera_measurement.keypoints = camera_extractor_->GetMeasurement();
          camera_measurement.T_VICONBASE_TARGET = T_viconbase_tgts[n].matrix();
          camera_measurement.camera_id = cam_iter;
          camera_measurement.target_id = n;
          camera_measurement.camera_frame =
              params_->camera_params[cam_iter]->frame;
          camera_measurement.target_frame =
              params_->target_params[n]->frame_id;
          camera_measurements_.push_back(camera_measurement);
        }
      }
      T_cam_tgts_estimated_prev = T_cam_tgts_estimated;
    }
  }
  LOG_INFO("Stored %d measurements for camera with frame id: %s", counter,
           sensor_frame.c_str());
}

void ViconCalibrator::GetLoopClosureMeasurements() {
  // TODO: complete this
}

void ViconCalibrator::RunCalibration(std::string config_file) {

  // get configuration settings
  config_file_path_ = GetJSONFileNameConfig(config_file);

  try {
    JsonTools json_loader;
    params_ = json_loader.LoadViconCalibratorParams(config_file_path_);
  } catch (nlohmann::detail::parse_error &ex) {
    LOG_ERROR("Unable to load json config file: %s", config_file_path_.c_str());
    LOG_ERROR("%s", ex.what());
  }

  // load bag file
  try {
    LOG_INFO("Opening bag: %s", params_->bag_file.c_str());
    bag_.open(params_->bag_file, rosbag::bagmode::Read);
  } catch (rosbag::BagException &ex) {
    LOG_ERROR("Bag exception : %s", ex.what());
  }

  // Load extrinsics
  this->LoadEstimatedExtrinsics();

  // loop through each lidar, get measurements and solve graph
  LOG_INFO("Loading lidar measurements.");
  for (uint8_t lidar_iter = 0; lidar_iter < params_->lidar_params.size();
       lidar_iter++) {
    this->GetLidarMeasurements(lidar_iter);
  }

  // loop through each camera, get measurements and solve graph
  LOG_INFO("Loading camera measurements.");
  for (uint8_t cam_iter = 0; cam_iter < params_->camera_params.size();
       cam_iter++) {
    this->GetCameraMeasurements(cam_iter);
  }

  bag_.close();

  // Build and solve graph
  graph_.SetLidarMeasurements(lidar_measurements_);
  graph_.SetTargetParams(params_->target_params);
  graph_.SetCameraParams(params_->camera_params);
  graph_.SetCameraMeasurements(camera_measurements_);
  // TODO: Change this to calibrations_initial_
  // graph_.SetInitialGuess(calibrations_initial_);
  graph_.SetInitialGuess(calibrations_perturbed_);
  graph_.SolveGraph();
  calibrations_result_ = graph_.GetResults();
  utils::OutputCalibrations(calibrations_initial_,
                            "Initial Calibration Estimates:");
  utils::OutputCalibrations(calibrations_perturbed_, "Perturbed Calibrations:");
  utils::OutputCalibrations(calibrations_result_, "Optimized Calibrations:");

  if (params_->run_verification) {
    CalibrationVerification ver;
    ver.SetConfig(config_file_path_);
    ver.SetParams(params_);
    ver.SetInitialCalib(calibrations_initial_);
    ver.SetPeturbedCalib(calibrations_perturbed_);
    ver.SetOptimizedCalib(calibrations_result_);
    ver.ProcessResults();
  }
  return;
}

} // end namespace vicon_calibration
