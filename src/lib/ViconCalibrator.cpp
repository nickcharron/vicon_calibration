#include <vicon_calibration/ViconCalibrator.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <time.h>

#include <Eigen/StdVector>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <nlohmann/json.hpp>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2/buffer_core.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_msgs/TFMessage.h>

#include <vicon_calibration/CalibrationVerification.h>
#include <vicon_calibration/JsonTools.h>
#include <vicon_calibration/Params.h>
#include <vicon_calibration/PclConversions.h>
#include <vicon_calibration/Utils.h>
#include <vicon_calibration/measurement_extractors/CameraExtractor.h>
#include <vicon_calibration/measurement_extractors/CameraExtractors.h>
#include <vicon_calibration/measurement_extractors/LidarExtractor.h>
#include <vicon_calibration/measurement_extractors/LidarExtractors.h>
#include <vicon_calibration/optimization/CeresOptimizer.h>
#include <vicon_calibration/optimization/Optimizer.h>

namespace vicon_calibration {

ViconCalibrator::ViconCalibrator(const CalibratorInputs& inputs)
    : inputs_(inputs) {}

void ViconCalibrator::LoadEstimatedExtrinsics() {
  if (!params_->initial_calibration_file.empty()) {
    LOG_INFO("Loading estimated extrinsics from json");
    // Look up transforms from json file
    try {
      estimate_extrinsics_->LoadJSON(params_->initial_calibration_file);
    } catch (nlohmann::detail::parse_error& ex) {
      LOG_ERROR("Unable to load json calibration file: %s",
                params_->initial_calibration_file.c_str());
    }
    return;
  }

  LOG_INFO("Loading estimated extrinsics from tf topic in bag.");
  LOG_WARN("If extrinsics are not published to tf, then you need to provide a "
           "path to the extrinsics using -initial_calibration");

  // Look up all transforms from /tf_static topic
  rosbag::View view(bag_, rosbag::TopicQuery("/tf_static"), ros::TIME_MIN,
                    ros::TIME_MAX, true);
  for (const auto& msg_instance : view) {
    auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
    if (tf_message == nullptr) { continue; }
    for (geometry_msgs::TransformStamped tf : tf_message->transforms) {
      try {
        estimate_extrinsics_->AddTransform(tf, true);
      } catch (...) {
        // Nothing
      }
    }

    // check if transform from baselink to vicon base exists, if not get it
    // from /tf topic. Sometimes, this static transform is broadcasted on /tf
    std::string to_frame = params_->vicon_baselink_frame;
    std::string from_frame = "base_link";
    try {
      Eigen::Affine3d T_Robot_Baselink =
          estimate_extrinsics_->GetTransformEigen(to_frame, from_frame);
    } catch (std::runtime_error& error) {
      LOG_INFO("Transform from base_link to %s not available on topic "
               "/tf_static, looking at topic /tf.",
               params_->vicon_baselink_frame.c_str());
      rosbag::View view2(bag_, rosbag::TopicQuery("/tf"), time_start_,
                         time_end_, true);
      for (const auto& msg_instance : view2) {
        auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
        if (tf_message == nullptr) { continue; }
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
      LOG_ERROR("Transform from base_link to %s not available on topic /tf",
                params_->vicon_baselink_frame.c_str());
    end_of_loop:;
    }
  }
}

void ViconCalibrator::LoadLookupTree() {
  lookup_tree_->Clear();

  // add an extra second of time before and after time window
  ros::Duration dt(1);
  rosbag::View view(bag_, rosbag::TopicQuery("/tf"), time_start_ - dt,
                    time_end_ + dt, true);
  for (const auto& msg_instance : view) {
    auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
    if (tf_message != nullptr) {
      for (geometry_msgs::TransformStamped tf : tf_message->transforms) {
        lookup_tree_->AddTransform(tf);
      }
    }
  }
}

void ViconCalibrator::GetInitialCalibrations() {
  calibrations_initial_.clear();
  for (uint8_t i = 0; i < params_->lidar_params.size(); i++) {
    Eigen::Affine3d T_Robot_Sensor = estimate_extrinsics_->GetTransformEigen(
        params_->vicon_baselink_frame, params_->lidar_params[i]->frame);
    vicon_calibration::CalibrationResult calib_initial;
    calib_initial.transform = T_Robot_Sensor.matrix();
    calib_initial.type = SensorType::LIDAR;
    calib_initial.sensor_id = i;
    calib_initial.to_frame = params_->vicon_baselink_frame;
    calib_initial.from_frame = params_->lidar_params[i]->frame;
    calibrations_initial_.push_back(calib_initial);
  }
  for (uint8_t i = 0; i < params_->camera_params.size(); i++) {
    Eigen::Affine3d T_Robot_Sensor = estimate_extrinsics_->GetTransformEigen(
        params_->vicon_baselink_frame, params_->camera_params[i]->frame);
    vicon_calibration::CalibrationResult calib_initial;
    calib_initial.transform = T_Robot_Sensor.matrix();
    calib_initial.type = SensorType::CAMERA;
    calib_initial.sensor_id = i;
    calib_initial.to_frame = params_->vicon_baselink_frame;
    calib_initial.from_frame = params_->camera_params[i]->frame;
    calibrations_initial_.push_back(calib_initial);
  }
}

void ViconCalibrator::GetLidarMeasurements(uint8_t& lidar_iter) {
  std::string topic = params_->lidar_params[lidar_iter]->topic;
  std::string sensor_frame = params_->lidar_params[lidar_iter]->frame;
  LOG_INFO("Getting lidar measurements for frame id: %s and topic: %s .",
           sensor_frame.c_str(), topic.c_str());

  // get calibration
  bool success;
  Eigen::Matrix4d T_Robot_Sensor = utils::GetT_Robot_Sensor(
      calibrations_initial_, SensorType::LIDAR, lidar_iter, success);

  if (!success) {
    LOG_ERROR("Unable to find calibration for lidar ID: %d",
              static_cast<int>(lidar_iter));
    throw std::runtime_error{"Unable to find calibration."};
  }

  rosbag::View view(bag_, rosbag::TopicQuery(topic), time_start_, time_end_,
                    true);

  if (view.size() == 0) {
    throw std::invalid_argument{
        "No lidar messages read. Check your topics in config file."};
  }

  int valid_measurements = 0;
  int current_measurement = 0;
  ros::Duration time_step(params_->time_steps);
  ros::Time time_last(0, 0);
  int num_tgts = params_->target_params.size();
  std::vector<Eigen::Matrix4d> T_Sensor_Tgts_prev(num_tgts);
  for (auto iter = view.begin(); iter != view.end(); iter++) {
    auto lidar_msg = iter->instantiate<sensor_msgs::PointCloud2>();

    if (lidar_msg == NULL) {
      LOG_ERROR(
          "Unable to instantiate message of type sensor_msgs::PointCloud2, "
          "make sure your input topics are of correct type.");
      throw std::runtime_error{"Unable to instantiate message."};
    }
    ros::Time time_current = lidar_msg->header.stamp;

    // check min and max of tf tree
    if (time_current <= lookup_tree_->GetStartTime() ||
        time_current >= lookup_tree_->GetEndTime()) {
      continue;
    }

    // check time increment is greater than min
    if (time_current <= time_last + time_step) { continue; }
    time_last = time_current;

    // iterate through all targets adding measurements for this device
    for (int tgt_id = 0; tgt_id < num_tgts; tgt_id++) {
      // get estimated target pose
      Eigen::Matrix4d T_Robot_Target =
          lookup_tree_
              ->GetTransformEigen(params_->vicon_baselink_frame,
                                  params_->target_params[tgt_id]->frame_id,
                                  time_current)
              .matrix();
      Eigen::Matrix4d T_Sensor_Tgt =
          T_Robot_Sensor.inverse().matrix() * T_Robot_Target;

      // check min motion
      if (!PassedMinMotion(T_Sensor_Tgt, T_Sensor_Tgts_prev[tgt_id])) {
        lidar_counters_.at(lidar_iter).rejected_still++;
        continue;
      }

      // check max velocity
      if (!PassedVelocityThreshold(T_Robot_Target, time_current, tgt_id)) {
        LOG_INFO("Target is moving too quickly. Skipping.");
        lidar_counters_.at(lidar_iter).rejected_fast++;
        continue;
      }

      // extract measurement
      pcl::PCLPointCloud2 cloud_pc2;
      pcl_conversions::toPCL(*lidar_msg, cloud_pc2);
      PointCloud cloud_tmp;
      pcl::fromPCLPointCloud2(cloud_pc2, cloud_tmp);
      PointCloud::Ptr cloud = std::make_shared<PointCloud>();
      input_cropbox_.Filter(cloud_tmp, *cloud);

      std::shared_ptr<LidarExtractor> lidar_extractor = LidarExtractor::Create(
          params_->target_params[tgt_id]->lidar_extractor_type,
          params_->lidar_params[lidar_iter], params_->target_params[tgt_id],
          params_->show_lidar_measurements, pcl_viewer_);

      if (params_->show_lidar_measurements) {
        std::cout << "---------------------------------\n"
                  << "Processing measurement for: Lidar"
                  << std::to_string(lidar_iter + 1) << ", Tgt"
                  << std::to_string(tgt_id + 1) << "\n"
                  << "Timestamp: " << std::to_string(time_current.toSec())
                  << "\n";
      }
      lidar_extractor->ProcessMeasurement(T_Sensor_Tgt, cloud,
                                          params_->show_lidar_measurements);

      if (lidar_extractor->GetMeasurementValid()) {
        lidar_counters_.at(lidar_iter).accepted++;
        valid_measurements++;
        auto lidar_measurement = std::make_shared<LidarMeasurement>();
        lidar_measurement->keypoints = lidar_extractor->GetMeasurement();
        lidar_measurement->T_Robot_Target = T_Robot_Target;
        lidar_measurement->lidar_id = lidar_iter;
        lidar_measurement->target_id = tgt_id;
        lidar_measurement->lidar_frame =
            params_->lidar_params[lidar_iter]->frame;
        lidar_measurement->target_frame =
            params_->target_params[tgt_id]->frame_id;
        lidar_measurement->time_stamp = time_current;
        lidar_measurements_[lidar_iter][current_measurement] =
            lidar_measurement;
        LOG_INFO("Measurement accepted.");
        T_Sensor_Tgts_prev[tgt_id] = T_Sensor_Tgt;
      } else {
        lidar_counters_.at(lidar_iter).rejected_invalid++;
        LOG_INFO("Measurement rejected.");
      }
      current_measurement++;
    }

    // check max measurements
    if (lidar_counters_.at(lidar_iter).accepted >= params_->max_measurements) {
      break;
    }
  }
  LOG_INFO("Stored %d measurements for lidar with frame id: %s",
           valid_measurements, sensor_frame.c_str());
}

void ViconCalibrator::GetCameraMeasurements(uint8_t& cam_iter) {
  std::string topic = params_->camera_params[cam_iter]->topic;
  std::string sensor_frame = params_->camera_params[cam_iter]->frame;
  LOG_INFO("Getting camera measurements for frame id: %s and topic: %s .",
           sensor_frame.c_str(), topic.c_str());

  // get calibration
  bool success;
  Eigen::Matrix4d T_Robot_Sensor = utils::GetT_Robot_Sensor(
      calibrations_initial_, SensorType::CAMERA, cam_iter, success);

  if (!success) {
    LOG_ERROR("Unable to find calibration for camera ID: %d",
              static_cast<int>(cam_iter));
    throw std::runtime_error{"Unable to find calibration."};
  }

  rosbag::View view(bag_, rosbag::TopicQuery(topic), time_start_, time_end_,
                    true);
  if (view.size() == 0) {
    throw std::invalid_argument{
        "No image messages read. Check your topics in config file."};
  }

  int valid_measurements = 0;
  int current_measurement = 0;
  ros::Duration time_step(params_->time_steps);
  ros::Time time_last(0, 0);
  int num_tgts = params_->target_params.size();
  std::vector<Eigen::Matrix4d> T_Sensor_Tgts_prev(num_tgts);

  for (auto iter = view.begin(); iter != view.end(); iter++) {
    sensor_msgs::ImageConstPtr img_msg =
        iter->instantiate<sensor_msgs::Image>();

    if (img_msg == NULL) {
      LOG_ERROR("Unable to instantiate message of type sensor_msgs::Image, "
                "make sure your input topics are of correct type.");
      throw std::runtime_error{"Unable to instantiate message."};
    }

    ros::Time time_current = img_msg->header.stamp;

    // check min and max of tf tree
    if (time_current <= lookup_tree_->GetStartTime() ||
        time_current >= lookup_tree_->GetEndTime()) {
      continue;
    }

    // check time increment is greater than min
    if (time_current <= time_last + time_step) { continue; }
    time_last = time_current;

    // iterate through all targets adding measurements for this device
    for (int tgt_id = 0; tgt_id < num_tgts; tgt_id++) {
      // get estimated target pose
      Eigen::Matrix4d T_Robot_Target =
          lookup_tree_
              ->GetTransformEigen(params_->vicon_baselink_frame,
                                  params_->target_params[tgt_id]->frame_id,
                                  time_current)
              .matrix();
      Eigen::Matrix4d T_Sensor_Tgt =
          T_Robot_Sensor.inverse().matrix() * T_Robot_Target;

      // check min motion
      if (!PassedMinMotion(T_Sensor_Tgt, T_Sensor_Tgts_prev[tgt_id])) {
        camera_counters_.at(cam_iter).rejected_still++;
        continue;
      }

      // check max velocity
      if (!PassedVelocityThreshold(T_Robot_Target, time_current, tgt_id)) {
        LOG_INFO("Target is moving too quickly. Skipping.");
        camera_counters_.at(cam_iter).rejected_fast++;
        continue;
      }

      // extract measurement
      std::shared_ptr<CameraExtractor> camera_extractor =
          CameraExtractor::Create(
              params_->target_params[tgt_id]->camera_extractor_type);

      if (params_->show_camera_measurements) {
        std::cout << "---------------------------------\n"
                  << "Processing measurement for: Cam"
                  << std::to_string(cam_iter + 1) << ", Tgt"
                  << std::to_string(tgt_id + 1) << "\n"
                  << "Timestamp: " << std::to_string(time_current.toSec())
                  << "\n";
      }
      camera_extractor->SetCameraParams(params_->camera_params[cam_iter]);
      camera_extractor->SetTargetParams(params_->target_params[tgt_id]);
      camera_extractor->SetShowMeasurements(params_->show_camera_measurements);
      auto img = utils::RosImgToMat(*img_msg);
      camera_extractor->ProcessMeasurement(T_Sensor_Tgt, img);
      params_->show_camera_measurements =
          camera_extractor->GetShowMeasurements();

      // process measurement
      if (camera_extractor->GetMeasurementValid()) {
        camera_counters_.at(cam_iter).accepted++;
        valid_measurements++;
        std::shared_ptr<CameraMeasurement> camera_measurement =
            std::make_shared<CameraMeasurement>();
        camera_measurement->img = img.clone();
        camera_measurement->keypoints = camera_extractor->GetMeasurement();
        camera_measurement->T_Robot_Target = T_Robot_Target;
        camera_measurement->camera_id = cam_iter;
        camera_measurement->target_id = tgt_id;
        camera_measurement->camera_frame =
            params_->camera_params[cam_iter]->frame;
        camera_measurement->target_frame =
            params_->target_params[tgt_id]->frame_id;
        camera_measurement->time_stamp = time_current;
        camera_measurements_[cam_iter][current_measurement] =
            camera_measurement;
        LOG_INFO("Measurement accepted.");
        T_Sensor_Tgts_prev[tgt_id] = T_Sensor_Tgt;
      } else {
        camera_counters_.at(cam_iter).rejected_invalid++;
        LOG_INFO("Measurement rejected.");
      }
      current_measurement++;
    }

    // check max measurements
    if (camera_counters_.at(cam_iter).accepted >= params_->max_measurements) {
      break;
    }
  }

  LOG_INFO("Stored %d measurements for camera with frame id: %s",
           valid_measurements, sensor_frame.c_str());
}

bool ViconCalibrator::PassedMinMotion(const Eigen::Matrix4d& TA_S_T_prev,
                                      const Eigen::Matrix4d& TA_S_T_curr) {
  double error_t =
      (TA_S_T_curr.block(0, 3, 3, 1) - TA_S_T_prev.block(0, 3, 3, 1)).norm();
  double error_r = utils::CalculateRotationError(TA_S_T_prev.block(0, 0, 3, 3),
                                                 TA_S_T_curr.block(0, 0, 3, 3));
  if (error_t > params_->min_target_motion) {
    return true;
  } else if (error_r > params_->min_target_motion) {
    return true;
  } else {
    return false;
  }
}

bool ViconCalibrator::PassedVelocityThreshold(
    const Eigen::Matrix4d& T_Robot_Target, const ros::Time& time_current,
    int tgt_id) {
  ros::Time time_prev =
      time_current - ros::Duration(params_->velocity_window_size_s / 2);
  ros::Time time_next =
      time_current + ros::Duration(params_->velocity_window_size_s / 2);

  if (time_prev <= lookup_tree_->GetStartTime() ||
      time_next >= lookup_tree_->GetEndTime()) {
    return false;
  }

  Eigen::Affine3d T_Robot_Target_Prev = lookup_tree_->GetTransformEigen(
      params_->vicon_baselink_frame, params_->target_params[tgt_id]->frame_id,
      time_prev);

  Eigen::Affine3d T_Robot_Target_Next = lookup_tree_->GetTransformEigen(
      params_->vicon_baselink_frame, params_->target_params[tgt_id]->frame_id,
      time_next);

  Eigen::Vector3d vel_mps =
      T_Robot_Target_Prev.translation() - T_Robot_Target_Next.translation();

  vel_mps[0] = std::abs(vel_mps[0]) / params_->velocity_window_size_s;
  vel_mps[1] = std::abs(vel_mps[1]) / params_->velocity_window_size_s;
  vel_mps[2] = std::abs(vel_mps[2]) / params_->velocity_window_size_s;

  if (vel_mps.norm() > params_->max_target_velocity_mps) {
    return false;
  } else {
    return true;
  }
}

void ViconCalibrator::GetTimeWindow() {
  // get all measurement topics to query rosbag
  std::vector<std::string> topics;
  for (int i = 0; i < params_->lidar_params.size(); i++) {
    topics.push_back(params_->lidar_params[i]->topic);
  }
  for (int i = 0; i < params_->camera_params.size(); i++) {
    topics.push_back(params_->camera_params[i]->topic);
  }

  // get view with all measurement topics
  rosbag::View view_tmp(bag_, rosbag::TopicQuery(topics), ros::TIME_MIN,
                        ros::TIME_MAX, true);

  if (view_tmp.size() == 0) {
    throw std::runtime_error{
        "Empty rosbag view, make sure your topics are correct."};
  }

  // Get start time of measurements
  ros::Time first_msg_time;
  auto lid_msg = view_tmp.begin()->instantiate<sensor_msgs::PointCloud2>();
  auto cam_msg = view_tmp.begin()->instantiate<sensor_msgs::Image>();

  if (lid_msg != NULL) {
    first_msg_time = lid_msg->header.stamp;
  } else if (cam_msg != NULL) {
    first_msg_time = cam_msg->header.stamp;
  } else {
    throw std::runtime_error{"Invalid topic message type."};
  }

  // Get end time of measurements
  for (auto iter = view_tmp.begin(); iter != view_tmp.end(); iter++) {
    lid_msg = iter->instantiate<sensor_msgs::PointCloud2>();
    cam_msg = iter->instantiate<sensor_msgs::Image>();
  }

  ros::Time last_msg_time;
  if (lid_msg != NULL) {
    last_msg_time = lid_msg->header.stamp;
  } else if (cam_msg != NULL) {
    last_msg_time = cam_msg->header.stamp;
  } else {
    throw std::runtime_error{"Invalid topic message type."};
  }

  // check that bag time corresponds to measurement times
  if (first_msg_time > view_tmp.getEndTime() ||
      last_msg_time < view_tmp.getBeginTime()) {
    LOG_ERROR("ROS time from bag does not align with measurement timestamps.");
    std::cout << "view start time: " << std::setprecision(10)
              << view_tmp.getBeginTime().toSec() << "\n"
              << "view end time: " << std::setprecision(10)
              << view_tmp.getEndTime().toSec() << "\n"
              << "first measurement time: " << std::setprecision(10)
              << first_msg_time.toSec() << "\n"
              << "last measurement time: " << std::setprecision(10)
              << last_msg_time.toSec() << "\n";
    throw std::runtime_error{
        "ROS time from bag does not align with message times."};
  }

  time_start_ = first_msg_time + ros::Duration(params_->crop_time[0]);
  time_end_ = last_msg_time - ros::Duration(params_->crop_time[1]);
}

void ViconCalibrator::Setup() {
  // Load Params
  JsonTools json_loader(inputs_);
  params_ = json_loader.LoadViconCalibratorParams();

  // load bag file
  try {
    LOG_INFO("Opening bag: %s", params_->bag_file.c_str());
    bag_.open(params_->bag_file, rosbag::bagmode::Read);
  } catch (rosbag::BagException& ex) {
    LOG_ERROR("Bag exception : %s", ex.what());
    throw std::runtime_error{"Unable to open bag."};
  }

  GetTimeWindow();
  LoadLookupTree();

  // initialize size of lidar measurement and camera measurement containers
  ros::Duration bag_length = time_end_ - time_start_;
  int num_measurements = std::floor(bag_length.toSec() / params_->time_steps);
  int m = num_measurements * params_->target_params.size();
  std::vector<std::shared_ptr<LidarMeasurement>> lidar_init =
      std::vector<std::shared_ptr<LidarMeasurement>>(m, nullptr);
  std::vector<std::shared_ptr<CameraMeasurement>> camera_init =
      std::vector<std::shared_ptr<CameraMeasurement>>(m, nullptr);
  lidar_measurements_ =
      std::vector<std::vector<std::shared_ptr<LidarMeasurement>>>(
          params_->lidar_params.size(), lidar_init);
  camera_measurements_ =
      std::vector<std::vector<std::shared_ptr<CameraMeasurement>>>(
          params_->camera_params.size(), camera_init);

  // Load extrinsics
  LoadEstimatedExtrinsics();

  // setup input cropbox
  Eigen::Vector3f min_vec(-input_cropbox_max_, -input_cropbox_max_,
                          -input_cropbox_max_);
  Eigen::Vector3f max_vec(input_cropbox_max_, input_cropbox_max_,
                          input_cropbox_max_);
  input_cropbox_.SetMinVector(min_vec);
  input_cropbox_.SetMaxVector(max_vec);
  input_cropbox_.SetRemoveOutsidePoints(true);
}

void ViconCalibrator::GetMeasurements() {
  // setup visualizer
  if (params_->show_lidar_measurements) {
    pcl_viewer_ = std::make_shared<Visualizer>("LidarMeasurementVis");
  }

  // loop through each lidar, get measurements and solve problem
  int num_lidars = params_->lidar_params.size();
  LOG_INFO("Loading lidar measurements for %d lidars.",
           static_cast<int>(num_lidars));
  lidar_counters_ = std::vector<vicon_calibration::Counters>(num_lidars);
  for (uint8_t lidar_iter = 0; lidar_iter < num_lidars; lidar_iter++) {
    GetLidarMeasurements(lidar_iter);
  }

  // close visualizer
  if (params_->show_lidar_measurements) { pcl_viewer_ = nullptr; }

  // loop through each camera, get measurements and solve problem
  int num_cameras = params_->camera_params.size();
  LOG_INFO("Loading camera measurements for %d cameras.",
           static_cast<int>(num_cameras));
  camera_counters_ = std::vector<vicon_calibration::Counters>(num_cameras);
  for (uint8_t cam_iter = 0; cam_iter < num_cameras; cam_iter++) {
    GetCameraMeasurements(cam_iter);
  }

  OutputMeasurementStats();

  bag_.close();
}

void ViconCalibrator::Solve() {
  OptimizerInputs optimizer_inputs{
      .target_params = params_->target_params,
      .camera_params = params_->camera_params,
      .lidar_measurements = lidar_measurements_,
      .camera_measurements = camera_measurements_,
      .calibration_initials = calibrations_initial_,
      .optimizer_config_path = inputs_.optimizer_config};

  std::shared_ptr<Optimizer> optimizer;
  if (params_->optimizer_type == "GTSAM") {
    LOG_ERROR("GTSAM Optimizer not yet implemented. For untested "
              "implementation, see add_gtsam_optimizer branch on github.");
    throw std::invalid_argument{"Invalid optimizer type."};
  } else if (params_->optimizer_type == "CERES") {
    optimizer = std::make_shared<CeresOptimizer>(optimizer_inputs);
  } else {
    optimizer = std::make_shared<CeresOptimizer>(optimizer_inputs);
    LOG_WARN("Invalid optimizer_type parameter. Options: CERES. Using "
             "default: CERES");
  }

  optimizer->Solve();
  calibrations_final_ = optimizer->GetResults();
  target_camera_corrections_ = optimizer->GetTargetCameraCorrections();
  target_lidar_corrections_ = optimizer->GetTargetLidarCorrections();
}

void ViconCalibrator::RunCalibration() {
  Setup();
  GetInitialCalibrations();
  GetMeasurements();
  Solve();
  RunVerification();
}

void ViconCalibrator::RunVerification() {
  if (inputs_.verification_config == "NONE") {
    utils::OutputCalibrations(calibrations_initial_,
                              "Initial Calibration Estimates:");
    utils::OutputCalibrations(calibrations_final_, "Optimized Calibrations:");
    std::cout << "Outputting Target Camera Corrections:\n\n";
    utils::OutputTargetCorrections(target_camera_corrections_);
    std::cout << "Outputting Target Lidar Corrections:\n\n";
    utils::OutputTargetCorrections(target_lidar_corrections_);
    LOG_INFO("Skipping calibration verification since no config was input");
    return;
  }

  CalibrationVerification ver(inputs_.verification_config,
                              inputs_.output_directory,
                              inputs_.calibration_config);
  ver.SetParams(params_);
  ver.SetLidarMeasurements(lidar_measurements_);
  ver.SetCameraMeasurements(camera_measurements_);
  ver.SetInitialCalib(calibrations_initial_);
  ver.SetOptimizedCalib(calibrations_final_);
  ver.SetTargetCameraCorrections(target_camera_corrections_);
  ver.SetTargetLidarCorrections(target_lidar_corrections_);
  ver.ProcessResults();
}

void ViconCalibrator::OutputMeasurementStats() {
  std::cout << "------------------------------------------------------------\n"
            << "Outputing Measurement Statistics\n"
            << "--------------------------------\n";
  for (int i = 0; i < lidar_counters_.size(); i++) {
    std::cout << "Lidar " << i << ":\n\n"
              << "Total possible measurements: " << lidar_counters_.at(i).total
              << "\nSaved: " << lidar_counters_.at(i).accepted
              << "\nRejected - no movement: "
              << lidar_counters_.at(i).rejected_still
              << "\nRejected - high motion: "
              << lidar_counters_.at(i).rejected_fast
              << "\nRejected - invalid result: "
              << lidar_counters_.at(i).rejected_invalid << "\n";
  }
  std::cout << "\n";
  for (int i = 0; i < camera_counters_.size(); i++) {
    std::cout << "Camera " << i << ":\n\n"
              << "Total possible measurements: " << camera_counters_.at(i).total
              << "\nSaved: " << camera_counters_.at(i).accepted
              << "\nRejected - no movement: "
              << camera_counters_.at(i).rejected_still
              << "\nRejected - high motion: "
              << camera_counters_.at(i).rejected_fast
              << "\nRejected - invalid result: "
              << camera_counters_.at(i).rejected_invalid << "\n";
  }
  std::cout << "------------------------------------------------------------\n";
}

} // end namespace vicon_calibration
