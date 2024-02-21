#include <vicon_calibration/CalibrationVerification.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/concave_hull.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2_msgs/TFMessage.h>

#include <vicon_calibration/CropBox.h>
#include <vicon_calibration/PclConversions.h>
#include <vicon_calibration/Utils.h>
#include <vicon_calibration/measurement_extractors/CheckerboardCameraExtractor.h>
#include <vicon_calibration/measurement_extractors/CheckerboardLidarExtractor.h>
#include <vicon_calibration/measurement_extractors/CylinderCameraExtractor.h>
#include <vicon_calibration/measurement_extractors/CylinderLidarExtractor.h>

namespace vicon_calibration {

namespace fs = std::filesystem;

CalibrationVerification::CalibrationVerification(
    const std::string& config_file_name, const std::string& output_directory,
    const std::string& calibration_config)
    : config_file_name_(config_file_name),
      output_directory_(output_directory),
      calibration_config_(calibration_config) {}

void CalibrationVerification::LoadJSON() {
  LOG_INFO("Loading CalibrationVerification Config File: %s",
           config_file_name_.c_str());
  nlohmann::json J;
  if (!utils::ReadJson(config_file_name_, J)) {
    LOG_ERROR("Using default calibration verification params.");
    return;
  }

  try {
    max_image_results_ = J["max_image_results"];
    max_lidar_results_ = J["max_lidar_results"];
    max_pixel_cor_dist_ = J["max_pixel_cor_dist"];
    max_point_cor_dist_ = J["max_point_cor_dist"];
    concave_hull_alpha_multiplier_ = J["concave_hull_alpha_multiplier"];
    show_target_outline_on_image_ = J["show_target_outline_on_image"];
    keypoint_circle_diameter_ = J["keypoint_circle_diameter"];
    outline_circle_diameter_ = J["outline_circle_diameter"];
  } catch (const nlohmann::json::exception& e) {
    LOG_ERROR("Cannot load json, one or more missing parameters. Error: %s",
              e.what());
  }
}

void CalibrationVerification::CheckInputs() {
  if (!optimized_calib_set_) {
    throw std::invalid_argument{"Optimized Calibration Not Set."};
  }
  if (!initial_calib_set_) {
    throw std::invalid_argument{"Initial Calibration Not Set."};
  }
  if (!params_set_) {
    throw std::invalid_argument{"Calibrator Params Not Set."};
  }
  if (!lidar_measurements_set_) {
    throw std::invalid_argument{"Lidar Measurements Not Set."};
  }
  if (!camera_measurements_set_) {
    throw std::invalid_argument{"Camera Measurements Not Set."};
  }
}

void CalibrationVerification::ProcessResults(bool save_measurements) {
  LOG_INFO("Processing calibration results.");
  CheckInputs();
  LoadJSON();

  // load bag file
  try {
    bag_.open(params_->bag_file, rosbag::bagmode::Read);
  } catch (rosbag::BagException& ex) {
    LOG_ERROR("Bag exception : %s", ex.what());
  }

  CreateDirectories();
  PrintConfig();
  PrintCalibrations(calibrations_initial_, "initial_calibrations.txt");
  PrintCalibrations(calibrations_result_, "optimized_calibrations.txt");
  if (ground_truth_calib_set_) {
    PrintCalibrations(calibrations_ground_truth_,
                      "ground_truth_calibrations.txt");
  }
  PrintTargetCorrections("optimized_calibrations.txt");
  PrintCalibrationErrors();
  if (save_measurements) {
    SaveLidarVisuals();
    SaveCameraVisuals();
  }
  GetLidarErrors();
  GetCameraErrors();
  PrintErrorsSummary();
  LOG_INFO("CalibrationVerification Complete.");
  bag_.close();
}

CalibrationVerification::Results CalibrationVerification::GetSummary() {
  if (!results_.ground_truth_set) {
    LOG_WARN("Ground truth calibrations not set. Translation and rotations "
             "errors are unavailable.");
  }
  return results_;
}

void CalibrationVerification::SetInitialCalib(
    const std::vector<vicon_calibration::CalibrationResult>& calib) {
  calibrations_initial_ = calib;
  initial_calib_set_ = true;
}

// TODO: Remove all GT calls. This was for simulation testing
void CalibrationVerification::SetGroundTruthCalib(
    const std::vector<vicon_calibration::CalibrationResult>& calib) {
  calibrations_ground_truth_ = calib;
  ground_truth_calib_set_ = true;
}

void CalibrationVerification::SetOptimizedCalib(
    const std::vector<vicon_calibration::CalibrationResult>& calib) {
  calibrations_result_ = calib;
  optimized_calib_set_ = true;
}

void CalibrationVerification::SetTargetCorrections(
    const std::vector<Eigen::Matrix4d>& corrections) {
  target_corrections_ = corrections;
}

void CalibrationVerification::SetParams(
    std::shared_ptr<CalibratorConfig>& params) {
  params_ = params;
  params_set_ = true;

  // Downsample template cloud
  pcl::VoxelGrid<pcl::PointXYZ> vox;
  vox.setLeafSize(template_downsample_size_[0], template_downsample_size_[1],
                  template_downsample_size_[2]);
  for (int i = 0; i < params_->target_params.size(); i++) {
    std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> downsampled_cloud =
        std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    vox.setInputCloud(params_->target_params[i]->template_cloud);
    vox.filter(*downsampled_cloud);
    params_->target_params[i]->template_cloud = downsampled_cloud;
  }
}

void CalibrationVerification::SetLidarMeasurements(
    const std::vector<std::vector<LidarMeasurementPtr>>& lidar_measurements) {
  lidar_measurements_ = lidar_measurements;
  lidar_measurements_set_ = true;
}

void CalibrationVerification::SetCameraMeasurements(
    const std::vector<std::vector<CameraMeasurementPtr>>& camera_measurements) {
  camera_measurements_ = camera_measurements;
  camera_measurements_set_ = true;
}

void CalibrationVerification::CreateDirectories() {
  date_and_time_ = utils::ConvertTimeToDate(std::chrono::system_clock::now());
  results_directory_ =
      (fs::path(output_directory_) / fs::path(date_and_time_)).string();
  fs::create_directory(results_directory_);
  fs::create_directory(fs::path(results_directory_) / fs::path("CAMERAS"));
  fs::create_directory(fs::path(results_directory_) / fs::path("LIDARS"));
  LOG_INFO("Saving results to: %s", results_directory_.c_str());
}

void CalibrationVerification::PrintCalibrations(
    std::vector<vicon_calibration::CalibrationResult>& calib,
    const std::string& file_name) {
  fs::path output_path = fs::path(results_directory_) / fs::path(file_name);
  std::ofstream file(output_path);
  for (uint16_t i = 0; i < calib.size(); i++) {
    Eigen::Matrix4d T = calib[i].transform;
    Eigen::Matrix3d R = T.block(0, 0, 3, 3);
    Eigen::Vector3d rpy = R.eulerAngles(0, 1, 2);
    file << "T_" << calib[i].to_frame << "_" << calib[i].from_frame << ":\n"
         << T << "\n"
         << "rpy (deg): [" << utils::RadToDeg(utils::WrapToTwoPi(rpy[0]))
         << ", " << utils::RadToDeg(utils::WrapToTwoPi(rpy[1])) << ", "
         << utils::RadToDeg(utils::WrapToTwoPi(rpy[2])) << "]\n";
  }
}

void CalibrationVerification::PrintTargetCorrections(
    const std::string& file_name) {
  fs::path output_path = fs::path(results_directory_) / fs::path(file_name);
  std::ofstream file(output_path);
  for (const auto& T : target_corrections_) {
    Eigen::Matrix3d R = T.block(0, 0, 3, 3);
    Eigen::Vector3d rpy = R.eulerAngles(0, 1, 2);
    file << "T_TargetCorrected_Target\n"
         << "rpy (deg): [" << utils::RadToDeg(utils::WrapToTwoPi(rpy[0]))
         << ", " << utils::RadToDeg(utils::WrapToTwoPi(rpy[1])) << ", "
         << utils::RadToDeg(utils::WrapToTwoPi(rpy[2])) << "]\n\n";
  }
}

std::string CalibrationVerification::CalibrationErrorsToString(
    const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2,
    const std::string& from_frame, const std::string& to_frame) {
  Eigen::Matrix3d R2 = T2.block(0, 0, 3, 3);
  Eigen::Matrix3d R1 = T1.block(0, 0, 3, 3);
  Eigen::Vector3d rpy2 = R2.eulerAngles(0, 1, 2);
  Eigen::Vector3d rpy1 = R1.eulerAngles(0, 1, 2);
  Eigen::Vector3d rpy_error;
  rpy_error[0] = utils::GetSmallestAngleErrorRad(rpy2[0], rpy1[0]);
  rpy_error[1] = utils::GetSmallestAngleErrorRad(rpy2[1], rpy1[1]);
  rpy_error[2] = utils::GetSmallestAngleErrorRad(rpy2[2], rpy1[2]);
  Eigen::Vector3d t_error = T2.block(0, 3, 3, 1) - T1.block(0, 3, 3, 1);
  t_error[0] = std::abs(t_error[0]);
  t_error[1] = std::abs(t_error[1]);
  t_error[2] = std::abs(t_error[2]);
  std::string output;
  output += "T_" + to_frame + "_" + from_frame + ":\n" + "rpy error (deg): [" +
            std::to_string(utils::RadToDeg(rpy_error[0])) + ", " +
            std::to_string(utils::RadToDeg(rpy_error[1])) + ", " +
            std::to_string(utils::RadToDeg(rpy_error[2])) + "]\n" +
            "translation error (mm): [" + std::to_string(t_error[0] * 1000) +
            ", " + std::to_string(t_error[1] * 1000) + ", " +
            std::to_string(t_error[2] * 1000) + "]\n\n";
  return output;
}

void CalibrationVerification::PrintCalibrationErrors() {
  fs::path output_path =
      fs::path(results_directory_) / fs::path("calibration_errors.txt");
  std::ofstream file(output_path);
  // first print errors between initial calibration and final
  file << "Showing errors between:\n"
       << "initial calibration estimates and optimized calibrations:\n\n";
  for (uint16_t i = 0; i < calibrations_result_.size(); i++) {
    file << CalibrationErrorsToString(
        calibrations_result_[i].transform, calibrations_initial_[i].transform,
        calibrations_result_[i].from_frame, calibrations_result_[i].to_frame);
  }

  if (!ground_truth_calib_set_) { return; }

  // next print errors between ground truth calibrations and optimized
  file << "---------------------------------------------------------\n\n"
       << "Showing errors between:\n"
       << "initial ground truth calibrations and optimized calibrations:\n\n";
  for (uint16_t i = 0; i < calibrations_result_.size(); i++) {
    file << CalibrationErrorsToString(calibrations_result_[i].transform,
                                      calibrations_ground_truth_[i].transform,
                                      calibrations_result_[i].from_frame,
                                      calibrations_result_[i].to_frame);
  }

  // calculate results summary
  results_.ground_truth_set = ground_truth_calib_set_;
  results_.calibration_translation_errors_mm.clear();
  results_.calibration_rotation_errors_deg.clear();
  for (uint16_t i = 0; i < calibrations_result_.size(); i++) {
    results_.calibration_translation_errors_mm.push_back(
        1000 * utils::CalculateTranslationErrorNorm(
                   calibrations_result_[i].transform.block(0, 3, 3, 1),
                   calibrations_ground_truth_[i].transform.block(0, 3, 3, 1)));
    results_.calibration_rotation_errors_deg.push_back(
        utils::RadToDeg(utils::CalculateRotationError(
            calibrations_result_[i].transform.block(0, 0, 3, 3),
            calibrations_ground_truth_[i].transform.block(0, 0, 3, 3))));
  }

  // next print errors between initial calibrations and ground truth calibration
  file << "---------------------------------------------------------\n\n"
       << "Showing errors between:\n"
       << "initial calibrations and ground truth calibrations:\n\n";
  for (uint16_t i = 0; i < calibrations_result_.size(); i++) {
    file << CalibrationErrorsToString(calibrations_initial_[i].transform,
                                      calibrations_ground_truth_[i].transform,
                                      calibrations_initial_[i].from_frame,
                                      calibrations_initial_[i].to_frame);
  }
}

void CalibrationVerification::PrintConfig() {
  nlohmann::json J_in;
  std::ifstream file_in(calibration_config_);
  std::ofstream file_out(results_directory_ + "ViconCalibratorConfig.json");
  file_in >> J_in;
  file_out << std::setw(4) << J_in << std::endl;
}

void CalibrationVerification::SaveLidarVisuals() {
  // iterate over each lidar
  for (uint8_t lidar_iter = 0; lidar_iter < params_->lidar_params.size();
       lidar_iter++) {
    int counter = 0;

    // create directories
    fs::path current_save_path =
        fs::path(results_directory_) / fs::path("LIDARS") /
        fs::path(params_->lidar_params[lidar_iter]->frame);
    fs::create_directory(current_save_path);

    // get lidar info
    std::string topic = params_->lidar_params[lidar_iter]->topic;
    std::string sensor_frame = params_->lidar_params[lidar_iter]->frame;

    // get initial calibration and optimized calibration
    Eigen::Affine3d TA_Robot_Sensor_est, TA_Robot_Sensor_opt;
    for (CalibrationResult calib : calibrations_initial_) {
      if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
        TA_Robot_Sensor_est.matrix() = calib.transform;
        break;
      }
    }

    for (CalibrationResult calib : calibrations_result_) {
      if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
        TA_Robot_Sensor_opt.matrix() = calib.transform;
        break;
      }
    }

    // iterate through all measurements for this lidar
    for (int meas_iter = 0; meas_iter < lidar_measurements_[lidar_iter].size();
         meas_iter++) {
      counter++;
      if (lidar_measurements_[lidar_iter][meas_iter] == nullptr) { continue; }
      LidarMeasurementPtr measurement =
          lidar_measurements_[lidar_iter][meas_iter];
      lookup_time_ = measurement->time_stamp;
      LoadLookupTree();

      // load scan and transform to viconbase frame
      PointCloud::Ptr scan = GetLidarScanFromBag(
          params_->lidar_params[measurement->lidar_id]->topic);
      PointCloud::Ptr scan_trans_est = std::make_shared<PointCloud>();
      PointCloud::Ptr scan_trans_opt = std::make_shared<PointCloud>();
      pcl::transformPointCloud(*scan, *scan_trans_est, TA_Robot_Sensor_est);
      pcl::transformPointCloud(*scan, *scan_trans_opt, TA_Robot_Sensor_opt);

      // load targets and transform to viconbase frame
      std::vector<Eigen::Affine3d> T_Robot_Targets;
      try {
        T_Robot_Targets = utils::GetTargetLocation(
            params_->target_params, params_->vicon_baselink_frame, lookup_time_,
            lookup_tree_);
      } catch (const std::runtime_error err) {
        LOG_ERROR("%s", err.what());
        continue;
      }

      PointCloud::Ptr targets_combined = std::make_shared<PointCloud>();
      for (uint8_t n = 0; n < T_Robot_Targets.size(); n++) {
        const PointCloud::Ptr target =
            params_->target_params[n]->template_cloud;
        PointCloud::Ptr target_transformed = std::make_shared<PointCloud>();
        pcl::transformPointCloud(*target, *target_transformed,
                                 T_Robot_Targets[n]);
        *targets_combined = *targets_combined + *target_transformed;
      }
      SaveScans(scan_trans_est, scan_trans_opt, targets_combined,
                current_save_path.string(), counter);
      if (counter == max_lidar_results_) { continue; }
    }
  }
}

PointCloud::Ptr
    CalibrationVerification::GetLidarScanFromBag(const std::string& topic) {
  ros::Duration time_window_half = ros::Duration(0.5);
  rosbag::View view(bag_, rosbag::TopicQuery(topic),
                    lookup_time_ - time_window_half,
                    lookup_time_ + time_window_half, true);
  pcl::PCLPointCloud2::Ptr cloud_pc2 = std::make_shared<pcl::PCLPointCloud2>();
  PointCloud::Ptr scan = std::make_shared<PointCloud>();
  for (auto iter = view.begin(); iter != view.end(); iter++) {
    auto lidar_msg = iter->instantiate<sensor_msgs::PointCloud2>();
    if (lidar_msg->header.stamp >= lookup_time_) {
      pcl_conversions::toPCL(*lidar_msg, *cloud_pc2);
      pcl::fromPCLPointCloud2(*cloud_pc2, *scan);
      return scan;
    }
  }
  throw std::runtime_error{"Cannot get lidar scan from bag."};
}

void CalibrationVerification::SaveScans(const PointCloud::Ptr& scan_est,
                                        const PointCloud::Ptr& scan_opt,
                                        const PointCloud::Ptr& targets,
                                        const std::string& save_path,
                                        const int& scan_count) {
  std::string save_path_full =
      save_path + "scan_" + std::to_string(scan_count) + ".pcd";
  PointCloud::Ptr scan_est_cropped = std::make_shared<PointCloud>();
  PointCloud::Ptr scan_opt_cropped = std::make_shared<PointCloud>();
  CropBox cropper;
  Eigen::Vector3f min{-10, -10, -10}, max{10, 10, 10};
  cropper.SetMinVector(min);
  cropper.SetMaxVector(max);
  cropper.Filter(*scan_est, *scan_est_cropped);
  cropper.Filter(*scan_opt, *scan_opt_cropped);

  PointCloudColor::Ptr scan_est_colored =
      utils::ColorPointCloud(scan_est_cropped, 255, 0, 0);
  PointCloudColor::Ptr scan_opt_colored =
      utils::ColorPointCloud(scan_opt_cropped, 0, 255, 0);
  PointCloudColor::Ptr targets_colored =
      utils::ColorPointCloud(targets, 0, 0, 255);
  PointCloudColor cloud_combined = *scan_est_colored;
  cloud_combined += *scan_opt_colored;
  cloud_combined += *targets_colored;
  pcl::io::savePCDFileBinary(save_path_full, cloud_combined);
}

void CalibrationVerification::GetLidarErrors() {
  // iterate over each lidar
  for (uint8_t lidar_iter = 0; lidar_iter < params_->lidar_params.size();
       lidar_iter++) {
    // get initial calibration and optimized calibration
    Eigen::Affine3d TA_Robot_Sensor_est, TA_Robot_Sensor_true,
        TA_Robot_Sensor_opt;
    if (ground_truth_calib_set_) {
      for (CalibrationResult calib : calibrations_ground_truth_) {
        if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
          TA_Robot_Sensor_true.matrix() = calib.transform;
          break;
        }
      }
    }
    for (CalibrationResult calib : calibrations_initial_) {
      if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
        TA_Robot_Sensor_est.matrix() = calib.transform;
        break;
      }
    }
    for (CalibrationResult calib : calibrations_result_) {
      if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
        TA_Robot_Sensor_opt.matrix() = calib.transform;
        break;
      }
    }

    // iterate through all measurements for this lidar
    LidarMeasurementPtr measurement;
    PointCloud::Ptr measured_keypoints, estimated_keypoints_target;
    PointCloud::Ptr estimated_keypoints_est = std::make_shared<PointCloud>();
    PointCloud::Ptr estimated_keypoints_opt = std::make_shared<PointCloud>();
    PointCloud::Ptr estimated_keypoints_true = std::make_shared<PointCloud>();
    Eigen::Matrix4d T_Sensor_Target_opt;
    Eigen::Matrix4d T_Sensor_Target_est;
    Eigen::Matrix4d T_Sensor_Target_true;
    std::vector<Eigen::Vector3d> lidar_errors_opt;
    std::vector<Eigen::Vector3d> lidar_errors_init;
    std::vector<Eigen::Vector3d> lidar_errors_true;
    for (int meas_iter = 0; meas_iter < lidar_measurements_[lidar_iter].size();
         meas_iter++) {
      if (lidar_measurements_[lidar_iter][meas_iter] == nullptr) { continue; }
      measurement = lidar_measurements_[lidar_iter][meas_iter];

      T_Sensor_Target_opt =
          TA_Robot_Sensor_opt.inverse().matrix() * measurement->T_Robot_Target;
      T_Sensor_Target_est =
          TA_Robot_Sensor_est.inverse().matrix() * measurement->T_Robot_Target;

      // get estimated keypoints given calibrations
      estimated_keypoints_target = std::make_shared<PointCloud>();
      const auto& kpts =
          params_->target_params[measurement->target_id]->keypoints_lidar;
      int num_keypoints = kpts.cols();
      for (int k = 0; k < num_keypoints; k++) {
        pcl::PointXYZ p(kpts(0, k), kpts(1, k), kpts(2, k));
        estimated_keypoints_target->push_back(p);
      }
      if (num_keypoints == 0) {
        if (params_->target_params[measurement->target_id]->template_cloud ==
            nullptr) {
          throw std::runtime_error{"No lidar keypoints available"};
        }
        estimated_keypoints_target =
            params_->target_params[measurement->target_id]->template_cloud;
      }

      pcl::transformPointCloud(*estimated_keypoints_target,
                               *estimated_keypoints_est, T_Sensor_Target_est);
      pcl::transformPointCloud(*estimated_keypoints_target,
                               *estimated_keypoints_opt, T_Sensor_Target_opt);

      lidar_errors_init =
          CalculateLidarErrors(measurement->keypoints, estimated_keypoints_est);
      lidar_errors_opt =
          CalculateLidarErrors(measurement->keypoints, estimated_keypoints_opt);

      lidar_errors_opt_.insert(lidar_errors_opt_.end(),
                               lidar_errors_opt.begin(),
                               lidar_errors_opt.end());

      lidar_errors_init_.insert(lidar_errors_init_.end(),
                                lidar_errors_init.begin(),
                                lidar_errors_init.end());

      if (ground_truth_calib_set_) {
        T_Sensor_Target_true = TA_Robot_Sensor_true.inverse().matrix() *
                               measurement->T_Robot_Target;
        pcl::transformPointCloud(*estimated_keypoints_target,
                                 *estimated_keypoints_true,
                                 T_Sensor_Target_true);
        lidar_errors_true = CalculateLidarErrors(measurement->keypoints,
                                                 estimated_keypoints_true);

        lidar_errors_true_.insert(lidar_errors_true_.end(),
                                  lidar_errors_true.begin(),
                                  lidar_errors_true.end());
      }

    } // measurement iter
  }   // lidar iter
}

std::vector<Eigen::Vector3d> CalibrationVerification::CalculateLidarErrors(
    const PointCloud::Ptr& measured_keypoints,
    const PointCloud::Ptr& estimated_keypoints) {
  // get correspondences
  corr_est_.setInputSource(measured_keypoints);
  corr_est_.setInputTarget(estimated_keypoints);
  corr_est_.determineCorrespondences(*correspondences_, max_point_cor_dist_);

  // get distances between correspondences
  std::vector<Eigen::Vector3d> lidar_errors;
  for (int i = 0; i < correspondences_->size(); i++) {
    int measurement_index = correspondences_->at(i).index_query;
    int estimated_index = correspondences_->at(i).index_match;
    const auto& p1 = measured_keypoints->at(measurement_index);
    const auto& p2 = estimated_keypoints->at(estimated_index);
    float ex = p1.x - p2.x;
    float ey = p1.y - p2.y;
    float ez = p1.z - p2.z;
    lidar_errors.emplace_back(std::abs(ex), std::abs(ey), std::abs(ez));
  }
  return lidar_errors;
}

void CalibrationVerification::SaveCameraVisuals() {
  // Iterate over each camera
  for (uint8_t cam_iter = 0; cam_iter < params_->camera_params.size();
       cam_iter++) {
    fs::path current_save_path =
        fs::path(results_directory_) / fs::path("CAMERAS") /
        fs::path(params_->camera_params[cam_iter]->frame);
    fs::create_directory(current_save_path);
    int counter = 0;
    std::string topic = params_->camera_params[cam_iter]->topic;
    std::string sensor_frame = params_->camera_params[cam_iter]->frame;
    std::vector<Eigen::Affine3d> T_cam_tgts_estimated_prev;
    rosbag::View view(bag_, rosbag::TopicQuery(topic), ros::TIME_MIN,
                      ros::TIME_MAX, true);

    // get initial calibration and optimized calibration
    Eigen::Affine3d TA_Robot_Sensor_est;
    Eigen::Affine3d TA_Robot_Sensor_opt;
    Eigen::Affine3d TA_Robot_Sensor_true;
    if (ground_truth_calib_set_) {
      for (CalibrationResult calib : calibrations_ground_truth_) {
        if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
          TA_Robot_Sensor_true.matrix() = calib.transform;
          break;
        }
      }
    }
    for (CalibrationResult calib : calibrations_initial_) {
      if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
        TA_Robot_Sensor_est.matrix() = calib.transform;
        break;
      }
    }
    for (CalibrationResult calib : calibrations_result_) {
      if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
        TA_Robot_Sensor_opt.matrix() = calib.transform;
        break;
      }
    }

    // iterate through all measurements for this camera
    for (int meas_iter = 0; meas_iter < camera_measurements_[cam_iter].size();
         meas_iter++) {
      if (camera_measurements_[cam_iter][meas_iter] == nullptr) { continue; }

      CameraMeasurementPtr measurement =
          camera_measurements_[cam_iter][meas_iter];
      lookup_time_ = measurement->time_stamp;
      LoadLookupTree();

      // load image from bag
      std::shared_ptr<cv::Mat> current_image = GetImageFromBag(
          params_->camera_params[measurement->camera_id]->topic);

      // convert to color if not already
      if (current_image->channels() != 3) {
#if CV_VERSION_MAJOR >= 4
        cv::cvtColor(*current_image, *current_image, cv::COLOR_GRAY2BGR);
#else
        cv::cvtColor(*current_image, *current_image, CV_GRAY2BGR);
#endif
      }

      // load targets and transform to robot frame
      std::vector<Eigen::Affine3d> T_Robot_Targets;
      try {
        T_Robot_Targets = utils::GetTargetLocation(
            params_->target_params, params_->vicon_baselink_frame, lookup_time_,
            lookup_tree_);
      } catch (const std::runtime_error err) {
        LOG_ERROR("%s", err.what());
        continue;
      }

      // Add measurements to image
      std::shared_ptr<cv::Mat> final_image = ProjectTargetToImage(
          current_image, T_Robot_Targets, TA_Robot_Sensor_est.matrix(),
          cam_iter, cv::Scalar(0, 0, 255));

      if (num_tgts_in_img_ == 0) { continue; }

      counter++;
      final_image = ProjectTargetToImage(final_image, T_Robot_Targets,
                                         TA_Robot_Sensor_opt.matrix(), cam_iter,
                                         cv::Scalar(255, 0, 0));

      if (ground_truth_calib_set_) {
        final_image = ProjectTargetToImage(final_image, T_Robot_Targets,
                                           TA_Robot_Sensor_true.matrix(),
                                           cam_iter, cv::Scalar(0, 255, 0));
      }

      // save image with targets
      fs::path save_path =
          current_save_path /
          fs::path("image_" + std::to_string(counter) + ".jpg");
      cv::imwrite(save_path.string(), *final_image);

      if (counter == max_image_results_) { break; }
    }
  }
}

std::shared_ptr<cv::Mat>
    CalibrationVerification::GetImageFromBag(const std::string& topic) {
  ros::Duration time_window_half = ros::Duration(0.5);
  rosbag::View view(bag_, rosbag::TopicQuery(topic),
                    lookup_time_ - time_window_half,
                    lookup_time_ + time_window_half, true);
  sensor_msgs::ImageConstPtr image_msg;
  std::shared_ptr<cv::Mat> image = std::make_shared<cv::Mat>();
  for (auto iter = view.begin(); iter != view.end(); iter++) {
    image_msg = iter->instantiate<sensor_msgs::Image>();
    if (image_msg->header.stamp >= lookup_time_) {
      *image = utils::RosImgToMat(*image_msg);
      return image;
    }
  }
  throw std::runtime_error{"Cannot get image from bag."};
}

void CalibrationVerification::GetCameraErrors() {
  // iterate over each camera
  for (uint8_t cam_iter = 0; cam_iter < params_->camera_params.size();
       cam_iter++) {
    // get initial calibration and optimized calibration
    Eigen::Affine3d TA_Robot_Sensor_est;
    Eigen::Affine3d TA_Robot_Sensor_true;
    Eigen::Affine3d TA_Robot_Sensor_opt;
    if (ground_truth_calib_set_) {
      for (CalibrationResult calib : calibrations_ground_truth_) {
        if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
          TA_Robot_Sensor_true.matrix() = calib.transform;
          break;
        }
      }
    }
    for (CalibrationResult calib : calibrations_initial_) {
      if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
        TA_Robot_Sensor_est.matrix() = calib.transform;
        break;
      }
    }
    for (CalibrationResult calib : calibrations_result_) {
      if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
        TA_Robot_Sensor_opt.matrix() = calib.transform;
        break;
      }
    }

    // iterate through all measurements for this camera
    for (int meas_iter = 0; meas_iter < camera_measurements_[cam_iter].size();
         meas_iter++) {
      if (camera_measurements_[cam_iter][meas_iter] == nullptr) { continue; }
      CameraMeasurementPtr measurement =
          camera_measurements_[cam_iter][meas_iter];

      // convert 2d measured keypoints to 3d
      PointCloud::Ptr measured_keypoints_3d = std::make_shared<PointCloud>();
      pcl::PointCloud<pcl::PointXY>::Ptr measured_keypoints_2d =
          measurement->keypoints;
      pcl::PointXYZ point3d{0, 0, 0};
      for (int i = 0; i < measured_keypoints_2d->size(); i++) {
        point3d.x = measured_keypoints_2d->at(i).x;
        point3d.y = measured_keypoints_2d->at(i).y;
        measured_keypoints_3d->push_back(point3d);
      }

      Eigen::Matrix4d T_Sensor_Target_opt =
          TA_Robot_Sensor_opt.inverse().matrix() * measurement->T_Robot_Target;
      Eigen::Matrix4d T_Sensor_Target_est =
          TA_Robot_Sensor_est.inverse().matrix() * measurement->T_Robot_Target;
      std::vector<Eigen::Vector2d> camera_errors_opt =
          CalculateCameraErrors(measured_keypoints_3d, T_Sensor_Target_opt,
                                measurement->target_id, measurement->camera_id);
      std::vector<Eigen::Vector2d> camera_errors_init =
          CalculateCameraErrors(measured_keypoints_3d, T_Sensor_Target_est,
                                measurement->target_id, measurement->camera_id);

      camera_errors_opt_.insert(camera_errors_opt_.end(),
                                camera_errors_opt.begin(),
                                camera_errors_opt.end());

      camera_errors_init_.insert(camera_errors_init_.end(),
                                 camera_errors_init.begin(),
                                 camera_errors_init.end());

      if (ground_truth_calib_set_) {
        Eigen::Matrix4d T_Sensor_Target_true =
            TA_Robot_Sensor_true.inverse().matrix() *
            measurement->T_Robot_Target;
        std::vector<Eigen::Vector2d> camera_errors_true = CalculateCameraErrors(
            measured_keypoints_3d, T_Sensor_Target_true, measurement->target_id,
            measurement->camera_id);

        camera_errors_true_.insert(camera_errors_true_.end(),
                                   camera_errors_true.begin(),
                                   camera_errors_true.end());
      }
    } // measurement iter
  }   // camera iter
}

std::vector<Eigen::Vector2d> CalibrationVerification::CalculateCameraErrors(
    const PointCloud::Ptr& measured_keypoints,
    const Eigen::Matrix4d& T_Sensor_Target, const int& target_id,
    const int& camera_id) {
  // get estimated (optimization or initial) keypoint locations
  PointCloud::Ptr keypoints_target_frame = std::make_shared<PointCloud>();
  int num_keypoints =
      params_->target_params[target_id]->keypoints_camera.cols();
  for (int k = 0; k < num_keypoints; k++) {
    pcl::PointXYZ p(params_->target_params[target_id]->keypoints_camera(0, k),
                    params_->target_params[target_id]->keypoints_camera(1, k),
                    params_->target_params[target_id]->keypoints_camera(2, k));
    keypoints_target_frame->push_back(p);
  }
  if (num_keypoints == 0) {
    if (params_->target_params[target_id]->template_cloud == nullptr) {
      throw std::runtime_error{"Cannot instantiate target keypoints, no input "
                               "keypoints or template cloud."};
    }
    keypoints_target_frame = params_->target_params[target_id]->template_cloud;
  }

  // project points to image plane and save as cloud
  PointCloud::Ptr keypoints_projected = utils::ProjectPoints(
      keypoints_target_frame, params_->camera_params[camera_id]->camera_model,
      T_Sensor_Target);

  // get correspondences
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
      corr_est;
  std::shared_ptr<pcl::Correspondences> correspondences =
      std::make_shared<pcl::Correspondences>();
  corr_est.setInputSource(measured_keypoints);
  corr_est.setInputTarget(keypoints_projected);
  corr_est.determineCorrespondences(*correspondences, max_pixel_cor_dist_);

  // get distances between correspondences
  int measurement_index, estimated_index;
  Eigen::Vector3d error3d;
  Eigen::Vector2d error2d;
  std::vector<Eigen::Vector2d> camera_errors;
  for (int i = 0; i < correspondences->size(); i++) {
    measurement_index = correspondences->at(i).index_query;
    estimated_index = correspondences->at(i).index_match;
    const auto& p1 = measured_keypoints->at(measurement_index);
    const auto& p2 = keypoints_projected->at(estimated_index);
    float ex = p1.x - p2.x;
    float ey = p1.y - p2.y;
    camera_errors.emplace_back(std::abs(ex), std::abs(ey));
  }
  return camera_errors;
}

std::shared_ptr<cv::Mat> CalibrationVerification::ProjectTargetToImage(
    const std::shared_ptr<cv::Mat>& img_in,
    const std::vector<Eigen::Affine3d>& T_Robot_Targets,
    const Eigen::Matrix4d& T_Robot_Sensor, const int& cam_iter,
    cv::Scalar colour) {
  // create all objects we'll need
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected =
      std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  Eigen::Vector4d point_transformed;
  Eigen::Vector4d point_target;
  pcl::PointXYZ point_pcl_projected;
  std::shared_ptr<cv::Mat> img_out = std::make_shared<cv::Mat>();
  *img_out = img_in->clone();

  // iterate through all targets
  Eigen::Affine3d T_Robot_Target;
  num_tgts_in_img_ = 0;
  for (int target_iter = 0; target_iter < T_Robot_Targets.size();
       target_iter++) {
    T_Robot_Target = T_Robot_Targets[target_iter];
    // check if target origin is in camera frame
    point_target = Eigen::Vector4d(0, 0, 0, 1);
    point_transformed =
        utils::InvertTransform(T_Robot_Sensor) * T_Robot_Target * point_target;
    bool origin_projection_valid;
    Eigen::Vector2d origin_projected;
    params_->camera_params[cam_iter]->camera_model->ProjectPoint(
        point_transformed.hnormalized(), origin_projected,
        origin_projection_valid);

    if (!origin_projection_valid) { continue; }

    num_tgts_in_img_++;

    // add keypoints to image
    cv::Point point_cv_projected;
    const auto& kpts = params_->target_params[target_iter]->keypoints_camera;
    for (int k = 0; k < kpts.cols(); k++) {
      Eigen::Vector3d point = kpts.col(k);
      point_target = point.homogeneous();
      point_transformed = utils::InvertTransform(T_Robot_Sensor) *
                          T_Robot_Target * point_target;
      bool point_projection_valid;
      Eigen::Vector2d point_projected;
      params_->camera_params[cam_iter]->camera_model->ProjectPoint(
          point_transformed.hnormalized(), point_projected,
          point_projection_valid);

      if (point_projection_valid) {
        cv::Point pixel_cv;
        pixel_cv.x = point_projected[0];
        pixel_cv.y = point_projected[1];
        cv::circle(*img_out, pixel_cv, keypoint_circle_diameter_, colour);
      }
    }

    if (!show_target_outline_on_image_) { return img_out; }
    // iterate through all target points
    pcl::PointCloud<pcl::PointXYZ>::Ptr target =
        params_->target_params[target_iter]->template_cloud;
    for (uint32_t i = 0; i < target->size(); i++) {
      point_target =
          Eigen::Vector4d(target->at(i).x, target->at(i).y, target->at(i).z, 1);
      point_transformed = utils::InvertTransform(T_Robot_Sensor) *
                          T_Robot_Target * point_target;
      bool point_projected_valid;
      Eigen::Vector2d point_projected;
      params_->camera_params[cam_iter]->camera_model->ProjectPoint(
          point_transformed.hnormalized(), point_projected,
          point_projected_valid);

      if (point_projected_valid) {
        point_pcl_projected.x = point_projected[0];
        point_pcl_projected.y = point_projected[1];
        point_pcl_projected.z = 0;
        cloud_projected->push_back(point_pcl_projected);
      }
    }

    // Get concave hull alpha
    // we want to make sure alpha is larget than two consecutive projected pts
    // assume template points are 5mm apart, calculate distance in pixels
    // between two consecutive projected points
    Eigen::Matrix4d T_Sensor_Target =
        utils::InvertTransform(T_Robot_Sensor) * T_Robot_Target.matrix();
    Eigen::Vector4d point1 = T_Sensor_Target * Eigen::Vector4d(0, 0, 0, 1);
    Eigen::Vector4d point2 = T_Sensor_Target * Eigen::Vector4d(0, 0, 0.005, 1);
    bool point1_projected_valid;
    Eigen::Vector2d point1_projected;
    params_->camera_params[cam_iter]->camera_model->ProjectPoint(
        point1.hnormalized(), point1_projected, point1_projected_valid);
    bool point2_projected_valid;
    Eigen::Vector2d point2_projected;
    params_->camera_params[cam_iter]->camera_model->ProjectPoint(
        point2.hnormalized(), point2_projected, point2_projected_valid);

    double distance = 3;
    if (point1_projected_valid && point2_projected_valid) {
      distance = (point1_projected - point2_projected).norm();
      // for really small distances, set minimum
      if (distance < 3) { distance = 3; }
    }

    // keep only perimeter points
    pcl::ConcaveHull<pcl::PointXYZ> concave_hull;
    concave_hull.setInputCloud(cloud_projected);
    concave_hull.setAlpha(concave_hull_alpha_multiplier_ * distance);
    concave_hull.reconstruct(*cloud_projected);

    // colour image
    point_cv_projected;
    for (uint32_t i = 0; i < cloud_projected->size(); i++) {
      point_pcl_projected = cloud_projected->at(i);
      point_cv_projected.x = point_pcl_projected.x;
      point_cv_projected.y = point_pcl_projected.y;
      cv::circle(*img_out, point_cv_projected, outline_circle_diameter_,
                 colour);
    }
  }
  return img_out;
}

void CalibrationVerification::LoadLookupTree() {
  lookup_tree_->Clear();
  ros::Duration time_window_half(1); // Check two second time window
  ros::Time start_time = lookup_time_ - time_window_half;
  ros::Time time_zero(0, 0);
  if (start_time <= time_zero) { start_time = time_zero; }
  ros::Time end_time = lookup_time_ + time_window_half;
  rosbag::View view(bag_, rosbag::TopicQuery("/tf"), start_time, end_time,
                    true);
  bool first_msg = true;
  for (const auto& msg_instance : view) {
    auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
    if (tf_message != nullptr) {
      for (geometry_msgs::TransformStamped tf : tf_message->transforms) {
        lookup_tree_->AddTransform(tf);
      }
    }
  }
}

void CalibrationVerification::PrintErrorsSummary() {
  std::string output_path = results_directory_ + "errors_summary.txt";
  std::ofstream file(output_path);
  file << "-----------------------------------------------------------\n"
       << "ERRORS SUMMARY \nfor bag: " << params_->bag_file
       << "\nprocessed on: " << date_and_time_ << "\n"
       << "-----------------------------------------------------------\n\n";

  // print lidar errors
  double norms_summed, norms_averaged;
  for (int i = 0; i < lidar_errors_opt_.size(); i++) {
    norms_summed += lidar_errors_opt_[i].norm();
  }
  norms_averaged = norms_summed / lidar_errors_opt_.size();

  file << "Outputting Error Statistics for Optimized Lidar Calibrations:\n"
       << "Average Error Norm (m): " << norms_averaged << "\n"
       << "Samples Used: " << lidar_errors_opt_.size() << "\n";

  // save to results summary
  results_.num_lidars = lidar_measurements_.size();
  results_.num_lidar_measurements = lidar_errors_opt_.size();
  results_.lidar_average_point_errors_mm = norms_averaged * 1000;

  norms_summed = 0;
  for (int i = 0; i < lidar_errors_init_.size(); i++) {
    norms_summed += lidar_errors_init_[i].norm();
  }
  norms_averaged = norms_summed / lidar_errors_init_.size();

  file << "\n-----------------------------------------------------------\n\n"
       << "Outputting Error Statistics for Initial Lidar Calibrations:\n"
       << "Average Error Norm (m): " << norms_averaged << "\n"
       << "Samples Used: " << lidar_errors_init_.size() << "\n";

  if (ground_truth_calib_set_) {
    norms_summed = 0;
    for (int i = 0; i < lidar_errors_true_.size(); i++) {
      norms_summed += lidar_errors_true_[i].norm();
    }
    norms_averaged = norms_summed / lidar_errors_true_.size();

    file << "\n-----------------------------------------------------------\n\n"
         << "Outputting Error Statistics for Ground Truth Lidar Calibrations:\n"
         << "Average Error Norm (m): " << norms_averaged << "\n"
         << "Samples Used: " << lidar_errors_true_.size() << "\n";
  }

  // print camera errors
  norms_summed = 0;
  for (int i = 0; i < camera_errors_opt_.size(); i++) {
    norms_summed += camera_errors_opt_[i].norm();
  }
  norms_averaged = norms_summed / camera_errors_opt_.size();

  file << "\n-----------------------------------------------------------\n\n"
       << "Outputting Error Statistics for Optimized Camera Calibrations:\n"
       << "Average Error Norm (pixels): " << norms_averaged << "\n"
       << "Samples Used: " << camera_errors_opt_.size() << "\n";

  // save to results summary
  results_.num_cameras = camera_measurements_.size();
  results_.num_camera_measurements = camera_errors_opt_.size();
  results_.camera_average_reprojection_errors_pixels = norms_averaged;

  norms_summed = 0;
  for (int i = 0; i < camera_errors_init_.size(); i++) {
    norms_summed += camera_errors_init_[i].norm();
  }
  norms_averaged = norms_summed / camera_errors_init_.size();

  file << "\n-----------------------------------------------------------\n\n"
       << "Outputting Error Statistics for Initial Camera Calibrations:\n"
       << "Average Error Norm (pixels): " << norms_averaged << "\n"
       << "Samples Used: " << camera_errors_init_.size() << "\n";

  if (ground_truth_calib_set_) {
    norms_summed = 0;
    for (int i = 0; i < camera_errors_true_.size(); i++) {
      norms_summed += camera_errors_true_[i].norm();
    }
    norms_averaged = norms_summed / camera_errors_true_.size();

    file
        << "\n-----------------------------------------------------------\n\n"
        << "Outputting Error Statistics for Ground Truth Camera Calibrations:\n"
        << "Average Error Norm (pixels): " << norms_averaged << "\n"
        << "Samples Used: " << camera_errors_true_.size() << "\n";
  }
}

} // end namespace vicon_calibration
