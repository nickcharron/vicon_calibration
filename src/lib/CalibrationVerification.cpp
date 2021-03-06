#include <vicon_calibration/CalibrationVerification.h>

#include <fstream>
#include <iostream>

#include <cv_bridge/cv_bridge.h>
#include <nlohmann/json.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/concave_hull.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2_msgs/TFMessage.h>

#include <vicon_calibration/CropBox.h>
#include <vicon_calibration/measurement_extractors/CylinderCameraExtractor.h>
#include <vicon_calibration/measurement_extractors/CylinderLidarExtractor.h>
#include <vicon_calibration/measurement_extractors/DiamondCameraExtractor.h>
#include <vicon_calibration/measurement_extractors/DiamondLidarExtractor.h>

namespace vicon_calibration {

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
  std::ifstream file(config_file_name_);
  file >> J;

  max_image_results_ = J["max_image_results"];
  max_lidar_results_ = J["max_lidar_results"];
  max_pixel_cor_dist_ = J["max_pixel_cor_dist"];
  max_point_cor_dist_ = J["max_point_cor_dist"];
  concave_hull_alpha_multiplier_ = J["concave_hull_alpha_multiplier"];
  show_target_outline_on_image_ = J["show_target_outline_on_image"];
  keypoint_circle_diameter_ = J["keypoint_circle_diameter"];
  outline_circle_diameter_ = J["outline_circle_diameter"];
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
  this->CheckInputs();
  this->LoadJSON();

  // load bag file
  try {
    bag_.open(params_->bag_file, rosbag::bagmode::Read);
  } catch (rosbag::BagException& ex) {
    LOG_ERROR("Bag exception : %s", ex.what());
  }

  this->CreateDirectories();
  this->PrintConfig();
  this->PrintCalibrations(calibrations_initial_, "initial_calibrations.txt");
  this->PrintCalibrations(calibrations_result_, "optimized_calibrations.txt");
  if (ground_truth_calib_set_) {
    this->PrintCalibrations(calibrations_ground_truth_,
                            "ground_truth_calibrations.txt");
  }
  this->PrintCalibrationErrors();
  if (save_measurements) {
    this->SaveLidarVisuals();
    this->SaveCameraVisuals();
  }
  this->GetLidarErrors();
  this->GetCameraErrors();
  this->PrintErrorsSummary();
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

// TODO: add checks for whether or not this was set. This
// should still work otherwise
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

void CalibrationVerification::SetParams(
    std::shared_ptr<CalibratorConfig>& params) {
  params_ = params;
  params_set_ = true;

  // Downsample template cloud
  pcl::VoxelGrid<pcl::PointXYZ> vox;
  vox.setLeafSize(template_downsample_size_[0], template_downsample_size_[1],
                  template_downsample_size_[2]);
  for (int i = 0; i < params_->target_params.size(); i++) {
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> downsampled_cloud =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    vox.setInputCloud(params_->target_params[i]->template_cloud);
    vox.filter(*downsampled_cloud);
    params_->target_params[i]->template_cloud = downsampled_cloud;
  }
}

void CalibrationVerification::SetLidarMeasurements(
    const std::vector<std::vector<std::shared_ptr<LidarMeasurement>>>&
        lidar_measurements) {
  lidar_measurements_ = lidar_measurements;
  lidar_measurements_set_ = true;
}

void CalibrationVerification::SetCameraMeasurements(
    const std::vector<std::vector<std::shared_ptr<CameraMeasurement>>>&
        camera_measurements) {
  camera_measurements_ = camera_measurements;
  camera_measurements_set_ = true;
}

void CalibrationVerification::CreateDirectories() {
  date_and_time_ = utils::ConvertTimeToDate(std::chrono::system_clock::now());
  results_directory_ = output_directory_ + date_and_time_ + "/";
  boost::filesystem::create_directory(results_directory_);
  boost::filesystem::create_directory(results_directory_ + "CAMERAS/");
  boost::filesystem::create_directory(results_directory_ + "LIDARS/");
  LOG_INFO("Saving results to: %s", results_directory_.c_str());
}

void CalibrationVerification::PrintCalibrations(
    std::vector<vicon_calibration::CalibrationResult>& calib,
    const std::string& file_name) {
  std::string output_path = results_directory_ + file_name;
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
  std::string output_path = results_directory_ + "calibration_errors.txt";
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
    std::string current_save_path = results_directory_ + "LIDARS/" +
                                    params_->lidar_params[lidar_iter]->frame +
                                    "/";
    boost::filesystem::create_directory(current_save_path);

    // get lidar info
    std::string topic = params_->lidar_params[lidar_iter]->topic;
    std::string sensor_frame = params_->lidar_params[lidar_iter]->frame;

    // get initial calibration and optimized calibration
    Eigen::Affine3d TA_VICONBASE_SENSOR_est, TA_VICONBASE_SENSOR_opt;
    for (CalibrationResult calib : calibrations_initial_) {
      if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
        TA_VICONBASE_SENSOR_est.matrix() = calib.transform;
        break;
      }
    }

    for (CalibrationResult calib : calibrations_result_) {
      if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
        TA_VICONBASE_SENSOR_opt.matrix() = calib.transform;
        break;
      }
    }

    // iterate through all measurements for this lidar
    std::shared_ptr<LidarMeasurement> measurement;
    for (int meas_iter = 0; meas_iter < lidar_measurements_[lidar_iter].size();
         meas_iter++) {
      counter++;
      if (lidar_measurements_[lidar_iter][meas_iter] == nullptr) { continue; }
      measurement = lidar_measurements_[lidar_iter][meas_iter];
      lookup_time_ = measurement->time_stamp;
      this->LoadLookupTree();

      // load scan and transform to viconbase frame
      PointCloud::Ptr scan = GetLidarScanFromBag(
          params_->lidar_params[measurement->lidar_id]->topic);
      PointCloud::Ptr scan_trans_est = boost::make_shared<PointCloud>();
      PointCloud::Ptr scan_trans_opt = boost::make_shared<PointCloud>();
      pcl::transformPointCloud(*scan, *scan_trans_est, TA_VICONBASE_SENSOR_est);
      pcl::transformPointCloud(*scan, *scan_trans_opt, TA_VICONBASE_SENSOR_opt);

      // load targets and transform to viconbase frame
      std::vector<Eigen::Affine3d, AlignAff3d> T_VICONBASE_TGTS;
      try {
        T_VICONBASE_TGTS = utils::GetTargetLocation(
            params_->target_params, params_->vicon_baselink_frame, lookup_time_,
            lookup_tree_);
      } catch (const std::runtime_error err) {
        LOG_ERROR("%s", err.what());
        continue;
      }

      PointCloud::Ptr targets_combined = boost::make_shared<PointCloud>();
      for (uint8_t n = 0; n < T_VICONBASE_TGTS.size(); n++) {
        const PointCloud::Ptr target =
            params_->target_params[n]->template_cloud;
        PointCloud::Ptr target_transformed = boost::make_shared<PointCloud>();
        pcl::transformPointCloud(*target, *target_transformed,
                                 T_VICONBASE_TGTS[n]);
        *targets_combined = *targets_combined + *target_transformed;
      }
      this->SaveScans(scan_trans_est, scan_trans_opt, targets_combined,
                      current_save_path, counter);
      if (counter == max_lidar_results_) { continue; }
    } // measurement iter
  }   // lidar iter
}

PointCloud::Ptr
    CalibrationVerification::GetLidarScanFromBag(const std::string& topic) {
  ros::Duration time_window_half = ros::Duration(0.5);
  rosbag::View view(bag_, rosbag::TopicQuery(topic),
                    lookup_time_ - time_window_half,
                    lookup_time_ + time_window_half, true);
  boost::shared_ptr<sensor_msgs::PointCloud2> lidar_msg;
  pcl::PCLPointCloud2::Ptr cloud_pc2 =
      boost::make_shared<pcl::PCLPointCloud2>();
  PointCloud::Ptr scan = boost::make_shared<PointCloud>();
  for (auto iter = view.begin(); iter != view.end(); iter++) {
    lidar_msg = iter->instantiate<sensor_msgs::PointCloud2>();
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
  PointCloud::Ptr scan_est_cropped = boost::make_shared<PointCloud>();
  PointCloud::Ptr scan_opt_cropped = boost::make_shared<PointCloud>();
  CropBox cropper;
  Eigen::Vector3f min{-10, -10, -10}, max{10, 10, 10};
  cropper.SetMinVector(min);
  cropper.SetMaxVector(max);
  cropper.Filter(*scan_est, *scan_est_cropped);
  cropper.Filter(*scan_opt, *scan_opt_cropped);

  PointCloudColor::Ptr cloud_combined = boost::make_shared<PointCloudColor>();
  PointCloudColor::Ptr scan_est_colored =
      utils::ColorPointCloud(scan_est_cropped, 255, 0, 0);
  PointCloudColor::Ptr scan_opt_colored =
      utils::ColorPointCloud(scan_opt_cropped, 0, 255, 0);
  PointCloudColor::Ptr targets_colored =
      utils::ColorPointCloud(targets, 0, 0, 255);
  *cloud_combined = *scan_est_colored;
  *cloud_combined = *cloud_combined + *scan_opt_colored;
  *cloud_combined = *cloud_combined + *targets_colored;
  pcl::io::savePCDFileBinary(save_path_full, *cloud_combined);
}

void CalibrationVerification::GetLidarErrors() {
  // iterate over each lidar
  for (uint8_t lidar_iter = 0; lidar_iter < params_->lidar_params.size();
       lidar_iter++) {
    // get initial calibration and optimized calibration
    Eigen::Affine3d TA_VICONBASE_SENSOR_est, TA_VICONBASE_SENSOR_true,
        TA_VICONBASE_SENSOR_opt;
    if (ground_truth_calib_set_) {
      for (CalibrationResult calib : calibrations_ground_truth_) {
        if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
          TA_VICONBASE_SENSOR_true.matrix() = calib.transform;
          break;
        }
      }
    }
    for (CalibrationResult calib : calibrations_initial_) {
      if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
        TA_VICONBASE_SENSOR_est.matrix() = calib.transform;
        break;
      }
    }
    for (CalibrationResult calib : calibrations_result_) {
      if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
        TA_VICONBASE_SENSOR_opt.matrix() = calib.transform;
        break;
      }
    }

    // iterate through all measurements for this lidar
    std::shared_ptr<LidarMeasurement> measurement;
    PointCloud::Ptr measured_keypoints, estimated_keypoints_target;
    PointCloud::Ptr estimated_keypoints_est = boost::make_shared<PointCloud>();
    PointCloud::Ptr estimated_keypoints_opt = boost::make_shared<PointCloud>();
    PointCloud::Ptr estimated_keypoints_true = boost::make_shared<PointCloud>();
    Eigen::Matrix4d T_SENSOR_TARGET_opt, T_SENSOR_TARGET_est,
        T_SENSOR_TARGET_true;
    std::vector<Eigen::Vector3d, AlignVec3d> lidar_errors_opt,
        lidar_errors_init, lidar_errors_true;
    for (int meas_iter = 0; meas_iter < lidar_measurements_[lidar_iter].size();
         meas_iter++) {
      if (lidar_measurements_[lidar_iter][meas_iter] == nullptr) { continue; }
      measurement = lidar_measurements_[lidar_iter][meas_iter];

      T_SENSOR_TARGET_opt = TA_VICONBASE_SENSOR_opt.inverse().matrix() *
                            measurement->T_VICONBASE_TARGET;
      T_SENSOR_TARGET_est = TA_VICONBASE_SENSOR_est.inverse().matrix() *
                            measurement->T_VICONBASE_TARGET;

      // get estimated keypoints given calibrations
      if (params_->target_params[measurement->target_id]
              ->keypoints_lidar.size() > 0) {
        estimated_keypoints_target = boost::make_shared<PointCloud>();
        for (Eigen::Vector3d keypoint :
             params_->target_params[measurement->target_id]->keypoints_lidar) {
          estimated_keypoints_target->push_back(
              utils::EigenPointToPCL(keypoint));
        }
      } else {
        estimated_keypoints_target =
            params_->target_params[measurement->target_id]->template_cloud;
      }
      pcl::transformPointCloud(*estimated_keypoints_target,
                               *estimated_keypoints_est, T_SENSOR_TARGET_est);
      pcl::transformPointCloud(*estimated_keypoints_target,
                               *estimated_keypoints_opt, T_SENSOR_TARGET_opt);

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
        T_SENSOR_TARGET_true = TA_VICONBASE_SENSOR_true.inverse().matrix() *
                               measurement->T_VICONBASE_TARGET;
        pcl::transformPointCloud(*estimated_keypoints_target,
                                 *estimated_keypoints_true,
                                 T_SENSOR_TARGET_true);
        lidar_errors_true = CalculateLidarErrors(measurement->keypoints,
                                                 estimated_keypoints_true);

        lidar_errors_true_.insert(lidar_errors_true_.end(),
                                  lidar_errors_true.begin(),
                                  lidar_errors_true.end());
      }

    } // measurement iter
  }   // lidar iter
}

std::vector<Eigen::Vector3d, AlignVec3d>
    CalibrationVerification::CalculateLidarErrors(
        const PointCloud::Ptr& measured_keypoints,
        const PointCloud::Ptr& estimated_keypoints) {
  // get correspondences
  corr_est_.setInputSource(measured_keypoints);
  corr_est_.setInputTarget(estimated_keypoints);
  corr_est_.determineCorrespondences(*correspondences_, max_point_cor_dist_);

  // get distances between correspondences
  int measurement_index, estimated_index;
  Eigen::Vector3d error;
  std::vector<Eigen::Vector3d, AlignVec3d> lidar_errors;
  for (int i = 0; i < correspondences_->size(); i++) {
    measurement_index = correspondences_->at(i).index_query;
    estimated_index = correspondences_->at(i).index_match;
    error = utils::PCLPointToEigen(measured_keypoints->at(measurement_index)) -
            utils::PCLPointToEigen(estimated_keypoints->at(estimated_index));
    error[0] = std::abs(error[0]);
    error[1] = std::abs(error[1]);
    error[2] = std::abs(error[2]);
    lidar_errors.push_back(error);
  }
  return lidar_errors;
}

void CalibrationVerification::SaveCameraVisuals() {
  std::vector<Eigen::Affine3d, AlignAff3d> T_VICONBASE_TGTS;

  // Iterate over each camera
  for (uint8_t cam_iter = 0; cam_iter < params_->camera_params.size();
       cam_iter++) {
    std::string current_save_path = results_directory_ + "CAMERAS/" +
                                    params_->camera_params[cam_iter]->frame +
                                    "/";
    boost::filesystem::create_directory(current_save_path);
    int counter = 0;
    std::string topic = params_->camera_params[cam_iter]->topic;
    std::string sensor_frame = params_->camera_params[cam_iter]->frame;
    std::vector<Eigen::Affine3d, AlignAff3d> T_cam_tgts_estimated_prev;
    rosbag::View view(bag_, rosbag::TopicQuery(topic), ros::TIME_MIN,
                      ros::TIME_MAX, true);

    // get initial calibration and optimized calibration
    Eigen::Affine3d TA_VICONBASE_SENSOR_est, TA_VICONBASE_SENSOR_opt,
        TA_VICONBASE_SENSOR_true;
    if (ground_truth_calib_set_) {
      for (CalibrationResult calib : calibrations_ground_truth_) {
        if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
          TA_VICONBASE_SENSOR_true.matrix() = calib.transform;
          break;
        }
      }
    }
    for (CalibrationResult calib : calibrations_initial_) {
      if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
        TA_VICONBASE_SENSOR_est.matrix() = calib.transform;
        break;
      }
    }
    for (CalibrationResult calib : calibrations_result_) {
      if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
        TA_VICONBASE_SENSOR_opt.matrix() = calib.transform;
        break;
      }
    }

    // iterate through all measurements for this lidar
    std::shared_ptr<cv::Mat> current_image, final_image;
    std::shared_ptr<CameraMeasurement> measurement;
    for (int meas_iter = 0; meas_iter < camera_measurements_[cam_iter].size();
         meas_iter++) {
      if (camera_measurements_[cam_iter][meas_iter] == nullptr) { continue; }

      measurement = camera_measurements_[cam_iter][meas_iter];
      lookup_time_ = measurement->time_stamp;
      this->LoadLookupTree();

      // load image from bag
      current_image = GetImageFromBag(
          params_->camera_params[measurement->camera_id]->topic);

      // load targets and transform to viconbase frame
      try {
        T_VICONBASE_TGTS = utils::GetTargetLocation(
            params_->target_params, params_->vicon_baselink_frame, lookup_time_,
            lookup_tree_);
      } catch (const std::runtime_error err) {
        LOG_ERROR("%s", err.what());
        continue;
      }

      // Add measurements to image
      final_image = this->ProjectTargetToImage(current_image, T_VICONBASE_TGTS,
                                               TA_VICONBASE_SENSOR_est.matrix(),
                                               cam_iter, cv::Scalar(0, 0, 255));

      if (num_tgts_in_img_ == 0) { continue; }

      counter++;
      final_image = this->ProjectTargetToImage(final_image, T_VICONBASE_TGTS,
                                               TA_VICONBASE_SENSOR_opt.matrix(),
                                               cam_iter, cv::Scalar(255, 0, 0));

      if (ground_truth_calib_set_) {
        final_image = this->ProjectTargetToImage(
            final_image, T_VICONBASE_TGTS, TA_VICONBASE_SENSOR_true.matrix(),
            cam_iter, cv::Scalar(0, 255, 0));
      }

      // save image with targets
      std::string save_path =
          current_save_path + "image_" + std::to_string(counter) + ".jpg";
      cv::imwrite(save_path, *final_image);

      if (counter == max_image_results_) { break; }
    } // measurement iter
  }   // camera iter
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
      *image =
          cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8)
              ->image;
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
    Eigen::Affine3d TA_VICONBASE_SENSOR_est, TA_VICONBASE_SENSOR_true,
        TA_VICONBASE_SENSOR_opt;
    if (ground_truth_calib_set_) {
      for (CalibrationResult calib : calibrations_ground_truth_) {
        if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
          TA_VICONBASE_SENSOR_true.matrix() = calib.transform;
          break;
        }
      }
    }
    for (CalibrationResult calib : calibrations_initial_) {
      if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
        TA_VICONBASE_SENSOR_est.matrix() = calib.transform;
        break;
      }
    }
    for (CalibrationResult calib : calibrations_result_) {
      if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
        TA_VICONBASE_SENSOR_opt.matrix() = calib.transform;
        break;
      }
    }

    // iterate through all measurements for this camera
    std::shared_ptr<CameraMeasurement> measurement;
    PointCloud::Ptr measured_keypoints;
    Eigen::Matrix4d T_SENSOR_TARGET_opt, T_SENSOR_TARGET_est,
        T_SENSOR_TARGET_true;
    std::vector<Eigen::Vector2d, AlignVec2d> camera_errors_opt,
        camera_errors_init, camera_errors_true;
    for (int meas_iter = 0; meas_iter < camera_measurements_[cam_iter].size();
         meas_iter++) {
      if (camera_measurements_[cam_iter][meas_iter] == nullptr) { continue; }
      measurement = camera_measurements_[cam_iter][meas_iter];

      // convert 2d measured keypoints to 3d
      PointCloud::Ptr measured_keypoints_3d = boost::make_shared<PointCloud>();
      pcl::PointCloud<pcl::PointXY>::Ptr measured_keypoints_2d =
          measurement->keypoints;
      pcl::PointXYZ point3d{0, 0, 0};
      for (int i = 0; i < measured_keypoints_2d->size(); i++) {
        point3d.x = measured_keypoints_2d->at(i).x;
        point3d.y = measured_keypoints_2d->at(i).y;
        measured_keypoints_3d->push_back(point3d);
      }

      T_SENSOR_TARGET_opt = TA_VICONBASE_SENSOR_opt.inverse().matrix() *
                            measurement->T_VICONBASE_TARGET;
      T_SENSOR_TARGET_est = TA_VICONBASE_SENSOR_est.inverse().matrix() *
                            measurement->T_VICONBASE_TARGET;
      camera_errors_opt =
          CalculateCameraErrors(measured_keypoints_3d, T_SENSOR_TARGET_opt,
                                measurement->target_id, measurement->camera_id);
      camera_errors_init =
          CalculateCameraErrors(measured_keypoints_3d, T_SENSOR_TARGET_est,
                                measurement->target_id, measurement->camera_id);

      camera_errors_opt_.insert(camera_errors_opt_.end(),
                                camera_errors_opt.begin(),
                                camera_errors_opt.end());

      camera_errors_init_.insert(camera_errors_init_.end(),
                                 camera_errors_init.begin(),
                                 camera_errors_init.end());

      if (ground_truth_calib_set_) {
        T_SENSOR_TARGET_true = TA_VICONBASE_SENSOR_true.inverse().matrix() *
                               measurement->T_VICONBASE_TARGET;
        camera_errors_true = CalculateCameraErrors(
            measured_keypoints_3d, T_SENSOR_TARGET_true, measurement->target_id,
            measurement->camera_id);

        camera_errors_true_.insert(camera_errors_true_.end(),
                                   camera_errors_true.begin(),
                                   camera_errors_true.end());
      }
    } // measurement iter
  }   // camera iter
}

std::vector<Eigen::Vector2d, AlignVec2d>
    CalibrationVerification::CalculateCameraErrors(
        const PointCloud::Ptr& measured_keypoints,
        const Eigen::Matrix4d& T_SENSOR_TARGET, const int& target_id,
        const int& camera_id) {
  // get estimated (optimization or initial) keypoint locations
  PointCloud::Ptr keypoints_target_frame;
  if (params_->target_params[target_id]->keypoints_camera.size() > 0) {
    keypoints_target_frame = boost::make_shared<PointCloud>();
    for (Eigen::Vector3d keypoint :
         params_->target_params[target_id]->keypoints_camera) {
      keypoints_target_frame->push_back(utils::EigenPointToPCL(keypoint));
    }
  } else {
    keypoints_target_frame = params_->target_params[target_id]->template_cloud;
  }

  // project points to image plane and save as cloud
  PointCloud::Ptr keypoints_projected = utils::ProjectPoints(
      keypoints_target_frame, params_->camera_params[camera_id]->camera_model,
      T_SENSOR_TARGET);

  // get correspondences
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
      corr_est;
  boost::shared_ptr<pcl::Correspondences> correspondences =
      boost::make_shared<pcl::Correspondences>();
  corr_est.setInputSource(measured_keypoints);
  corr_est.setInputTarget(keypoints_projected);
  corr_est.determineCorrespondences(*correspondences, max_pixel_cor_dist_);

  // get distances between correspondences
  int measurement_index, estimated_index;
  Eigen::Vector3d error3d;
  Eigen::Vector2d error2d;
  std::vector<Eigen::Vector2d, AlignVec2d> camera_errors;
  for (int i = 0; i < correspondences->size(); i++) {
    measurement_index = correspondences->at(i).index_query;
    estimated_index = correspondences->at(i).index_match;
    error3d =
        utils::PCLPointToEigen(measured_keypoints->at(measurement_index)) -
        utils::PCLPointToEigen(keypoints_projected->at(estimated_index));
    error2d[0] = std::abs(error3d[0]);
    error2d[1] = std::abs(error3d[1]);
    camera_errors.push_back(error2d);
  }
  return camera_errors;
}

std::shared_ptr<cv::Mat> CalibrationVerification::ProjectTargetToImage(
    const std::shared_ptr<cv::Mat>& img_in,
    const std::vector<Eigen::Affine3d, AlignAff3d>& T_VICONBASE_TGTS,
    const Eigen::Matrix4d& T_VICONBASE_SENSOR, const int& cam_iter,
    cv::Scalar colour) {
  // create all objects we'll need
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  Eigen::Vector4d point_transformed;
  Eigen::Vector4d point_target;
  pcl::PointXYZ point_pcl_projected;
  std::shared_ptr<cv::Mat> img_out = std::make_shared<cv::Mat>();
  *img_out = img_in->clone();

  // iterate through all targets
  Eigen::Affine3d T_VICONBASE_TARGET;
  num_tgts_in_img_ = 0;
  for (int target_iter = 0; target_iter < T_VICONBASE_TGTS.size();
       target_iter++) {
    T_VICONBASE_TARGET = T_VICONBASE_TGTS[target_iter];
    // check if target origin is in camera frame
    point_target = Eigen::Vector4d(0, 0, 0, 1);
    point_transformed = utils::InvertTransform(T_VICONBASE_SENSOR) *
                        T_VICONBASE_TARGET * point_target;
    opt<Eigen::Vector2d> origin_projected =
        params_->camera_params[cam_iter]->camera_model->ProjectPointPrecise(
            point_transformed.hnormalized());

    if (!origin_projected.has_value()) { continue; }

    num_tgts_in_img_++;

    // add keypoints to image
    cv::Point point_cv_projected;
    for (Eigen::Vector3d point :
         params_->target_params[target_iter]->keypoints_camera) {
      point_target = point.homogeneous();
      point_transformed = utils::InvertTransform(T_VICONBASE_SENSOR) *
                          T_VICONBASE_TARGET * point_target;
      opt<Eigen::Vector2d> point_projected =
          params_->camera_params[cam_iter]->camera_model->ProjectPointPrecise(
              point_transformed.hnormalized());

      if (point_projected.has_value()) {
        cv::Point pixel_cv;
        pixel_cv.x = point_projected.value()[0];
        pixel_cv.y = point_projected.value()[1];
        cv::circle(*img_out, pixel_cv, keypoint_circle_diameter_, colour);
      }
    }

    if (!show_target_outline_on_image_) { return img_out; }
    // iterate through all target points
    pcl::PointCloud<pcl::PointXYZ>::Ptr target =
        params_->target_params[target_iter]->template_cloud;
    for (uint32_t i = 0; i < target->size(); i++) {
      point_target = utils::PCLPointToEigen(target->at(i)).homogeneous();
      point_transformed = utils::InvertTransform(T_VICONBASE_SENSOR) *
                          T_VICONBASE_TARGET * point_target;
      opt<Eigen::Vector2d> point_projected =
          params_->camera_params[cam_iter]->camera_model->ProjectPointPrecise(
              point_transformed.hnormalized());

      if (point_projected.has_value()) {
        point_pcl_projected.x = point_projected.value()[0];
        point_pcl_projected.y = point_projected.value()[1];
        point_pcl_projected.z = 0;
        cloud_projected->push_back(point_pcl_projected);
      }
    }

    // Get concave hull alpha
    // we want to make sure alpha is larget than two consecutive projected pts
    // assume template points are 5mm apart, calculate distance in pixels
    // between two consecutive projected points
    Eigen::Matrix4d T_SENSOR_TARGET =
        utils::InvertTransform(T_VICONBASE_SENSOR) *
        T_VICONBASE_TARGET.matrix();
    Eigen::Vector4d point1 = T_SENSOR_TARGET * Eigen::Vector4d(0, 0, 0, 1);
    Eigen::Vector4d point2 = T_SENSOR_TARGET * Eigen::Vector4d(0, 0, 0.005, 1);
    opt<Eigen::Vector2d> point1_projected =
        params_->camera_params[cam_iter]->camera_model->ProjectPointPrecise(
            point1.hnormalized());
    opt<Eigen::Vector2d> point2_projected =
        params_->camera_params[cam_iter]->camera_model->ProjectPointPrecise(
            point2.hnormalized());

    double distance = 3;
    if (point1_projected.has_value() && point2_projected.has_value()) {
      distance = (point1_projected.value() - point2_projected.value()).norm();
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
