#include "vicon_calibration/CalibrationVerification.h"
#include "beam_filtering/CropBox.h"
#include "vicon_calibration/measurement_extractors/CylinderCameraExtractor.h"
#include "vicon_calibration/measurement_extractors/CylinderLidarExtractor.h"
#include "vicon_calibration/measurement_extractors/DiamondCameraExtractor.h"
#include "vicon_calibration/measurement_extractors/DiamondLidarExtractor.h"
#include <fstream>
#include <iostream>

// ROS headers
#include <cv_bridge/cv_bridge.h>
#include <nlohmann/json.hpp>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2_msgs/TFMessage.h>

// PCL specific headers
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/concave_hull.h>
#include <pcl_conversions/pcl_conversions.h>

namespace vicon_calibration {

void CalibrationVerification::LoadJSON(const std::string &file_name) {
  LOG_INFO("Loading CalibrationVerification Config File: %s",
           file_name.c_str());
  nlohmann::json J;
  std::ifstream file(file_name);
  file >> J;

  output_directory_ = J["output_directory"];
  double t = J["time_increment"];
  time_increment_ = ros::Duration(t);
  max_image_results_ = J["max_image_results"];
  max_lidar_results_ = J["max_lidar_results"];
  max_pixel_cor_dist_ = J["max_pixel_cor_dist"];
  max_point_cor_dist_ = J["max_point_cor_dist"];
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
  if (!config_path_set_) {
    throw std::invalid_argument{"Calibrator Config File Path Not Set."};
  }
  if (!lidar_measurements_set_) {
    throw std::invalid_argument{"Lidar Measurements Not Set."};
  }
  if (!camera_measurements_set_) {
    throw std::invalid_argument{"Camera Measurements Not Set."};
  }
}

void CalibrationVerification::ProcessResults() {
  LOG_INFO("Processing calibration results.");
  this->CheckInputs();

  this->LoadJSON(utils::GetFilePathConfig("CalibrationVerification.json"));

  // load bag file
  try {
    bag_.open(params_->bag_file, rosbag::bagmode::Read);
  } catch (rosbag::BagException &ex) {
    LOG_ERROR("Bag exception : %s", ex.what());
  }

  this->CreateDirectories();
  this->PrintConfig();
  this->PrintCalibrations(calibrations_initial_, "initial_calibrations.txt");
  this->PrintCalibrations(calibrations_result_, "optimized_calibrations.txt");
  if (params_->using_simulation && perturbed_calib_set_) {
    this->PrintCalibrations(calibrations_perturbed_,
                            "perturbed_calibrations.txt");
  }
  this->PrintCalibrationErrors();
  this->SaveLidarVisuals();
  this->GetLidarErrors();
  this->SaveCameraVisuals();
  this->GetCameraErrors();
  this->PrintErrorsSummary();
}

void CalibrationVerification::SetConfig(const std::string &calib_config) {
  calibration_config_ = calib_config;
  config_path_set_ = true;
}

void CalibrationVerification::SetInitialCalib(
    const std::vector<vicon_calibration::CalibrationResult> &calib) {
  calibrations_initial_ = calib;
  initial_calib_set_ = true;
}

// TODO: add checks for whether or not this was set. This
// should still work otherwise
void CalibrationVerification::SetPeturbedCalib(
    const std::vector<vicon_calibration::CalibrationResult> &calib) {
  calibrations_perturbed_ = calib;
  perturbed_calib_set_ = true;
}

void CalibrationVerification::SetOptimizedCalib(
    const std::vector<vicon_calibration::CalibrationResult> &calib) {
  calibrations_result_ = calib;
  optimized_calib_set_ = true;
}

void CalibrationVerification::SetParams(
    std::shared_ptr<CalibratorConfig> &params) {
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
    const std::vector<std::vector<std::shared_ptr<LidarMeasurement>>>
        &lidar_measurements) {
  lidar_measurements_ = lidar_measurements;
  lidar_measurements_set_ = true;
}

void CalibrationVerification::SetCameraMeasurements(
    const std::vector<std::vector<std::shared_ptr<CameraMeasurement>>>
        &camera_measurements) {
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
    std::vector<vicon_calibration::CalibrationResult> &calib,
    const std::string &file_name) {
  std::string output_path = results_directory_ + file_name;
  std::ofstream file(output_path);
  for (uint16_t i = 0; i < calib.size(); i++) {
    Eigen::Matrix4d T = calib[i].transform;
    Eigen::Matrix3d R = T.block(0, 0, 3, 3);
    Eigen::Vector3d rpy = R.eulerAngles(0, 1, 2);
    file << "T_" << calib[i].to_frame << "_" << calib[i].from_frame << ":\n"
         << T << "\n"
         << "rpy (deg): [" << utils::WrapToTwoPi(rpy[0]) * RAD_TO_DEG << ", "
         << utils::WrapToTwoPi(rpy[1]) * RAD_TO_DEG << ", "
         << utils::WrapToTwoPi(rpy[2]) * RAD_TO_DEG << "]\n";
  }
}

void CalibrationVerification::PrintCalibrationErrors() {
  std::string output_path = results_directory_ + "calibration_errors.txt";
  std::ofstream file(output_path);
  // first print errors between initial calibration and final
  file << "Showing errors between:\n"
       << "initial calibration estimates and optimized calibrations:\n\n";
  for (uint16_t i = 0; i < calibrations_result_.size(); i++) {
    Eigen::Matrix4d T_final = calibrations_result_[i].transform;
    Eigen::Matrix4d T_init = calibrations_initial_[i].transform;
    Eigen::Matrix3d R_final = T_final.block(0, 0, 3, 3);
    Eigen::Matrix3d R_init = T_init.block(0, 0, 3, 3);
    Eigen::Vector3d rpy_final = R_final.eulerAngles(0, 1, 2);
    Eigen::Vector3d rpy_init = R_init.eulerAngles(0, 1, 2);
    Eigen::Vector3d rpy_error = rpy_final - rpy_init;
    rpy_error[0] = utils::GetAngleErrorPi(rpy_error[0]);
    rpy_error[1] = utils::GetAngleErrorPi(rpy_error[1]);
    rpy_error[2] = utils::GetAngleErrorPi(rpy_error[2]);
    Eigen::Vector3d t_final = T_final.block(0, 3, 3, 1);
    Eigen::Vector3d t_init = T_init.block(0, 3, 3, 1);
    Eigen::Vector3d t_error = t_final - t_init;
    t_error[0] = std::abs(t_error[0]);
    t_error[1] = std::abs(t_error[1]);
    t_error[2] = std::abs(t_error[2]);
    file << "T_" << calibrations_result_[i].to_frame << "_"
         << calibrations_result_[i].from_frame << ":\n"
         << "rpy error (deg): [" << rpy_error[0] * RAD_TO_DEG << ", "
         << rpy_error[1] * RAD_TO_DEG << ", " << rpy_error[2] * RAD_TO_DEG
         << "]\n"
         << "translation error (mm): [" << t_error[0] * 1000 << ", "
         << t_error[1] * 1000 << ", " << t_error[2] * 1000 << "]\n\n";
  }

  if (!params_->using_simulation) {
    return;
  }

  // next print errors between initial calirations and perturbed calibration
  file << "---------------------------------------------------------\n\n"
       << "Showing errors between:\n"
       << "initial calibrations and perturbed calibrations:\n\n";
  for (uint16_t i = 0; i < calibrations_result_.size(); i++) {
    Eigen::Matrix4d T_final = calibrations_perturbed_[i].transform;
    Eigen::Matrix4d T_init = calibrations_initial_[i].transform;
    Eigen::Matrix3d R_final = T_final.block(0, 0, 3, 3);
    Eigen::Matrix3d R_init = T_init.block(0, 0, 3, 3);
    Eigen::Vector3d rpy_final = R_final.eulerAngles(0, 1, 2);
    Eigen::Vector3d rpy_init = R_init.eulerAngles(0, 1, 2);
    Eigen::Vector3d rpy_error = rpy_final - rpy_init;
    rpy_error[0] = utils::GetAngleErrorPi(rpy_error[0]);
    rpy_error[1] = utils::GetAngleErrorPi(rpy_error[1]);
    rpy_error[2] = utils::GetAngleErrorPi(rpy_error[2]);
    Eigen::Vector3d t_final = T_final.block(0, 3, 3, 1);
    Eigen::Vector3d t_init = T_init.block(0, 3, 3, 1);
    Eigen::Vector3d t_error = t_final - t_init;
    t_error[0] = std::abs(t_error[0]);
    t_error[1] = std::abs(t_error[1]);
    t_error[2] = std::abs(t_error[2]);
    file << "T_" << calibrations_result_[i].to_frame << "_"
         << calibrations_result_[i].from_frame << ":\n"
         << "rpy error (deg): [" << rpy_error[0] * RAD_TO_DEG << ", "
         << rpy_error[1] * RAD_TO_DEG << ", " << rpy_error[2] * RAD_TO_DEG
         << "]\n"
         << "translation error (mm): [" << t_error[0] * 1000 << ", "
         << t_error[1] * 1000 << ", " << t_error[2] * 1000 << "]\n\n";
  }

  // next print errors between perturbed calibration and final
  file << "---------------------------------------------------------\n\n"
       << "Showing errors between:\n"
       << "initial perturbed calibrations and optimized calibrations:\n\n";
  for (uint16_t i = 0; i < calibrations_result_.size(); i++) {
    Eigen::Matrix4d T_final = calibrations_result_[i].transform;
    Eigen::Matrix4d T_init = calibrations_perturbed_[i].transform;
    Eigen::Matrix3d R_final = T_final.block(0, 0, 3, 3);
    Eigen::Matrix3d R_init = T_init.block(0, 0, 3, 3);
    Eigen::Vector3d rpy_final = R_final.eulerAngles(0, 1, 2);
    Eigen::Vector3d rpy_init = R_init.eulerAngles(0, 1, 2);
    Eigen::Vector3d rpy_error = rpy_final - rpy_init;
    rpy_error[0] = utils::GetAngleErrorPi(rpy_error[0]);
    rpy_error[1] = utils::GetAngleErrorPi(rpy_error[1]);
    rpy_error[2] = utils::GetAngleErrorPi(rpy_error[2]);
    Eigen::Vector3d t_final = T_final.block(0, 3, 3, 1);
    Eigen::Vector3d t_init = T_init.block(0, 3, 3, 1);
    Eigen::Vector3d t_error = t_final - t_init;
    t_error[0] = std::abs(t_error[0]);
    t_error[1] = std::abs(t_error[1]);
    t_error[2] = std::abs(t_error[2]);
    file << "T_" << calibrations_result_[i].to_frame << "_"
         << calibrations_result_[i].from_frame << ":\n"
         << "rpy error (deg): [" << rpy_error[0] * RAD_TO_DEG << ", "
         << rpy_error[1] * RAD_TO_DEG << ", " << rpy_error[2] * RAD_TO_DEG
         << "]\n"
         << "translation error (mm): [" << t_error[0] * 1000 << ", "
         << t_error[1] * 1000 << ", " << t_error[2] * 1000 << "]\n\n";
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
  std::vector<Eigen::Affine3d, AlignAff3d> T_VICONBASE_TGTS;

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
    ros::Time time_last(0, 0);

    // get initial calibration and optimized calibration
    Eigen::Affine3d TA_VICONBASE_SENSOR_est, TA_VICONBASE_SENSOR_opt;
    if (params_->using_simulation && perturbed_calib_set_) {
      for (CalibrationResult calib : calibrations_perturbed_) {
        if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
          TA_VICONBASE_SENSOR_est.matrix() = calib.transform;
          break;
        }
      }
    } else {
      for (CalibrationResult calib : calibrations_initial_) {
        if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
          TA_VICONBASE_SENSOR_est.matrix() = calib.transform;
          break;
        }
      }
    }

    for (CalibrationResult calib : calibrations_result_) {
      if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
        TA_VICONBASE_SENSOR_opt.matrix() = calib.transform;
        break;
      }
    }

    // Initialize all the clouds we will need
    PointCloud::Ptr scan = boost::make_shared<PointCloud>();
    PointCloud::Ptr scan_trans_est = boost::make_shared<PointCloud>();
    PointCloud::Ptr scan_trans_opt = boost::make_shared<PointCloud>();
    PointCloud::Ptr target = boost::make_shared<PointCloud>();
    PointCloud::Ptr target_transformed = boost::make_shared<PointCloud>();
    PointCloud::Ptr targets_combined = boost::make_shared<PointCloud>();

    // iterate through all measurements for this lidar
    std::shared_ptr<LidarMeasurement> measurement;
    ros::Time time_current;
    for (int meas_iter = 0; meas_iter < lidar_measurements_[lidar_iter].size();
         meas_iter++) {
      if (lidar_measurements_[lidar_iter][meas_iter] == nullptr) {
        continue;
      }
      measurement = lidar_measurements_[lidar_iter][meas_iter];
      time_current = measurement->time_stamp;
      if (time_current < time_last + time_increment_) {
        continue;
      }
      lookup_time_ = time_current;
      this->LoadLookupTree();
      time_last = time_current;

      // load scan and transform to viconbase frame
      scan = GetLidarScanFromBag(
          params_->lidar_params[measurement->lidar_id]->topic);
      pcl::transformPointCloud(*scan, *scan_trans_est, TA_VICONBASE_SENSOR_est);
      pcl::transformPointCloud(*scan, *scan_trans_opt, TA_VICONBASE_SENSOR_opt);

      // load targets and transform to viconbase frame
      try {
        T_VICONBASE_TGTS = utils::GetTargetLocation(
            params_->target_params, params_->vicon_baselink_frame, lookup_time_,
            lookup_tree_);
      } catch (const std::runtime_error err) {
        LOG_ERROR("%s", err.what());
        continue;
      }

      for (uint8_t n = 0; n < T_VICONBASE_TGTS.size(); n++) {
        target = params_->target_params[n]->template_cloud;
        pcl::transformPointCloud(*target, *target_transformed,
                                 T_VICONBASE_TGTS[n]);
        *targets_combined = *targets_combined + *target_transformed;
      }
      this->SaveScans(scan_trans_est, scan_trans_opt, targets_combined,
                      current_save_path, counter);
      if (counter == max_lidar_results_) {
        return;
      }
    } // measurement iter
  }   // lidar iter
}

PointCloud::Ptr
CalibrationVerification::GetLidarScanFromBag(const std::string &topic) {
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

void CalibrationVerification::SaveScans(const PointCloud::Ptr &scan_est,
                                        const PointCloud::Ptr &scan_opt,
                                        const PointCloud::Ptr &targets,
                                        const std::string &save_path,
                                        const int &scan_count) {
  std::string save_path_full =
      save_path + "scan_" + std::to_string(scan_count + 1) + ".pcd";
  PointCloud::Ptr scan_est_cropped = boost::make_shared<PointCloud>();
  PointCloud::Ptr scan_opt_cropped = boost::make_shared<PointCloud>();
  beam_filtering::CropBox cropper;
  Eigen::Vector3f min{-6, -6, -6}, max{6, 6, 6};
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
    Eigen::Affine3d TA_VICONBASE_SENSOR_est, TA_VICONBASE_SENSOR_pert,
        TA_VICONBASE_SENSOR_opt;
    if (params_->using_simulation && perturbed_calib_set_) {
      for (CalibrationResult calib : calibrations_perturbed_) {
        if (calib.type == SensorType::LIDAR && calib.sensor_id == lidar_iter) {
          TA_VICONBASE_SENSOR_pert.matrix() = calib.transform;
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
    PointCloud::Ptr measured_keypoints;
    Eigen::Matrix4d T_SENSOR_TARGET_opt, T_SENSOR_TARGET_est,
        T_SENSOR_TARGET_pert;
    std::vector<Eigen::Vector3d, AlignVec3d> lidar_errors_opt,
        lidar_errors_init, lidar_errors_pert;
    for (int meas_iter = 0; meas_iter < lidar_measurements_[lidar_iter].size();
         meas_iter++) {
      if (lidar_measurements_[lidar_iter][meas_iter] == nullptr) {
        continue;
      }
      measurement = lidar_measurements_[lidar_iter][meas_iter];
      
      T_SENSOR_TARGET_opt = TA_VICONBASE_SENSOR_opt.inverse().matrix() *
                            measurement->T_VICONBASE_TARGET;
      T_SENSOR_TARGET_est = TA_VICONBASE_SENSOR_est.inverse().matrix() *
                            measurement->T_VICONBASE_TARGET;
      lidar_errors_opt = CalculateLidarErrors(
          measurement->keypoints, T_SENSOR_TARGET_opt, measurement->target_id);
      lidar_errors_init = CalculateLidarErrors(
          measurement->keypoints, T_SENSOR_TARGET_est, measurement->target_id);

      lidar_errors_opt_.insert(lidar_errors_opt_.end(),
                               lidar_errors_opt.begin(),
                               lidar_errors_opt.end());

      lidar_errors_init_.insert(lidar_errors_init_.end(),
                                lidar_errors_init.begin(),
                                lidar_errors_init.end());

      if (params_->using_simulation && perturbed_calib_set_) {
        T_SENSOR_TARGET_pert = TA_VICONBASE_SENSOR_pert.inverse().matrix() *
                               measurement->T_VICONBASE_TARGET;
        lidar_errors_pert =
            CalculateLidarErrors(measurement->keypoints, T_SENSOR_TARGET_pert,
                                 measurement->target_id);

        lidar_errors_pert_.insert(lidar_errors_pert_.end(),
                                  lidar_errors_pert.begin(),
                                  lidar_errors_pert.end());
      }

    } // measurement iter
  }   // lidar iter
}

std::vector<Eigen::Vector3d, AlignVec3d>
CalibrationVerification::CalculateLidarErrors(
    const PointCloud::Ptr &measured_keypoints,
    const Eigen::Matrix4d &T_SENSOR_TARGET, const int &target_id) {

  // get estimated keypoints given calibrations
  PointCloud::Ptr estimated_keypoints;
  if (params_->target_params[target_id]->keypoints_lidar.size() > 0) {
    estimated_keypoints = boost::make_shared<PointCloud>();
    for (Eigen::Vector3d keypoint :
         params_->target_params[target_id]->keypoints_lidar) {
      estimated_keypoints->push_back(utils::EigenPointToPCL(keypoint));
    }
  } else {
    estimated_keypoints = params_->target_params[target_id]->template_cloud;
  }
  pcl::transformPointCloud(*estimated_keypoints, *estimated_keypoints,
                           T_SENSOR_TARGET);

  // get correspondences
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
      corr_est;
  boost::shared_ptr<pcl::Correspondences> correspondences =
      boost::make_shared<pcl::Correspondences>();
  corr_est.setInputSource(measured_keypoints);
  corr_est.setInputTarget(estimated_keypoints);
  corr_est.determineCorrespondences(*correspondences, max_point_cor_dist_);

  // get distances between correspondences
  int measurement_index, estimated_index;
  Eigen::Vector3d error;
  std::vector<Eigen::Vector3d, AlignVec3d> lidar_errors;
  for (int i = 0; i < correspondences->size(); i++) {
    measurement_index = correspondences->at(i).index_query;
    estimated_index = correspondences->at(i).index_match;
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
    ros::Time time_last(0, 0);

    // get initial calibration and optimized calibration
    Eigen::Affine3d TA_VICONBASE_SENSOR_est, TA_VICONBASE_SENSOR_opt;
    if (params_->using_simulation && perturbed_calib_set_) {
      for (CalibrationResult calib : calibrations_perturbed_) {
        if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
          TA_VICONBASE_SENSOR_est.matrix() = calib.transform;
          break;
        }
      }
    } else {
      for (CalibrationResult calib : calibrations_initial_) {
        if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
          TA_VICONBASE_SENSOR_est.matrix() = calib.transform;
          break;
        }
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
    ros::Time time_current;
    for (int meas_iter = 0; meas_iter < camera_measurements_[cam_iter].size();
         meas_iter++) {

      if (camera_measurements_[cam_iter][meas_iter] == nullptr) {
        continue;
      }

      measurement = camera_measurements_[cam_iter][meas_iter];
      time_current = measurement->time_stamp;
      if (time_current < time_last + time_increment_) {
        continue;
      }
      lookup_time_ = time_current;
      this->LoadLookupTree();
      time_last = time_current;

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

      final_image = this->ProjectTargetToImage(current_image, T_VICONBASE_TGTS,
                                               TA_VICONBASE_SENSOR_est.matrix(),
                                               cam_iter, cv::Scalar(0, 0, 255));

      if (num_tgts_in_img_ == 0) {
        continue;
      }

      // save image with targets
      counter++;
      final_image = this->ProjectTargetToImage(final_image, T_VICONBASE_TGTS,
                                               TA_VICONBASE_SENSOR_opt.matrix(),
                                               cam_iter, cv::Scalar(255, 0, 0));
      std::string save_path =
          current_save_path + "image_" + std::to_string(counter) + ".jpg";
      cv::imwrite(save_path, *final_image);

      if (counter == max_image_results_) {
        break;
      }
    } // measurement iter
  }   // camera iter
}

std::shared_ptr<cv::Mat>
CalibrationVerification::GetImageFromBag(const std::string &topic) {
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
    Eigen::Affine3d TA_VICONBASE_SENSOR_est, TA_VICONBASE_SENSOR_pert,
        TA_VICONBASE_SENSOR_opt;
    if (params_->using_simulation && perturbed_calib_set_) {
      for (CalibrationResult calib : calibrations_perturbed_) {
        if (calib.type == SensorType::CAMERA && calib.sensor_id == cam_iter) {
          TA_VICONBASE_SENSOR_pert.matrix() = calib.transform;
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
        T_SENSOR_TARGET_pert;
    std::vector<Eigen::Vector2d, AlignVec2d> camera_errors_opt,
        camera_errors_init, camera_errors_pert;
    for (int meas_iter = 0; meas_iter < camera_measurements_[cam_iter].size();
         meas_iter++) {
      if (camera_measurements_[cam_iter][meas_iter] == nullptr) {
        continue;
      }
      measurement = camera_measurements_[cam_iter][meas_iter];

      // convert 2d measured keypoints to 3d
      PointCloud::Ptr measured_keypoints_3d = boost::make_shared<PointCloud>();
      pcl::PointCloud<pcl::PointXY>::Ptr measured_keypoints_2d = measurement->keypoints;
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

      if (params_->using_simulation && perturbed_calib_set_) {
        T_SENSOR_TARGET_pert = TA_VICONBASE_SENSOR_pert.inverse().matrix() *
                               measurement->T_VICONBASE_TARGET;
        camera_errors_pert = CalculateCameraErrors(
            measured_keypoints_3d, T_SENSOR_TARGET_pert, measurement->target_id,
            measurement->camera_id);

        camera_errors_pert_.insert(camera_errors_pert_.end(),
                                   camera_errors_pert.begin(),
                                   camera_errors_pert.end());
      }
    } // measurement iter
  }   // camera iter
}

std::vector<Eigen::Vector2d, AlignVec2d>
CalibrationVerification::CalculateCameraErrors(
    const PointCloud::Ptr &measured_keypoints,
    const Eigen::Matrix4d &T_SENSOR_TARGET, const int &target_id,
    const int &camera_id) {

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
      params_->camera_params[camera_id]->images_distorted, T_SENSOR_TARGET);

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
    const std::shared_ptr<cv::Mat> &img_in,
    const std::vector<Eigen::Affine3d, AlignAff3d> &T_VICONBASE_TGTS,
    const Eigen::Matrix4d &T_VICONBASE_SENSOR, const int &cam_iter,
    cv::Scalar colour) {
  // create all objects we'll need
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  Eigen::Vector2d point_projected;
  Eigen::Vector4d point_transformed, point_target;
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
    if (params_->camera_params[cam_iter]->images_distorted) {
      point_projected =
          params_->camera_params[cam_iter]->camera_model->ProjectPoint(
              utils::HomoPointToPoint(point_transformed));
    } else {
      point_projected = params_->camera_params[cam_iter]
                            ->camera_model->ProjectUndistortedPoint(
                                utils::HomoPointToPoint(point_transformed));
    }
    if (!params_->camera_params[cam_iter]->camera_model->PixelInImage(
            point_projected)) {
      continue;
    }

    num_tgts_in_img_++;

    // iterate through all target points
    pcl::PointCloud<pcl::PointXYZ>::Ptr target =
        params_->target_params[target_iter]->template_cloud;
    for (uint32_t i = 0; i < target->size(); i++) {
      point_target =
          utils::PointToHomoPoint(utils::PCLPointToEigen(target->at(i)));
      point_transformed = utils::InvertTransform(T_VICONBASE_SENSOR) *
                          T_VICONBASE_TARGET * point_target;
      if (params_->camera_params[cam_iter]->images_distorted) {
        point_projected =
            params_->camera_params[cam_iter]->camera_model->ProjectPoint(
                utils::HomoPointToPoint(point_transformed));
      } else {
        point_projected = params_->camera_params[cam_iter]
                              ->camera_model->ProjectUndistortedPoint(
                                  utils::HomoPointToPoint(point_transformed));
      }
      if (params_->camera_params[cam_iter]->camera_model->PixelInImage(
              point_projected)) {
        point_pcl_projected.x = point_projected[0];
        point_pcl_projected.y = point_projected[1];
        point_pcl_projected.z = 0;
        cloud_projected->push_back(point_pcl_projected);
      }
    }

    // keep only perimeter points
    pcl::ConcaveHull<pcl::PointXYZ> concave_hull;
    concave_hull.setInputCloud(cloud_projected);
    concave_hull.setAlpha(concave_hull_alpha_);
    concave_hull.reconstruct(*cloud_projected);

    // colour image
    int radius = 1;
    cv::Point point_cv_projected;
    for (uint32_t i = 0; i < cloud_projected->size(); i++) {
      point_pcl_projected = cloud_projected->at(i);
      point_cv_projected.x = point_pcl_projected.x;
      point_cv_projected.y = point_pcl_projected.y;
      cv::circle(*img_out, point_cv_projected, radius, colour);
    }
  }
  return img_out;
}

void CalibrationVerification::LoadLookupTree() {
  lookup_tree_->Clear();
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

  norms_summed = 0;
  for (int i = 0; i < lidar_errors_init_.size(); i++) {
    norms_summed += lidar_errors_init_[i].norm();
  }
  norms_averaged = norms_summed / lidar_errors_init_.size();

  file << "\n-----------------------------------------------------------\n\n"
       << "Outputting Error Statistics for Initial Lidar Calibrations:\n"
       << "Average Error Norm (m): " << norms_averaged << "\n"
       << "Samples Used: " << lidar_errors_init_.size() << "\n";

  if(params_->using_simulation && perturbed_calib_set_){
    norms_summed = 0;
    for (int i = 0; i < lidar_errors_pert_.size(); i++) {
      norms_summed += lidar_errors_pert_[i].norm();
    }
    norms_averaged = norms_summed / lidar_errors_pert_.size();

    file << "\n-----------------------------------------------------------\n\n"
         << "Outputting Error Statistics for Perturbed Lidar Calibrations:\n"
         << "Average Error Norm (m): " << norms_averaged << "\n"
         << "Samples Used: " << lidar_errors_pert_.size() << "\n";
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

  norms_summed = 0;
  for (int i = 0; i < camera_errors_init_.size(); i++) {
    norms_summed += camera_errors_init_[i].norm();
  }
  norms_averaged = norms_summed / camera_errors_init_.size();

  file << "\n-----------------------------------------------------------\n\n"
       << "Outputting Error Statistics for Initial Camera Calibrations:\n"
       << "Average Error Norm (pixels): " << norms_averaged << "\n"
       << "Samples Used: " << camera_errors_init_.size() << "\n";

  if(params_->using_simulation && perturbed_calib_set_){
    norms_summed = 0;
    for (int i = 0; i < camera_errors_pert_.size(); i++) {
      norms_summed += camera_errors_pert_[i].norm();
    }
    norms_averaged = norms_summed / camera_errors_pert_.size();

    file << "\n-----------------------------------------------------------\n\n"
         << "Outputting Error Statistics for Perturbed Camera Calibrations:\n"
         << "Average Error Norm (pixels): " << norms_averaged << "\n"
         << "Samples Used: " << camera_errors_pert_.size() << "\n";
  }
}

} // end namespace vicon_calibration
