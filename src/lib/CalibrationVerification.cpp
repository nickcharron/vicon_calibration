#include "vicon_calibration/CalibrationVerification.h"
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

void CalibrationVerification::ProcessResults() {
  LOG_INFO("Processing calibration results.");
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
  if (params_->using_simulation) {
    this->PrintCalibrations(calibrations_perturbed_,
                            "perturbed_calibrations.txt");
  }
  this->PrintCalibrationErrors();
  this->SaveLidarResults();
  this->SaveCameraResults();
  this->PrintErrors();
}

void CalibrationVerification::SetConfig(const std::string &calib_config) {
  calibration_config_ = calib_config;
}

void CalibrationVerification::SetInitialCalib(
    const std::vector<vicon_calibration::CalibrationResult> &calib) {
  calibrations_initial_ = calib;
}

void CalibrationVerification::SetPeturbedCalib(
    const std::vector<vicon_calibration::CalibrationResult> &calib) {
  calibrations_perturbed_ = calib;
}

void CalibrationVerification::SetOptimizedCalib(
    const std::vector<vicon_calibration::CalibrationResult> &calib) {
  calibrations_result_ = calib;
}

void CalibrationVerification::SetParams(
    std::shared_ptr<CalibratorConfig> &params) {
  params_ = params;

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

void CalibrationVerification::SaveLidarResults() {
  // Iterate over each lidar
  std::vector<Eigen::Affine3d, AlignAff3d> T_VICONBASE_TGTS;
  for (uint8_t lidar_iter = 0; lidar_iter < params_->lidar_params.size();
       lidar_iter++) {
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
    if (params_->using_simulation) {
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
    pcl::PCLPointCloud2::Ptr cloud_pc2 =
        boost::make_shared<pcl::PCLPointCloud2>();
    PointCloud::Ptr scan = boost::make_shared<PointCloud>();
    PointCloud::Ptr scan_trans_est = boost::make_shared<PointCloud>();
    PointCloud::Ptr scan_trans_opt = boost::make_shared<PointCloud>();
    PointCloud::Ptr target = boost::make_shared<PointCloud>();
    PointCloud::Ptr target_transformed = boost::make_shared<PointCloud>();
    PointCloud::Ptr targets_combined = boost::make_shared<PointCloud>();

    // iterate through all scans in bag for this lidar
    rosbag::View view(bag_, rosbag::TopicQuery(topic), ros::TIME_MIN,
                      ros::TIME_MAX, true);
    int counter = 0;
    for (auto iter = view.begin(); iter != view.end(); iter++) {
      boost::shared_ptr<sensor_msgs::PointCloud2> lidar_msg =
          iter->instantiate<sensor_msgs::PointCloud2>();
      ros::Time time_current = lidar_msg->header.stamp;

      // check if we want to use this scan
      if (time_current > time_last + time_increment_) {
        // set time and transforms
        lookup_time_ = time_current;
        this->LoadLookupTree();
        time_last = time_current;

        // load scan and transform to viconbase frame
        pcl_conversions::toPCL(*lidar_msg, *cloud_pc2);
        pcl::fromPCLPointCloud2(*cloud_pc2, *scan);
        pcl::transformPointCloud(*scan, *scan_trans_est,
                                 TA_VICONBASE_SENSOR_est);
        pcl::transformPointCloud(*scan, *scan_trans_opt,
                                 TA_VICONBASE_SENSOR_opt);

        // load targets and transform to viconbase frame
        try {
          T_VICONBASE_TGTS = utils::GetTargetLocation(
              params_->target_params, params_->vicon_baselink_frame,
              lookup_time_, lookup_tree_);
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

        // save all the scans that are viewed
        counter++;
        std::string save_path1, save_path2, save_path3;
        save_path1 =
            current_save_path + "scan_" + std::to_string(counter) + "_est.pcd";
        save_path2 =
            current_save_path + "scan_" + std::to_string(counter) + "_opt.pcd";
        save_path3 = current_save_path + "scan_" + std::to_string(counter) +
                     "_targets.pcd";

        pcl::io::savePCDFileBinary(save_path1, *scan_trans_est);
        pcl::io::savePCDFileBinary(save_path2, *scan_trans_opt);
        pcl::io::savePCDFileBinary(save_path3, *targets_combined);

        std::vector<Eigen::Vector3d, AlignVec3d> lidar_errors_opt =
            GetLidarErrors(T_VICONBASE_TGTS, scan_trans_opt, lidar_iter);
        lidar_errors_opt_.insert(lidar_errors_opt_.end(),
                                 lidar_errors_opt.begin(),
                                 lidar_errors_opt.begin());
        std::vector<Eigen::Vector3d, AlignVec3d> lidar_errors_init =
            GetLidarErrors(T_VICONBASE_TGTS, scan_trans_est, lidar_iter);
        lidar_errors_init_.insert(lidar_errors_init_.end(),
                                  lidar_errors_init.begin(),
                                  lidar_errors_init.begin());

        if (counter == max_lidar_results_) {
          return;
        }
      }
    }
  }
}

std::vector<Eigen::Vector3d, AlignVec3d>
CalibrationVerification::GetLidarErrors(
    const std::vector<Eigen::Affine3d, AlignAff3d> &T_VICONBASE_TGTS,
    const PointCloud::Ptr &scan_viconbase, uint8_t &lidar_id) {
  std::shared_ptr<LidarExtractor> lidar_extractor;
  std::vector<Eigen::Vector3d, AlignVec3d> lidar_errors;
  for (int i = 0; i < T_VICONBASE_TGTS.size(); i++) {
    std::string extractor_type = params_->target_params[i]->extractor_type;
    if (extractor_type == "CYLINDER") {
      lidar_extractor =
          std::make_shared<vicon_calibration::CylinderLidarExtractor>();
    } else if (extractor_type == "DIAMOND") {
      lidar_extractor =
          std::make_shared<vicon_calibration::DiamondLidarExtractor>();
    } else {
      throw std::invalid_argument{
          "Invalid extractor type. Options: CYLINDER, DIAMOND"};
    }
    lidar_extractor->SetLidarParams(params_->lidar_params[lidar_id]);
    lidar_extractor->SetTargetParams(params_->target_params[i]);
    lidar_extractor->SetShowMeasurements(false);
    lidar_extractor->ProcessMeasurement(T_VICONBASE_TGTS[i].matrix(),
                                        scan_viconbase);
    if (lidar_extractor->GetMeasurementValid()) {
      // get measured keypoints
      PointCloud::Ptr measured_keypoints = lidar_extractor->GetMeasurement();

      // get estimated (optimization or initial) keypoint locations
      PointCloud::Ptr estimated_keypoints;
      if (params_->target_params[i]->keypoints_lidar.size() > 0) {
        for (Eigen::Vector3d keypoint :
             params_->target_params[i]->keypoints_lidar) {
          estimated_keypoints->push_back(utils::EigenPointToPCL(keypoint));
        }
      } else {
        estimated_keypoints = params_->target_params[i]->template_cloud;
      }
      pcl::transformPointCloud(*estimated_keypoints, *estimated_keypoints,
                               T_VICONBASE_TGTS[i]);

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
      for (int i = 0; i < correspondences->size(); i++) {
        measurement_index = correspondences->at(i).index_query;
        estimated_index = correspondences->at(i).index_match;
        error =
            utils::PCLPointToEigen(measured_keypoints->at(measurement_index)) -
            utils::PCLPointToEigen(estimated_keypoints->at(estimated_index));
        error[0] = std::abs(error[0]);
        error[1] = std::abs(error[1]);
        error[2] = std::abs(error[2]);
        lidar_errors.push_back(error);
      }
    }
  }
  return lidar_errors;
}

void CalibrationVerification::SaveCameraResults() {
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
    ros::Time time_zero(0, 0);

    // get initial calibration and optimized calibration
    Eigen::Affine3d TA_VICONBASE_SENSOR_est, TA_VICONBASE_SENSOR_opt;
    if (params_->using_simulation) {
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
    std::shared_ptr<cv::Mat> current_image = std::make_shared<cv::Mat>();
    std::shared_ptr<cv::Mat> final_image;
    for (auto iter = view.begin(); iter != view.end(); iter++) {
      sensor_msgs::ImageConstPtr img_msg =
          iter->instantiate<sensor_msgs::Image>();
      ros::Time time_current = img_msg->header.stamp;
      // skip first instance to avoid errors at beginning of bag
      if (time_last == time_zero) {
        time_last = time_current;
      }
      *current_image =
          cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8)
              ->image;
      if (time_current > time_last + time_increment_) {
        lookup_time_ = time_current;
        this->LoadLookupTree();
        time_last = time_current;
        std::vector<Eigen::Affine3d, AlignAff3d> T_VICONBASE_TGTS =
            utils::GetTargetLocation(params_->target_params,
                                     params_->vicon_baselink_frame,
                                     lookup_time_, lookup_tree_);
        final_image = this->ProjectTargetToImage(
            current_image, T_VICONBASE_TGTS, TA_VICONBASE_SENSOR_est.matrix(),
            cam_iter, cv::Scalar(0, 0, 255));
        if (num_tgts_in_img_ > 0) {
          counter++;

          // save image with targets
          final_image = this->ProjectTargetToImage(
              final_image, T_VICONBASE_TGTS, TA_VICONBASE_SENSOR_opt.matrix(),
              cam_iter, cv::Scalar(255, 0, 0));
          std::string save_path =
              current_save_path + "image_" + std::to_string(counter) + ".jpg";
          cv::imwrite(save_path, *final_image);

          // extract measurements
          std::vector<Eigen::Vector2d, AlignVec2d> camera_errors_opt =
              GetCameraErrors(T_VICONBASE_TGTS,
                              TA_VICONBASE_SENSOR_opt.matrix(), current_image,
                              cam_iter);
          camera_errors_opt_.insert(camera_errors_opt_.end(),
                                    camera_errors_opt.begin(),
                                    camera_errors_opt.begin());
          std::vector<Eigen::Vector2d, AlignVec2d> camera_errors_init =
              GetCameraErrors(T_VICONBASE_TGTS,
                              TA_VICONBASE_SENSOR_est.matrix(), current_image,
                              cam_iter);
          camera_errors_init_.insert(camera_errors_init_.end(),
                                     camera_errors_init.begin(),
                                     camera_errors_init.begin());
        }
      }
      if (counter == max_image_results_) {
        break;
      }
    } // end bag iter
  }   // end cam iter
}

std::vector<Eigen::Vector2d, AlignVec2d>
CalibrationVerification::GetCameraErrors(
    const std::vector<Eigen::Affine3d, AlignAff3d> &T_VICONBASE_TGTS,
    const Eigen::Matrix4d &T_VICONBASE_CAMERA,
    const std::shared_ptr<cv::Mat> &img, uint8_t &camera_id) {
  std::shared_ptr<CameraExtractor> camera_extractor;
  std::vector<Eigen::Vector2d, AlignVec2d> camera_errors;
  for (int i = 0; i < T_VICONBASE_TGTS.size(); i++) {
    Eigen::Matrix4d T_CAMERA_TGT = utils::InvertTransform(T_VICONBASE_CAMERA) *
                                   T_VICONBASE_TGTS[i].matrix();
    std::string extractor_type = params_->target_params[i]->extractor_type;
    if (extractor_type == "CYLINDER") {
      camera_extractor =
          std::make_shared<vicon_calibration::CylinderCameraExtractor>();
    } else if (extractor_type == "DIAMOND") {
      camera_extractor =
          std::make_shared<vicon_calibration::DiamondCameraExtractor>();
    } else {
      throw std::invalid_argument{
          "Invalid extractor type. Options: CYLINDER, DIAMOND"};
    }
    camera_extractor->SetCameraParams(params_->camera_params[camera_id]);
    camera_extractor->SetTargetParams(params_->target_params[i]);
    camera_extractor->SetShowMeasurements(false);
    camera_extractor->ProcessMeasurement(T_CAMERA_TGT, *img);
    if (camera_extractor->GetMeasurementValid()) {
      // extract measured keypoints
      pcl::PointCloud<pcl::PointXY>::Ptr measured_keypoints =
          camera_extractor->GetMeasurement();
      pcl::PointCloud<pcl::PointXYZ>::Ptr measured_keypoints_3d;
      pcl::PointXYZ point3d;
      for (int i = 0; i < measured_keypoints->size(); i++) {
        point3d.x = measured_keypoints->at(i).x;
        point3d.y = measured_keypoints->at(i).y;
        point3d.z = 0;
        measured_keypoints_3d->push_back(point3d);
      }

      // get estimated (optimization or initial) keypoint locations
      PointCloud::Ptr keypoints_target_frame;
      if (params_->target_params[i]->keypoints_camera.size() > 0) {
        for (Eigen::Vector3d keypoint :
             params_->target_params[i]->keypoints_camera) {
          keypoints_target_frame->push_back(utils::EigenPointToPCL(keypoint));
        }
      } else {
        keypoints_target_frame = params_->target_params[i]->template_cloud;
      }

      // project points to image plane and save as cloud
      PointCloud::Ptr keypoints_projected = utils::ProjectPoints(
          keypoints_target_frame,
          params_->camera_params[camera_id]->camera_model,
          params_->camera_params[camera_id]->images_distorted, T_CAMERA_TGT);

      // get correspondences
      pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
          corr_est;
      boost::shared_ptr<pcl::Correspondences> correspondences =
          boost::make_shared<pcl::Correspondences>();
      corr_est.setInputSource(measured_keypoints_3d);
      corr_est.setInputTarget(keypoints_projected);
      corr_est.determineCorrespondences(*correspondences, max_pixel_cor_dist_);

      // get distances between correspondences
      int measurement_index, estimated_index;
      Eigen::Vector3d error3d;
      Eigen::Vector2d error2d;
      for (int i = 0; i < correspondences->size(); i++) {
        measurement_index = correspondences->at(i).index_query;
        estimated_index = correspondences->at(i).index_match;
        error3d =
            utils::PCLPointToEigen(
                measured_keypoints_3d->at(measurement_index)) -
            utils::PCLPointToEigen(keypoints_projected->at(estimated_index));
        error2d[0] = std::abs(error3d[0]);
        error2d[1] = std::abs(error3d[1]);
        camera_errors.push_back(error2d);
      }
    }
  }
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

void CalibrationVerification::PrintErrors() {
  std::string output_path = results_directory_ + "errors_summary.txt";
  std::ofstream file(output_path);

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

  norms_summed = 0;
  for (int i = 0; i < camera_errors_opt_.size(); i++) {
    norms_summed += camera_errors_opt_[i].norm();
  }
  norms_averaged = norms_summed / camera_errors_opt_.size();

  file << "\n-----------------------------------------------------------\n\n"
       << "Outputting Error Statistics for Optimized Camera Calibrations:\n"
       << "Average Error Norm (m): " << norms_averaged << "\n"
       << "Samples Used: " << camera_errors_opt_.size() << "\n";

  norms_summed = 0;
  for (int i = 0; i < camera_errors_init_.size(); i++) {
    norms_summed += camera_errors_init_[i].norm();
  }
  norms_averaged = norms_summed / camera_errors_init_.size();

  file << "\n-----------------------------------------------------------\n\n"
       << "Outputting Error Statistics for Initial Camera Calibrations:\n"
       << "Average Error Norm (m): " << norms_averaged << "\n"
       << "Samples Used: " << camera_errors_init_.size() << "\n";
}

} // end namespace vicon_calibration
