#include "vicon_calibration/JsonTools.h"
#include "vicon_calibration/utils.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vicon_calibration {

std::string JsonTools::GetJSONFileNameConfig(const std::string &file_name) {
  std::string file_location = __FILE__;
  std::string current_location = "src/lib/JsonTools.cpp";
  file_location.erase(file_location.end() - current_location.size(), file_location.end());
  file_location += "config/";
  file_location += file_name;
  return file_location;
}

std::string JsonTools::GetJSONFileNameData(const std::string &file_name) {
  std::string file_location = __FILE__;
  std::string current_location = "src/lib/JsonTools.cpp";
  file_location.erase(file_location.end() - current_location.size(), file_location.end());
  file_location += "data/";
  file_location += file_name;
  return file_location;
}

std::shared_ptr<TargetParams>
JsonTools::LoadTargetParams(const std::string &file_name) {
  LOG_INFO("Loading Target Config File: %s", file_name.c_str());
  std::shared_ptr<TargetParams> params = std::make_shared<TargetParams>();
  nlohmann::json J_target;
  std::ifstream file(file_name);
  file >> J_target;

  std::vector<double> vect1, vect2;
  for (const auto &value : J_target["crop_scan"]) {
    vect1.push_back(value);
  }
  Eigen::Vector3d crop_scan;
  crop_scan << vect1[0], vect1[1], vect1[2];
  params->crop_scan = crop_scan;
  for (const auto &value : J_target["crop_image"]) {
    vect2.push_back(value);
  }
  Eigen::Vector2d crop_image;
  crop_image << vect2[0], vect2[1];
  params->crop_image = crop_image;
  std::string template_cloud_path =
      GetJSONFileNameData(J_target.at("template_cloud"));
  pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud_path,
                                          *template_cloud) == -1) {
    LOG_ERROR("Couldn't read template file: %s\n",
              template_cloud_path.c_str());
  }
  params->template_cloud = template_cloud;
  for (const auto &keypoint : J_target["keypoints_lidar"]) {
    Eigen::Vector3d point;
    point << keypoint.at("x"), keypoint.at("y"), keypoint.at("z");
    params->keypoints_lidar.push_back(point);
  }
  for (const auto &keypoint : J_target["keypoints_camera"]) {
    Eigen::Vector3d point;
    point << keypoint.at("x"), keypoint.at("y"), keypoint.at("z");
    params->keypoints_camera.push_back(point);
  }
  return params;
}

std::shared_ptr<TargetParams> JsonTools::LoadTargetParams(const nlohmann::json &J_in) {
  std::string target_config = J_in.at("target_config");
  std::string target_config_full_path = GetJSONFileNameConfig(target_config);
  std::shared_ptr<TargetParams> params = LoadTargetParams(target_config_full_path);
  params->frame_id = J_in.at("frame_id");
  params->extractor_type = J_in.at("extractor_type");
  params->target_config_path = target_config_full_path;
  return params;
}

std::shared_ptr<CameraParams>
JsonTools::LoadCameraParams(const nlohmann::json &J_in) {
  std::shared_ptr<CameraParams> params = std::make_shared<CameraParams>();
  params->topic = J_in.at("topic");
  params->frame = J_in.at("frame");
  std::string intrinsics_filename = J_in.at("intrinsics");
  params->intrinsics = GetJSONFileNameData(intrinsics_filename);
  params->time_steps = J_in.at("time_steps");
  params->images_distorted = J_in.at("images_distorted");
  return params;
}

std::shared_ptr<LidarParams>
JsonTools::LoadLidarParams(const nlohmann::json &J_in) {
  std::shared_ptr<LidarParams> params = std::make_shared<LidarParams>();
  params->topic = J_in.at("topic");
  params->frame = J_in.at("frame");
  params->time_steps = J_in.at("time_steps");
  return params;
}

// TODO: add try and catch blocks
std::shared_ptr<CalibratorConfig>
JsonTools::LoadViconCalibratorParams(const std::string &file_name) {
  LOG_INFO("Loading ViconCalibrator Config File: %s", file_name.c_str());
  nlohmann::json J;
  std::ifstream file(file_name);
  file >> J;

  std::shared_ptr<CalibratorConfig> params = std::make_shared<CalibratorConfig>();
  params->bag_file = J["bag_file"];
  params->initial_calibration_file = J["initial_calibration"];
  params->lookup_tf_calibrations = J["lookup_tf_calibrations"];
  std::vector<double> vect;
  for (const auto &value : J["initial_guess_perturb"]) {
    vect.push_back(value);
  }
  Eigen::VectorXd tmp(6, 1);
  tmp << vect[0], vect[1], vect[2], vect[3], vect[4], vect[5];
  params->initial_guess_perturbation = tmp;
  params->min_measurement_motion = J["min_measurement_motion"];
  params->vicon_baselink_frame = J["vicon_baselink_frame"];
  params->show_measurements = J["show_measurements"];
  params->run_verification = J["run_verification"];

  for (const auto &target : J["targets"]) {
    std::shared_ptr<TargetParams> target_info = LoadTargetParams(target);
    params->target_params.push_back(target_info);
  }

  for (const auto &camera : J["camera_params"]) {
    std::shared_ptr<CameraParams> cam_params = LoadCameraParams(camera);
    params->camera_params.push_back(cam_params);
  }

  for (const auto &lidar : J["lidar_params"]) {
    std::shared_ptr<LidarParams> lid_params = LoadLidarParams(lidar);
    params->lidar_params.push_back(lid_params);
  }

  return params;
}

} // end namespace vicon_calibration
