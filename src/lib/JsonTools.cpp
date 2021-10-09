#include <vicon_calibration/JsonTools.h>

#include <boost/filesystem.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vicon_calibration/Utils.h>

namespace vicon_calibration {

JsonTools::JsonTools(const CalibratorInputs& inputs) : inputs_(inputs) {}

std::shared_ptr<TargetParams>
    JsonTools::LoadTargetParams(const std::string& target_config_full_path) {
  LOG_INFO("Loading Target Config File: %s", target_config_full_path.c_str());

  std::shared_ptr<TargetParams> params = std::make_shared<TargetParams>();
  nlohmann::json J_target;
  std::ifstream file(target_config_full_path);
  file >> J_target;

  std::vector<float> vect1;
  for (const auto& value : J_target["crop_scan"]) { vect1.push_back(value); }
  Eigen::VectorXf crop_scan(6);
  if (vect1.size() != 6) {
    throw std::invalid_argument{
        "Wrong number of inputs to crop_scan. Required: 6"};
  }
  crop_scan << vect1[0], vect1[1], vect1[2], vect1[3], vect1[4], vect1[5];
  params->crop_scan = crop_scan;

  std::vector<double> vect2;
  for (const auto& value : J_target["crop_image"]) { vect2.push_back(value); }
  if (vect2.size() != 2) {
    throw std::invalid_argument{
        "Wrong number of inputs to crop_image. Required: 2"};
  }
  Eigen::Vector2d crop_image;
  crop_image << vect2[0], vect2[1];
  params->crop_image = crop_image;

  // load template cloud
  std::string template_name = J_target["template_cloud"];
  std::string template_cloud_path = inputs_.target_data_path + template_name;
  PointCloud::Ptr template_cloud = std::make_shared<PointCloud>();
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud_path,
                                          *template_cloud) == -1) {
    LOG_ERROR("Couldn't read template file: %s\n", template_cloud_path.c_str());
  }
  params->template_cloud = template_cloud;

  params->is_target_2d = J_target["is_target_2d"];

  for (const auto& keypoint : J_target["keypoints_lidar"]) {
    Eigen::Vector3d point;
    point << keypoint["x"], keypoint["y"], keypoint["z"];
    params->keypoints_lidar.push_back(point);
  }
  for (const auto& keypoint : J_target["keypoints_camera"]) {
    Eigen::Vector3d point;
    point << keypoint["x"], keypoint["y"], keypoint["z"];
    params->keypoints_camera.push_back(point);
  }
  return params;
}

std::shared_ptr<TargetParams>
    JsonTools::LoadTargetParams(const nlohmann::json& J_in) {
  std::string target_config = J_in.at("target_config");
  std::string target_config_full_path =
      inputs_.target_config_path + target_config;
  std::shared_ptr<TargetParams> params =
      LoadTargetParams(target_config_full_path);
  params->frame_id = J_in["frame_id"];
  params->extractor_type = J_in["extractor_type"];
  params->target_config_path = target_config_full_path;
  return params;
}

std::shared_ptr<CameraParams>
    JsonTools::LoadCameraParams(const nlohmann::json& J_in) {
  std::string intrinsics_filename = J_in.at("intrinsics");
  intrinsics_filename = inputs_.camera_intrinsics_path + intrinsics_filename;
  if (!boost::filesystem::exists(intrinsics_filename)) {
    LOG_ERROR("Cannot find intrinsics filename in the data folder, or at "
              "absolute path %s",
              J_in["intrinsics"].dump().c_str());
    throw std::invalid_argument{"Invalid intrinsics path."};
  }
  std::shared_ptr<CameraParams> params = std::make_shared<CameraParams>(
      intrinsics_filename, J_in["topic"], J_in["frame"]);
  return params;
}

std::shared_ptr<LidarParams>
    JsonTools::LoadLidarParams(const nlohmann::json& J_in) {
  std::shared_ptr<LidarParams> params = std::make_shared<LidarParams>();
  params->topic = J_in["topic"];
  params->frame = J_in["frame"];
  params->max_angular_resolution_deg = J_in.at("max_angular_resolution_deg");
  return params;
}

std::shared_ptr<CalibratorConfig> JsonTools::LoadViconCalibratorParams() {
  std::shared_ptr<CalibratorConfig> params =
      std::make_shared<CalibratorConfig>();

  // Copy required input params to main params struct
  params->bag_file = inputs_.bag;
  params->initial_calibration_file = inputs_.initial_calibration;
  params->show_camera_measurements = inputs_.show_camera_measurements;
  params->show_lidar_measurements = inputs_.show_lidar_measurements;

  LOG_INFO("Loading ViconCalibrator Config File: %s",
           inputs_.calibration_config.c_str());
  nlohmann::json J;
  std::ifstream file(inputs_.calibration_config);
  file >> J;

  params->min_target_motion = J["min_target_motion"];
  params->min_target_rotation = J["min_target_rotation"];
  params->max_target_velocity = J["max_target_velocity"];
  params->vicon_baselink_frame = J["vicon_baselink_frame"];
  params->time_steps = J["time_steps"];
  params->max_measurements = J["max_measurements"];
  params->use_loop_closure_measurements = J["use_loop_closure_measurements"];
  params->optimizer_type = J["optimizer_type"];

  std::vector<double> crop_time;
  for (const auto& value : J["crop_time"]) { crop_time.push_back(value); }
  params->crop_time = crop_time;

  int counter = 0;
  for (const auto& target : J["targets"]) {
    std::shared_ptr<TargetParams> target_info = LoadTargetParams(target);
    params->target_params.push_back(target_info);
  }

  for (const auto& camera : J["camera_params"]) {
    std::shared_ptr<CameraParams> cam_params = LoadCameraParams(camera);
    params->camera_params.push_back(cam_params);
  }

  for (const auto& lidar : J["lidar_params"]) {
    std::shared_ptr<LidarParams> lid_params = LoadLidarParams(lidar);
    params->lidar_params.push_back(lid_params);
  }

  LOG_INFO("Successfully loaded calibrator parameters.");
  return params;
}

} // end namespace vicon_calibration
