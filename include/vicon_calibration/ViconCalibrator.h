#pragma once

#include "vicon_calibration/params.h"
#include "beam_calibration/TfTree.h"
#include "vicon_calibration/LidarCylExtractor.h"
// #include "vicon_calibration/CamCylExtractor.h"
#include <rosbag/bag.h>
#include <ros/time.h>
#include <Eigen/Geometry>

namespace vicon_calibration {

/**
 * @brief class for running the vicon calibration
 */
class ViconCalibrator {

  struct CalibratorConfig {
    std::string bag_file;
    std::string initial_calibration_file;
    bool lookup_tf_calibrations;
    std::string vicon_baselink_frame;
    std::vector<double> initial_guess_perturbation; // for testing sim
    vicon_calibration::ImageProcessingParams image_processing_params;
    vicon_calibration::RegistrationParams registration_params;
    vicon_calibration::CylinderTgtParams target_params;
    std::vector<vicon_calibration::CameraParams> camera_params;
    std::vector<vicon_calibration::LidarParams> lidar_params;
  };

public:
  /**
   * @brief default constructor
   */
  ViconCalibrator() = default;

  /**
   * @brief default deconstructor
   */
  ~ViconCalibrator() = default;

  /**
   * @brief run calibration
   */
  void RunCalibration(std::string config_file = "ViconCalibrationConfig.json");

private:
  /**
   * @brief gets full name to file inside data subfolder
   * @param file_name name of file to find
   */
  std::string GetJSONFileNameData(std::string file_name);

  /**
   * @brief gets full name to file inside config subfolder
   * @param file_name name of file to find
   */
  std::string GetJSONFileNameConfig(std::string file_name);

  /**
   * @brief load parameters from json config file
   * @param file_name name of file to find
   */
  void LoadJSON(std::string file_name);

  /**
   * @brief loads extrinsic estimates and saves as tf tree object. These come
   * from either a JSON or from the tf messages in the bag
   */
  void LoadEstimatedExtrinsics();
  /**
   * @brief get initial guess of where the targets are located at the current
   * time point
   * @param time
   * @param sensor_frame
   * @return T_sensor_tgts_estimated estimated transform from targets to sensor
   * frame
   */
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
  GetInitialGuess(ros::Time &time, std::string &sensor_frame);

  /**
   * @brief Compute the measurements from one lidar
   * @param lidar_iter
   */
  void GetLidarMeasurements(uint8_t &lidar_iter);

  /**
   * @brief Compute the measurements from the cameras
   * @param bag
   * @param topic
   * @param frame
   */
  void GetCameraMeasurements(rosbag::Bag &bag, std::string &topic,
                             std::string &frame);

  CalibratorConfig params_;
  beam_calibration::TfTree estimate_extrinsics_;
  vicon_calibration::LidarCylExtractor lidar_extractor_;
  //vicon_calibration::CameraCylExtractor camera_extractor_;
  std::vector<vicon_calibration::LidarMeasurement> lidar_measurements_;
  std::vector<vicon_calibration::CameraMeasurement> camera_measurements_;
  std::vector<vicon_calibration::CalibrationResult> lidar_calibration_results_,
      camera_calibration_results_;
  rosbag::Bag bag_;
};

} // end namespace vicon_calibration
