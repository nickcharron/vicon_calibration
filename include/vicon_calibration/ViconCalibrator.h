#pragma once

#include "vicon_calibration/CylinderCameraExtractor.h"
#include "vicon_calibration/CylinderLidarExtractor.h"
#include "vicon_calibration/GTSAMGraph.h"
#include "vicon_calibration/TfTree.h"
#include "vicon_calibration/params.h"
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <ros/time.h>
#include <rosbag/bag.h>

namespace vicon_calibration {

/**
 * @brief class for running the vicon calibration
 */
class ViconCalibrator {

  struct CalibratorConfig {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::string bag_file;
    std::string initial_calibration_file;
    bool lookup_tf_calibrations{false};
    std::string vicon_baselink_frame;
    bool show_measurements{false};
    bool save_results{true};
    std::string output_directory;
    Eigen::VectorXd initial_guess_perturbation; // for testing sim
    double min_measurement_motion{0.01};
    std::vector<std::shared_ptr<vicon_calibration::TargetParams>>
        target_params_list;
    std::vector<std::shared_ptr<vicon_calibration::CameraParams>> camera_params;
    std::vector<std::shared_ptr<vicon_calibration::LidarParams>> lidar_params;
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

  void LoadLookupTree();

  void GetInitialCalibration(std::string &sensor_frame, SensorType type,
                             uint8_t &sensor_id);

  void GetInitialCalibrationPerturbed(std::string &sensor_frame,
                                      SensorType type, uint8_t &sensor_id);

  /**
   * @brief get initial guess of where the targets are located at the current
   * time point
   * @param sensor_frame
   * @return T_sensor_tgts_estimated estimated transform from targets to sensor
   * frame
   */
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
  GetInitialGuess(std::string &sensor_frame);

  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
  GetTargetLocation();

  /**
   * @brief Compute the measurements from one lidar
   * @param lidar_iter
   */
  void GetLidarMeasurements(uint8_t &lidar_iter);

  /**
   * @brief Compute the measurements from the cameras
   * @param cam_iter
   */
  void GetCameraMeasurements(uint8_t &cam_iter);

  void GetLoopClosureMeasurements();

  void SetCalibrationInitials();

  void ProcessCalibResults();

  void ViewClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr c1,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr c2,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr c3);

  CalibratorConfig params_;
  std::string results_directory_;
  std::string config_file_path_;
  std::shared_ptr<LidarExtractor> lidar_extractor_;
  std::shared_ptr<CameraExtractor> camera_extractor_;
  ros::Time lookup_time_;
  TfTree estimate_extrinsics_, lookup_tree_;
  std::vector<vicon_calibration::LidarMeasurement> lidar_measurements_;
  std::vector<vicon_calibration::CameraMeasurement> camera_measurements_;
  std::vector<vicon_calibration::CalibrationResult> calibrations_result_,
      calibrations_initial_,
      calibrations_perturbed_; // pertubed use for testing with simulation ONLY
  rosbag::Bag bag_;
  vicon_calibration::GTSAMGraph graph_;
  Eigen::MatrixXd T_VICONBASE_SENSOR_ = Eigen::MatrixXd(4,4);
  Eigen::MatrixXd T_VICONBASE_SENSOR_pert_ = Eigen::MatrixXd(4,4);
          // Pert for testing simulation ONLY
};

} // end namespace vicon_calibration
