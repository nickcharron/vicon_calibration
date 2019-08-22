#pragma once

#include <string>
#include <Eigen/Geometry>

namespace vicon_calibration {

struct LidarParams {
  std::string topic;
  std::string frame;
  double time_steps;
};

struct CameraParams {
  std::string topic;
  std::string frame;
  std::string intrinsics;
  double time_steps;
};

struct RegistrationParams {
  double max_correspondance_distance{1}; // correspondences with higher
                                         // distances will be ignored maximum
  int max_iterations{10};                // iterations in the optimization
  double transform_epsilon{1e-8}; // The epsilon (difference) between the
                                  // previous transformation and the current
                                  // estimated transformation required for
                                  // convergence
  double euclidean_epsilon{1e-4}; // The sum of Euclidean squared errors for
                                  // convergence requirement
  bool show_transform{false};            // show the transform used for the
                                         // measurement calc
  double dist_acceptance_criteria{0.05}; // maximum distance measurement from
                                         // initial guess for acceptance
  double rot_acceptance_criteria{0.5};  // maximum rotation measurement from
                                        // initial guess for acceptance
};

struct CylinderTgtParams {
  double radius;
  double height;
  double crop_threshold;
  std::string template_cloud;     // full path to template cloud
  std::vector<std::string> vicon_target_frames;
};

struct ImageProcessingParams {
  int num_intersections;
  double min_length_ratio;
  double max_gap_ratio;
  double canny_ratio;
  std::string encoding;
  bool show_measurements;
};

struct LidarMeasurement {
  Eigen::Affine3d measurement;
  int lidar_id;
  int target_id;
};

struct CameraMeasurement {
  Eigen::Affine3d measurement;
  int camera_id;
  int target_id;
};

struct CalibrationResult {
  Eigen::Affine3d transform;
  std::string to_frame;
  std::string from_frame;
};

} // end namespace vicon_calibration
