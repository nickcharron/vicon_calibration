#pragma once

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "vicon_calibration/params.h"

namespace vicon_calibration {

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudColor;

#define FILENAME \
    (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define LOG_ERROR(M, ...) \
    fprintf(              \
      stderr, "[ERROR] [%s:%d] " M "\n", FILENAME, __LINE__, ##__VA_ARGS__)

#define LOG_INFO(M, ...) fprintf(stdout, "[INFO] " M "\n", ##__VA_ARGS__)

#define LOG_WARN(M, ...) fprintf(stdout, "[WARNING] " M "\n", ##__VA_ARGS__)

namespace utils {

Eigen::MatrixXd RoundMatrix(const Eigen::MatrixXd &M, int precision);

bool IsRotationMatrix(const Eigen::Matrix3d R);

bool IsTransformationMatrix(const Eigen::Matrix4d T);

Eigen::Affine3d PerturbTransform(const Eigen::Affine3d &T_in,
                                 const std::vector<double> &perturbations);

Eigen::Vector3d invSkewTransform(const Eigen::Matrix3d &M);

Eigen::Matrix3d skewTransform(const Eigen::Vector3d &V);

Eigen::Vector3d RToLieAlgebra(const Eigen::Matrix3d &R);

Eigen::Matrix3d LieAlgebraToR(const Eigen::Vector3d &eps);

Eigen::Matrix4d RemoveYaw(const Eigen::Matrix4d &T_in);

Eigen::Matrix4d RemoveYaw2(const Eigen::Matrix4d &T_in);

void OutputTransformInformation(Eigen::Affine3d &T, std::string transform_name);

void OutputLidarMeasurements(std::vector<vicon_calibration::LidarMeasurement> &measurements);

void OutputCalibrations(
    std::vector<vicon_calibration::CalibrationResult> &calib,
    std::string output_string);

} // namespace utils

} // namespace vicon_calibration
