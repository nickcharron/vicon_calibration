#pragma once

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vicon_calibration {

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudColor;

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

} // namespace utils

} // namespace vicon_calibration
