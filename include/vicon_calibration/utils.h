#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Geometry>

namespace vicon_calibration {

  typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
  typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudColor;
  
namespace utils {

Eigen::MatrixXd RoundMatrix(const Eigen::MatrixXd &M, int precision);

bool IsRotationMatrix(Eigen::Matrix3d R);

bool IsTransformationMatrix(Eigen::Matrix4d T);

} // namespace utils

} // namespace vicon_calibration
