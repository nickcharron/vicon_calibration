#pragma once

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sys/time.h>

namespace vicon_calibration {

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudColor = pcl::PointCloud<pcl::PointXYZRGB>;
using PointCloudColorPtr = PointCloudColor::Ptr;

typedef Eigen::aligned_allocator<Eigen::Vector4d> AlignVec4d;
typedef Eigen::aligned_allocator<Eigen::Vector3d> AlignVec3d;
typedef Eigen::aligned_allocator<Eigen::Vector2d> AlignVec2d;
typedef Eigen::aligned_allocator<Eigen::Matrix3d> AlignMat3d;
typedef Eigen::aligned_allocator<Eigen::Matrix4d> AlignMat4d;
typedef Eigen::aligned_allocator<Eigen::Affine3d> AlignAff3d;

} // namespace vicon_calibration
