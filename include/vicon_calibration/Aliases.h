#pragma once

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sys/time.h>

namespace vicon_calibration {

using json = nlohmann::json;
using PointCloud = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudColor = pcl::PointCloud<pcl::PointXYZRGB>;
using PointCloudColorPtr = PointCloudColor::Ptr;
using AlignVec3d = Eigen::aligned_allocator<Eigen::Vector3d>;
using AlignVec2d = Eigen::aligned_allocator<Eigen::Vector2d>;
using AlignAff3d = Eigen::aligned_allocator<Eigen::Affine3d>;
using AlignMat4d = Eigen::aligned_allocator<Eigen::Matrix4d>;

} // namespace vicon_calibration
