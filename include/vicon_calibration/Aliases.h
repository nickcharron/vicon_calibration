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

} // namespace vicon_calibration
