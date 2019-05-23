#ifndef VICON_CALIBRATION_UTILS_HPP
#define VICON_CALIBRATION_UTILS_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vicon_calibration {
  typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
  typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudColor;
}

#endif // VICON_CALIBRATION_UTILS_HPP
