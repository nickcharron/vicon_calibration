#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vicon_calibration {

/**
 * @brief class for extracting cylinder measurements from lidar scan
 */
class LidarCylExtractor {
public:
  LidarCylExtractor() = default;

  LidarCylExtractor(pcl::PointCloud<pcl::PointXYZ>::Ptr& template_cloud);

  ~LidarCylExtractor() = default;

  void SetTemplateCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& template_cloud){
    template_cloud_ = template_cloud;
  }

private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud_;

};

} // end namespace vicon_calibration
