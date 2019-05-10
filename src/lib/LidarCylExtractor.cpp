#include "vicon_calibration/LidarCylExtractor.h"

namespace vicon_calibration {

LidarCylExtractor::LidarCylExtractor(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &template_cloud)
    : template_cloud_(template_cloud) {}


} // end namespace vicon_calibration
