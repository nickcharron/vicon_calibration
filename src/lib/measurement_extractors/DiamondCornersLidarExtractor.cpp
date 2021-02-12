#include "vicon_calibration/measurement_extractors/DiamondCornersLidarExtractor.h"

#include <math.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/surface/concave_hull.h>

// #include <pcl/io/pcd_io.h>

namespace vicon_calibration {

void DiamondCornersLidarExtractor::GetKeypoints() {
  measurement_valid_ = true;

  // setup icp
  PointCloud::Ptr scan_registered = boost::make_shared<PointCloud>();
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setTransformationEpsilon(icp_transform_epsilon_);
  icp.setEuclideanFitnessEpsilon(icp_euclidean_epsilon_);
  icp.setMaximumIterations(icp_max_iterations_);
  icp.setMaxCorrespondenceDistance(icp_max_correspondence_dist_);

  // setup concave hull extractor
  PointCloud::Ptr scan_hull = boost::make_shared<PointCloud>();
  PointCloud::Ptr template_hull = boost::make_shared<PointCloud>();
  pcl::ConcaveHull<pcl::PointXYZ> concave_hull;
  concave_hull.setAlpha(concave_hull_alpha_);

  // extract concave hull for template cloud and scan
  concave_hull.setInputCloud(scan_isolated_);
  concave_hull.reconstruct(*scan_hull);
  // pcl::io::savePCDFileBinary("/home/nick/tmp/scan_hull.pcd", *scan_hull);

  icp.setInputSource(scan_hull);
  icp.setInputTarget(target_params_->template_cloud);
  icp.align(*scan_registered,
            utils::InvertTransform(T_LIDAR_TARGET_EST_).cast<float>());

  if (!icp.hasConverged()) {
    measurement_valid_ = false;
    if (show_measurements_) {
      std::cout << "ICP failed." << std::endl;
    }
    T_LIDAR_TARGET_OPT_ = T_LIDAR_TARGET_EST_;
  } else {
    measurement_valid_ = true;
    T_LIDAR_TARGET_OPT_ =
        utils::InvertTransform(icp.getFinalTransformation().cast<double>());
  }

  // transform keypoints from json using opt. transform and store
  Eigen::Vector4d keypoint_trans;
  keypoints_measured_->clear();
  for (Eigen::Vector3d keypoint : target_params_->keypoints_lidar) {
    keypoint_trans = T_LIDAR_TARGET_OPT_ * keypoint.homogeneous();
    keypoints_measured_->push_back(
        utils::EigenPointToPCL(keypoint_trans.hnormalized()));
  }
}

void DiamondCornersLidarExtractor::CheckMeasurementValid() {
  if(!measurement_valid_){
    return;
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  PointCloud::Ptr template_transformed = boost::make_shared<PointCloud>();
  pcl::transformPointCloud(*target_params_->template_cloud,
                           *template_transformed,
                           T_LIDAR_TARGET_OPT_.cast<float>());
  kd_tree.setInputCloud(template_transformed);
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it = scan_isolated_->begin();
       it != scan_isolated_->end(); ++it) {
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    int num_neighbors =
        kd_tree.radiusSearch(*it, allowable_keypoint_error_,
                             pointIdxRadiusSearch, pointRadiusSquaredDistance);
    if (num_neighbors == 0) {
      measurement_valid_ = false;
      return;
    }
  }
  measurement_valid_ = true;
}

} // namespace vicon_calibration
