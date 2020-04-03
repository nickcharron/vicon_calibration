#include "vicon_calibration/measurement_extractors/DiamondLidarExtractor.h"
#include <pcl/surface/concave_hull.h>

namespace vicon_calibration {

void DiamondLidarExtractor::GetKeypoints() {
  measurement_failed_ = false;

  // setup icp
  PointCloud::Ptr scan_registered = boost::make_shared<PointCloud>();
  IterativeClosestPointCustom<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setTransformationEpsilon(icp_transform_epsilon_);
  icp.setEuclideanFitnessEpsilon(icp_euclidean_epsilon_);
  icp.setMaximumIterations(icp_max_iterations_);
  icp.setMaxCorrespondenceDistance(icp_max_correspondence_dist_);

  // setup concave hull extractor
  PointCloud::Ptr scan_hull =
      boost::make_shared<PointCloud>();
  PointCloud::Ptr template_hull =
      boost::make_shared<PointCloud>();
  pcl::ConcaveHull<pcl::PointXYZ> concave_hull;
  concave_hull.setAlpha(concave_hull_alpha_);


  // extract concave hull for template cloud and scan
  concave_hull.setInputCloud(scan_isolated_);
  concave_hull.reconstruct(*scan_hull);
  concave_hull.setInputCloud(target_params_->template_cloud);
  concave_hull.reconstruct(*template_hull);

  icp.setInputSource(scan_hull);
  icp.setInputTarget(template_hull);
  icp.align(*scan_registered, utils::InvertTransform(T_LIDAR_TARGET_EST_).cast<float>());

  if (!icp.hasConverged()) {
    measurement_failed_ = true;
    measurement_valid_ = false;
    measurement_complete_ = true;
    if (show_measurements_) {
      std::cout << "ICP failed. Displaying cropped scan." << std::endl;
      boost::shared_ptr<PointCloudColor>
          scan_isolated_coloured;
      scan_isolated_coloured =
          boost::make_shared<PointCloudColor>();
      Eigen::MatrixXd T_identity = Eigen::MatrixXd(4,4);
      T_identity.setIdentity();
      scan_isolated_coloured = utils::ColorPointCloud(scan_isolated_, 255, 0, 0);
      this->AddColouredPointCloudToViewer(scan_isolated_coloured,
                                          "red_cloud", T_identity);
      this->AddPointCloudToViewer(scan_in_, "white_cloud", T_identity);
      this->ShowFailedMeasurement();
      pcl_viewer_->resetStoppedFlag();
    }
    return;
  }

  Eigen::MatrixXd T_LIDAR_TARGET_OPT = Eigen::MatrixXd(4,4);
  T_LIDAR_TARGET_OPT =
      utils::InvertTransform(icp.getFinalTransformation().cast<double>());
  measurement_valid_ = true;

  // TODO: create error metric that depends on distance to sensor

  // transform keypoints from json using opt. transform and store
  Eigen::Vector4d keypoint_homo;
  Eigen::Vector4d keypoint_trans_homo;
  Eigen::Vector3d keypoint_trans;
  keypoints_measured_->clear();
  for (Eigen::Vector3d keypoint : target_params_->keypoints_lidar){
    keypoint_homo = utils::PointToHomoPoint(keypoint);
    keypoint_trans_homo = T_LIDAR_TARGET_OPT * keypoint_homo;
    keypoint_trans = utils::HomoPointToPoint(keypoint_trans_homo);
    keypoints_measured_->push_back(utils::EigenPointToPCL(keypoint_trans));
  }

  if (show_measurements_) {
    if (!measurement_valid_) {
      std::cout << "Measurement Invalid\n"
                << "Showing detected keypoints in Yellow\n";
    } else {
      std::cout << "Measurement Valid\n"
                << "Showing detected keypoints in Yellow\n";
    }

    // add estimated template cloud
    PointCloudColor::Ptr estimated_template_cloud =
        utils::ColorPointCloud(target_params_->template_cloud, 0, 0, 255);
    pcl::transformPointCloud(*estimated_template_cloud,
                             *estimated_template_cloud,
                             T_LIDAR_TARGET_EST_.cast<float>());
    this->AddColouredPointCloudToViewer(estimated_template_cloud,
                                        "blue_cloud",
                                        T_LIDAR_TARGET_EST_);

    // add measured template cloud
    PointCloudColor::Ptr measured_template_cloud =
        utils::ColorPointCloud(target_params_->template_cloud, 0, 255, 0);
    pcl::transformPointCloud(*measured_template_cloud, *measured_template_cloud,
                             T_LIDAR_TARGET_OPT.cast<float>());
    this->AddColouredPointCloudToViewer(
        measured_template_cloud, "green_cloud", T_LIDAR_TARGET_OPT);

    // add keypoints
    PointCloudColor::Ptr measured_keypoints =
        utils::ColorPointCloud(keypoints_measured_, 255, 255, 0);
    this->AddColouredPointCloudToViewer(
        measured_keypoints, "keypoints", T_LIDAR_TARGET_OPT, 5);

    Eigen::Matrix4d T_identity;
    T_identity.setIdentity();
    this->AddPointCloudToViewer(scan_isolated_, "white_cloud", T_identity);
    this->ShowFinalTransformation();
  }
}

} // namespace vicon_calibration
