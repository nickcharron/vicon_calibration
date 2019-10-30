#pragma once

#include "vicon_calibration/LidarExtractor.h"
#include "vicon_calibration/utils.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>

namespace vicon_calibration {

/**
 * @brief Class that inherits from pcl's icp with a function to access the
 * final correspondences
 */
template <typename PointSource, typename PointTarget, typename Scalar = float>
class IterativeClosestPointCustom
    : public pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar> {
public:
  pcl::CorrespondencesPtr getCorrespondencesPtr() {
    return this->correspondences_;
  }
};

class CylinderLidarExtractor : public LidarExtractor {
public:
  // Inherit base class constructors
  using LidarExtractor::LidarExtractor;

  ~CylinderLidarExtractor() override = default;

  /**
   * @brief Get the type of LidarExtractor
   * @return Returns type as one of LidarExtractor types specified in the enum
   * LidarExtractorType
   */
  LidarExtractorType GetType() const override {
    return LidarExtractorType::CYLINDER;
  };

  void ExtractKeypoints(Eigen::Matrix4d &T_LIDAR_TARGET_EST,
                        PointCloud::Ptr &cloud_in) override;

private:
  void CheckInputs();

  void CropScan();

  void RegisterScan();

  PointCloudColor::Ptr ColourPointCloud(PointCloud::Ptr &cloud, int r, int g,
                                        int b);

  void AddPointCloudToViewer(PointCloud::Ptr cloud, std::string cloud_name,
                             Eigen::Affine3d &T);

  void AddColouredPointCloudToViewer(PointCloudColor::Ptr cloud,
                                     std::string cloud_name,
                                     Eigen::Affine3d &T);

  void ShowFailedMeasurement();

  void ConfirmMeasurementKeyboardCallback(
      const pcl::visualization::KeyboardEvent &event, void *viewer_void);

  void ShowFinalTransformation();

  void SaveMeasurement();

  PointCloud::Ptr scan_in_;
  PointCloud::Ptr scan_cropped_;
  PointCloud::Ptr scan_best_points_; // points that are corresponding after icp
  Eigen::Matrix4d T_LIDAR_TARGET_EST_;
  pcl::visualization::PCLVisualizer::Ptr pcl_viewer_;
  bool test_registration_{true}; // Whether to use ICP to test that the target
                                 // template can converge to the scan correctly
  double max_keypoint_distance_{0.005}; // keypoints will only be the taken when
                                        // the correspondence distance with the
                                        // est. tgt. loc. is less than this
  double dist_acceptance_criteria_{0.025}; // acceptable error between estimated
                                          // target and registered target
  double icp_transform_epsilon_{1e-8};
  double icp_euclidean_epsilon_{1e-2};
  int icp_max_iterations_{80};
  double icp_max_correspondance_dist_{1};
  bool measurement_failed_{false}; // used for visualization only
  bool close_viewer_{false};
};

} // namespace vicon_calibration
