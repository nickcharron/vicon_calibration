#pragma once

#include "vicon_calibration/measurement_extractors/LidarExtractor.h"
#include "vicon_calibration/utils.h"
#include <pcl/visualization/pcl_visualizer.h>

namespace vicon_calibration {

class CylinderLidarExtractor : public LidarExtractor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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

  void AddPointCloudToViewer(PointCloud::Ptr cloud,
                             const std::string &cloud_name,
                             const Eigen::Matrix4d &T);

  void AddColouredPointCloudToViewer(PointCloudColor::Ptr cloud,
                                     const std::string &cloud_name,
                                     const Eigen::Matrix4d &T);

  void ShowFailedMeasurement();

  void ConfirmMeasurementKeyboardCallback(
      const pcl::visualization::KeyboardEvent &event, void *viewer_void);

  void ShowFinalTransformation();

  void SaveMeasurement();

  PointCloud::Ptr scan_in_;
  PointCloud::Ptr scan_cropped_;
  PointCloud::Ptr scan_best_points_; // points that are corresponding after icp
  Eigen::MatrixXd T_LIDAR_TARGET_EST_ = Eigen::MatrixXd(4, 4);
  pcl::visualization::PCLVisualizer::Ptr pcl_viewer_;
  bool test_registration_{true}; // Whether to use ICP to test that the target
                                 // template can converge to the scan correctly
  double max_keypoint_distance_{0.005}; // keypoints will only be the taken when
                                        // the correspondence distance with the
                                        // est. tgt. loc. is less than this
  double dist_acceptance_criteria_{0.05}; // acceptable error between estimated
                                          // target and registered target
  double icp_transform_epsilon_{1e-8};
  double icp_euclidean_epsilon_{1e-2};
  int icp_max_iterations_{80};
  double icp_max_correspondence_dist_{1};
  bool measurement_failed_{false}; // used for visualization only
  bool close_viewer_{false};
};

} // namespace vicon_calibration
