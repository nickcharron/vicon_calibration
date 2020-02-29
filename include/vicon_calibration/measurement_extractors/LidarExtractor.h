#pragma once

#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace vicon_calibration {

/**
 * @brief Enum class for different types of camera extractors we can use
 */
enum class LidarExtractorType { CYLINDER = 0, DIAMOND };

/**
 * @brief Class that inherits from pcl's icp with a function to access the
 * final correspondences
 */
template <typename PointSource, typename PointTarget, typename Scalar = float>
class IterativeClosestPointCustom
    : public pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  pcl::CorrespondencesPtr getCorrespondencesPtr() {
    return this->correspondences_;
  }
};

/**
 * @brief Abstract class for LidarExtractor
 */
class LidarExtractor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LidarExtractor(); // ADD INITIALIZATION OF POINTERS

  virtual ~LidarExtractor() = default;

  // alias for clarity
  using Ptr = std::shared_ptr<LidarExtractor>;

  /**
   * @brief Get the type of LidarExtractor
   * @return Returns type as one of LidarExtractor types specified in the enum
   * LidarExtractorType
   */
  virtual LidarExtractorType GetType() const = 0;

  void
  SetLidarParams(std::shared_ptr<vicon_calibration::LidarParams> &lidar_params);

  void SetTargetParams(
      std::shared_ptr<vicon_calibration::TargetParams> &target_params);

  void SetShowMeasurements(const bool &show_measurements);

  bool GetShowMeasurements();

  void ProcessMeasurement(const Eigen::Matrix4d &T_LIDAR_TARGET_EST,
                          const PointCloud::Ptr &cloud_in);

  bool GetMeasurementValid();

  PointCloud::Ptr GetMeasurement();

protected:
  // this is what we will need to override in the derived class
  virtual void GetKeypoints() = 0;

  void CheckInputs();

  void CropScan();

  void AddColouredPointCloudToViewer(const PointCloudColor::Ptr &cloud,
                                     const std::string &cloud_name,
                                     boost::optional<Eigen::MatrixXd &> T,
                                     int point_size = 1);

  void AddPointCloudToViewer(const PointCloud::Ptr &cloud,
                             const std::string &cloud_name,
                             const Eigen::Matrix4d &T,
                             int point_size = 1);

  void ConfirmMeasurementKeyboardCallback(
      const pcl::visualization::KeyboardEvent &event, void *viewer_void);

  void ShowFailedMeasurement();

  void ShowFinalTransformation();

  void LoadConfig();

  // member variables
  PointCloud::Ptr scan_in_;
  PointCloud::Ptr scan_cropped_;
  Eigen::MatrixXd T_LIDAR_TARGET_EST_ = Eigen::MatrixXd(4, 4);
  pcl::visualization::PCLVisualizer::Ptr pcl_viewer_;
  PointCloud::Ptr keypoints_measured_;
  std::string template_cloud_path_;
  bool measurement_valid_{false};
  bool measurement_complete_{false};
  bool target_params_set_{false};
  bool lidar_params_set_{false};
  bool close_viewer_{false};
  bool measurement_failed_{false}; // used for visualization only
  bool white_cloud_on_ = true;
  bool blue_cloud_on_ = true;
  bool green_cloud_on_ = true;
  bool viewer_key_down_ = false;
  PointCloudColor::Ptr blue_cloud_, green_cloud_;
  PointCloud::Ptr white_cloud_;

  // params:
  std::shared_ptr<vicon_calibration::LidarParams> lidar_params_;
  std::shared_ptr<vicon_calibration::TargetParams> target_params_;
  bool crop_scan_{true};
  bool show_measurements_{false};
  double max_keypoint_distance_{0.005}; // keypoints will only be the taken when
                                        // the correspondence distance with the
                                        // est. tgt. loc. is less than this
  double dist_acceptance_criteria_{0.05}; // acceptable error between estimated
                                          // target and registered target
  double icp_transform_epsilon_{1e-8};
  double icp_euclidean_epsilon_{1e-2};
  int icp_max_iterations_{80};
  double icp_max_correspondence_dist_{1};
};

} // namespace vicon_calibration
