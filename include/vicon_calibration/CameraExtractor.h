#pragma once

#include "vicon_calibration/params.h"
#include <beam_calibration/CameraModel.h>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vicon_calibration {

/**
 * @brief Enum class for different types of camera extractors we can use
 */
enum class CameraExtractorType { CYLINDER = 0, DIAMOND };

/**
 * @brief Abstract class for CameraExtractor
 */
class CameraExtractor {
public:
  CameraExtractor(); // ADD INITIALIZATION OF POINTERS

  virtual ~CameraExtractor() = default;

  // alias for clarity
  using Ptr = std::shared_ptr<CameraExtractor>;

  /**
   * @brief Get the type of CameraExtractor
   * @return Returns type as one of CameraExtractor types specified in the enum
   * CameraExtractorType
   */
  virtual CameraExtractorType GetType() const = 0;

  void SetCameraParams(CameraParams &camera_params);

  void SetTargetParams(TargetParams &target_params);

  void SetShowMeasurements(bool &show_measurements);

  virtual void ExtractKeypoints(Eigen::Matrix4d &T_CAMERA_TARGET_EST,
                                cv::Mat &img_in) = 0;

  bool GetMeasurementValid();

  pcl::PointCloud<pcl::PointXY>::Ptr GetMeasurement();

protected:
  CameraParams camera_params_;
  TargetParams target_params_;
  bool crop_image_{true};
  pcl::PointCloud<pcl::PointXY>::Ptr keypoints_measured_;
  bool measurement_complete_{false};
  bool measurement_valid_{false};
  bool target_params_set_{false};
  bool camera_params_set_{false};
  bool show_measurements_{false};
  std::shared_ptr<beam_calibration::CameraModel> camera_model_;
};

} // namespace vicon_calibration
