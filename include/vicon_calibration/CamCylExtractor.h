#pragma once

#include "vicon_calibration/params.h"
#include <Eigen/Geometry>
#include <beam_calibration/CameraModel.h>
#include <vector>

namespace vicon_calibration {

/**
 * @brief class for extracting cylinder measurements from images
 */
class CamCylExtractor {
public:
  // Constructor
  CamCylExtractor();

  // Destructor
  ~CamCylExtractor() = default;

  void SetTargetParams(vicon_calibration::CylinderTgtParams &params);

  void
  SetImageProcessingParams(vicon_calibration::ImageProcessingParams &params);

  void SetCameraParams(vicon_calibration::CameraParams &params);

  /**
   * @brief Extracts cylinder from an image and calculate measurement
   * @param T_CAMERA_TARGET_EST transform from estimated target to camera frame
   * @param image image to extract cylinder from
   */
  void ExtractCylinder(Eigen::Affine3d T_CAMERA_TARGET_EST, cv::Mat &image);

  /**
   * @brief Returns the results of cylinder extraction
   * @return a pair of Eigen vector 3d (u,v,angle) and boolean to indicate if
   *         the measurement is valid or not
   */
  std::pair<Eigen::Vector3d, bool> GetMeasurementInfo();

  /**
   * @brief Gets calculated distance and rotation errors after extraction
   * @return a pair of distance error and rotation error
   */
  std::pair<double, double> GetErrors();

private:
  void SetDefaultParameters();

  /**
   * @brief Configures camera model object
   * @param intrinsic_file intrinsic calibration file path for camera model
   */
  void SetCameraModel(std::string intrinsic_file);

  /**
   * @brief Populates cylinder points used for defining a bounding box and
   *        detecting cylinder target in the image
   */
  void PopulateCylinderPoints();

  /**
   * @brief Calculates points in a circle
   * @param radius radius of the circle
   * @param height distance along the cylinder where the circle is located
   * @param offset offset to add to radius and height
   * @return a vector of Vector4d points in homogeneous form
   */
  std::vector<Eigen::Vector4d> GetCylinderPoints(double radius, double height,
                                                 double offset);

  /**
   * @brief Gets a bounding box around the cylinder target in the image using
   *        estimated transformation
   * @return a pair of Vector2d points indicating min and max points of
   *         the bounding box
   */
  std::pair<Eigen::Vector2d, Eigen::Vector2d> GetBoundingBox();

  /**
   * @brief Determines estimated edge lines of cylinder target in the image
   * @return a pair of Vec4i estimated edge lines
   */
  std::pair<cv::Vec4i, cv::Vec4i> GetEstimatedEdges();

  /**
   * @brief Detects lines in the image using Canny Detection and probalistic
   *        Hough line transform
   * @param orig_image original image used to determine the median pixel
   * intensity
   * @param cropped_image image with black pixels outside the bounding box to
   *        perform edge detection
   * @param min_line_length minimum number of points to form a line
   * @param max_line_gap maximum gap between two points to be accepted as a line
   * @return a vector of Vec4i detected lines
   */
  std::vector<cv::Vec4i> DetectLines(cv::Mat &orig_image,
                                     cv::Mat &cropped_image,
                                     int min_line_length, int max_line_gap);

  /**
   * @brief Calculates measurement using the center line defined by 2 edge lines
   * @param edge_line1
   * @param edge_line2
   * @return Vector3d of the calculated measurement
   */
  Eigen::Vector3d CalcMeasurement(cv::Vec4i edge_line1, cv::Vec4i edge_line2);

  /**
   * @brief Colours the pixels outside the bounding box black
   * @param image image to crop
   * @param min_vector minimum point defining the bounding box
   * @param max_vector maximum point defining the bounding box
   * @return cropped image
   */
  cv::Mat CropImage(cv::Mat &image, Eigen::Vector2d min_vector,
                    Eigen::Vector2d max_vector);

  /**
   * @brief Calculates the median pixel intensity
   * @param image image to calculate median of
   * @return median pixel intensity value
   */
  double Median(cv::Mat &image);

  /**
   * @brief Draws a line on an image with specified color
   * @param image image to draw a line on
   * @param line Vec4i of end points of the line
   * @param r red value to color the line
   * @param g green value to color the line
   * @param b blue value to color the line
   */
  void DrawLine(cv::Mat &image, cv::Vec4i line, int r, int g, int b);

  /**
   * @brief Colors pixels on an image with specified color
   * @param image image to color pixels on
   * @param vector of 2d points to color
   * @param r red value to color the pixels
   * @param g green value to color the pixels
   * @param b blue value to color the pixels
   * @param radius radius of coloring area around the pixels
   */
  cv::Mat ColorPixelsOnImage(cv::Mat &image,
                             std::vector<Eigen::Vector2d> points, int r, int g,
                             int b, int radius);

  /**
   * @brief Calculates distance between two lines
   * @param line1 first line
   * @param line2 second line
   * @return double distance between two lines
   */
  double CalcDistBetweenLines(cv::Vec4i line1, cv::Vec4i line2);

  /**
   * @brief Calculates distance between a line and a point
   * @param line_p1 first end point of a line
   * @param line_p2 second end point of a line
   * @param p a point
   * @return double distance between the line and the point
   */
  double CalcDistBetweenLineAndPoint(Eigen::Vector2d line_p1,
                                     Eigen::Vector2d line_p2,
                                     Eigen::Vector2d p);

  /**
   * @brief Transforms and projects a homogeneous cylinder point to 2d pixel
   * @param point point to project
   * @return Vector2d pixel that has been projected in camera frame
   */
  Eigen::Vector2d CylinderPointToPixel(Eigen::Vector4d point);

  Eigen::Affine3d T_CAMERA_TARGET_EST_;
  vicon_calibration::ImageProcessingParams image_processing_params_;
  vicon_calibration::CylinderTgtParams target_params_;
  vicon_calibration::CameraParams camera_params_;
  bool target_params_set_, camera_params_set_, measurement_valid_,
  measurement_complete_;
  double dist_err_, rot_err_;
  Eigen::Vector3d measurement_;

  std::vector<Eigen::Vector4d> bounding_points_;
  std::pair<Eigen::Vector4d, Eigen::Vector4d> center_line_points_;
  std::vector<Eigen::Vector4d> top_cylinder_points_;
  std::vector<Eigen::Vector4d> bottom_cylinder_points_;

  std::shared_ptr<beam_calibration::CameraModel> camera_model_;

};

} // end namespace vicon_calibration
