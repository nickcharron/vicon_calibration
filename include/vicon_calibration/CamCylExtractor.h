#pragma once

#include <Eigen/Geometry>

#include <vector>

#include <beam_calibration/CameraModel.h>

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

  /**
   * @brief Configures camera model object
   * @param intrinsic_file intrinsic calibration file path for camera model
   */
  void ConfigureCameraModel(std::string intrinsic_file);

  /**
   * @brief Extracts cylinder from an image and calculate measurement
   * @param T_CAMERA_TARGET_EST transform from estimated target to camera frame
   * @param image image to extract cylinder from
   */
  void ExtractCylinder(Eigen::Affine3d T_CAMERA_TARGET_EST, cv::Mat &image);

  /**
   * @brief Sets cylinder dimensions
   * @param radius radius of the cylinder
   * @param height height of the cylinder
   */
  void SetCylinderDimension(double radius, double height);

  /**
   * @brief Sets offset of cropping the image
   * @param offset offset to add for the bounding box to crop the image
   */
  void SetOffset(double offset);

  /**
   * @brief Sets the parameters used to detect cylinder edges in the image
   * @param num_intersections minimum number of intersections to "detect" a line
   *        for probalistic hough line transform
   * @param min_length_percent percentage of the bounding box to define the
   *        minimum number of points to form a line
   * @param max_gap_percent percentage of the bounding box to define the maximum
   *        gap between two points to be considered as a line
   * @param canny_percent percentage to define the lower and upper thresholds
   *        used for canny detection
   */
  void SetEdgeDetectionParameters(int num_intersections,
                                  double min_length_percent,
                                  double max_gap_percent, double canny_percent);

  /**
   * @brief Sets show_measurement_
   */
  void SetShowMeasurement(bool show_measurement);
  /**
   * @brief Returns the results of cylinder extraction
   * @return a pair of Eigen vector 3d (u,v,angle) and boolean to indicate if
   *         the measurement is valid or not
   */
  std::pair<Eigen::Vector3d, bool> GetMeasurementInfo();

private:
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

  double radius_{0.0635};
  double height_{0.5};
  double offset_{0.15};

  Eigen::Affine3d T_CAMERA_TARGET_EST_;

  std::vector<Eigen::Vector4d> bounding_points_;
  std::pair<Eigen::Vector4d, Eigen::Vector4d> center_line_points_;
  std::vector<Eigen::Vector4d> top_cylinder_points_;
  std::vector<Eigen::Vector4d> bottom_cylinder_points_;

  // Minimum number of intersections for a line to be accepted
  // through Hough line transform
  int num_intersections_{300};
  // Percentage to determine minimum number of pixels required for a line to be
  // accepted as a line through Hough line transform
  double min_length_percent_{0.5};
  // Percentage to determine maximum gap between two pixels for a line to be
  // accepted as a line through Hough line transform
  double max_gap_percent_{0.075};

  // percentage to define the lower and upper thresholds used for canny
  // detection:
  //  1. if the pixel gradient is lower than the low threshold, it is rejected
  //     as an edge
  //  2. if the gradient is higher than upper threshold, it is accepted
  //  3. if the gradient is between the two thresholds, it is accepted only if
  //     it is connected to a pixel that is above the upper threshold
  double canny_percent_{0.1};

  std::shared_ptr<beam_calibration::CameraModel> camera_model_;

  bool measurement_valid_;
  bool measurement_complete_;
  Eigen::Vector3d measurement_;

  // flag used to allow user input for accepting measurement
  bool show_measurement_{false};
};

} // end namespace vicon_calibration
