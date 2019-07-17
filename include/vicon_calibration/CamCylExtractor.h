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
   * @param measurement_num measurement number identifier to display images
   */
  void ExtractCylinder(Eigen::Affine3d T_CAMERA_TARGET_EST, cv::Mat &image,
                       int measurement_num);

  /**
   * @brief Sets cylinder dimensions
   * @param radius radius of the cylinder
   * @param height height of the cylinder
   */
  void SetCylinderDimension(double radius, double height);

  /**
   * @brief Sets threshold of cropping the image
   * @param threshold threshold to add for the bounding box to crop the image
   */
  void SetThreshold(double threshold);

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
   * @brief Returns the results of cylinder extraction
   * @return a pair of Eigen vector 3d (u,v,angle) and boolean to indicate if
   * the measurement is valid or not
   */
  std::pair<Eigen::Vector3d, bool> GetMeasurementInfo();

private:
  /**
   * @brief Populates cylinder points used for defining a bounding box and detecting cylinder target in the image
   */
  void PopulateCylinderPoints();


  std::vector<Eigen::Vector4d> GetCylinderPoints(double radius, double height,
                                                 double threshold);

  std::pair<Eigen::Vector2d, Eigen::Vector2d> GetBoundingBox();

  std::pair<cv::Vec4i, cv::Vec4i> GetCylinderEdges();

  std::vector<cv::Vec4i> DetectLines(cv::Mat &orig_img, cv::Mat &cropped_img,
                                     int measurement_num, int min_line_length,
                                     int max_line_gap);

  cv::Mat CropImage(cv::Mat &image, Eigen::Vector2d min_vector,
                    Eigen::Vector2d max_vector);

  double Median(cv::Mat &channel);

  void DrawLine(cv::Mat &img, cv::Vec4i line, int r, int g, int b);

  cv::Mat ColorPixelsOnImage(cv::Mat &img, std::vector<Eigen::Vector2d> points,
                             int r, int g, int b, int radius);

  double CalcDistBetweenLines(cv::Vec4i line1, cv::Vec4i line2);

  double CalcDistBetweenLineAndPoint(Eigen::Vector2d line_p1,
                                     Eigen::Vector2d line_p2,
                                     Eigen::Vector2d p);

  Eigen::Vector2d CylinderPointToPixel(Eigen::Vector4d point);

  double radius_{0.0635};
  double height_{0.5};
  double threshold_{0.15};

  Eigen::Affine3d T_CAMERA_TARGET_EST_;

  std::vector<Eigen::Vector4d> bounding_points_;
  std::pair<Eigen::Vector4d, Eigen::Vector4d> center_line_points_;
  std::vector<Eigen::Vector4d> top_cylinder_points_;
  std::vector<Eigen::Vector4d> bottom_cylinder_points_;
  int num_intersections_{300};
  double min_length_percent_{0.5};
  double max_gap_percent_{0.075};
  /*
   * percentage to define the lower and upper thresholds used for canny
   * detection
   *    1. if the pixel gradient is lower than the low threshold, it is rejected
   *       as an edge
   *    2. if the gradient is higher than upper threshold, it is accepted
   *    3. if the gradient is between the two thresholds, it is accepted only if
   *       it is connected to a pixel that is above the upper threshold
   */
  double canny_percent_{0.1};

  std::shared_ptr<beam_calibration::CameraModel> camera_model_;

  bool measurement_valid_;
  bool measurement_complete_;
  Eigen::Vector3d measurement_;
};

} // end namespace vicon_calibration
