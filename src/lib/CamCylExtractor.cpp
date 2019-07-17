#include "vicon_calibration/CamCylExtractor.h"
#include "vicon_calibration/utils.hpp"

#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <thread>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

CamCylExtractor::CamCylExtractor() { PopulateCylinderPoints(); }

void CamCylExtractor::ConfigureCameraModel(std::string intrinsic_file) {
  camera_model_ = beam_calibration::CameraModel::LoadJSON(intrinsic_file);
}

void CamCylExtractor::SetCylinderDimension(double radius, double height) {
  radius_ = radius;
  height_ = height;

  PopulateCylinderPoints();
}

void CamCylExtractor::SetThreshold(double threshold) {
  threshold_ = threshold;

  if (threshold_ == 0) {
    std::cout << "[WARNING] Using threshold of 0 to crop the image."
              << std::endl;
  }

  // Re-populate points for cropping the image
  auto bottom_bounding_points = GetCylinderPoints(radius_, 0, -threshold_);
  bounding_points_ = GetCylinderPoints(radius_, height_, threshold_);
  bounding_points_.insert(bounding_points_.end(),
                          bottom_bounding_points.begin(),
                          bottom_bounding_points.end());
}

void CamCylExtractor::SetEdgeDetectionParameters(int num_intersections,
                                                 double min_length_percent,
                                                 double max_gap_percent,
                                                 double canny_percent) {
  num_intersections_ = num_intersections;
  min_length_percent_ = min_length_percent;
  max_gap_percent_ = max_gap_percent;
  canny_percent_ = canny_percent;
}

std::pair<Eigen::Vector3d, bool> CamCylExtractor::GetMeasurementInfo() {
  if (measurement_complete_) {
    return std::make_pair(measurement_, measurement_valid_);
  } else {
    throw std::runtime_error{"Measurement incomplete. Please run "
                             "ExtractCylinder() before getting the "
                             "measurement information."};
  }
}

void CamCylExtractor::PopulateCylinderPoints() {
  if (threshold_ == 0) {
    std::cout << "[WARNING] Using threshold of 0 to crop the image."
              << std::endl;
  }
  // Populate points for detecting edges
  center_line_points_ = std::make_pair<Eigen::Vector4d, Eigen::Vector4d>(
      Eigen::Vector4d(0, 0, 0, 1), Eigen::Vector4d(height_, 0, 0, 1));
  bottom_cylinder_points_ = GetCylinderPoints(radius_, 0, 0);
  top_cylinder_points_ = GetCylinderPoints(radius_, height_, 0);

  // Populate points for cropping the image
  auto bottom_bounding_points = GetCylinderPoints(radius_, 0, -threshold_);
  bounding_points_ = GetCylinderPoints(radius_, height_, threshold_);
  bounding_points_.insert(bounding_points_.end(),
                          bottom_bounding_points.begin(),
                          bottom_bounding_points.end());
}

std::vector<Eigen::Vector4d>
CamCylExtractor::GetCylinderPoints(double radius, double height,
                                   double threshold) {
  Eigen::Vector4d point(0, 0, 0, 1); // For cylinder used to detect edge lines
  std::vector<Eigen::Vector4d> points;

  double r = radius + threshold;

  for (double theta = 0; theta < 2 * M_PI; theta += M_PI / 12) {
    point(2) = r * std::cos(theta);

    point(1) = r * std::sin(theta);

    point(0) = height + threshold;
    points.push_back(point);
  }

  return points;
}

void CamCylExtractor::ExtractCylinder(Eigen::Affine3d T_CAMERA_TARGET_EST,
                                      cv::Mat &image, int measurement_num) {
  T_CAMERA_TARGET_EST_ = T_CAMERA_TARGET_EST;
  auto undistorted_img = camera_model_->UndistortImage(image);

  auto bounding_box = GetBoundingBox();

  auto cropped_image =
      CropImage(undistorted_img, bounding_box.first, bounding_box.second);

  int min_line_length =
      (bounding_box.second(1) - bounding_box.first(1)) * min_length_percent_;
  int max_gap =
      (bounding_box.second(1) - bounding_box.first(1)) * max_gap_percent_;

  auto lines = DetectLines(undistorted_img, cropped_image, measurement_num,
                           min_line_length, max_gap);

  auto edge_lines = GetCylinderEdges();

  std::vector<cv::Vec4i> bounding_lines;
  bounding_lines.push_back(
      cv::Vec4i(bounding_box.first(0), bounding_box.first(1),
                bounding_box.first(0), bounding_box.second(1)));
  bounding_lines.push_back(
      cv::Vec4i(bounding_box.second(0), bounding_box.first(1),
                bounding_box.second(0), bounding_box.second(1)));
  bounding_lines.push_back(
      cv::Vec4i(bounding_box.first(0), bounding_box.first(1),
                bounding_box.second(0), bounding_box.first(1)));
  bounding_lines.push_back(
      cv::Vec4i(bounding_box.first(0), bounding_box.second(1),
                bounding_box.second(0), bounding_box.second(1)));

  cv::Vec4i cylinder_line_1, cylinder_line_2;
  std::map<double, cv::Vec4i> dist_line_map_1, dist_line_map_2;

  double d1, d2, dist_to_bounding_box;
  Eigen::Vector2d end_point;
  bool is_bounding_line;
  for (auto line : lines) {
    is_bounding_line = false;
    // If an end point of the line is close to bounding box, ignore that line
    for (auto bounding_line : bounding_lines) {
      if (CalcDistBetweenLines(bounding_line, line) <= 5) {
        is_bounding_line = true;
        break;
      }
    }

    if (is_bounding_line)
      continue;
    d1 = CalcDistBetweenLines(edge_lines.first, line);
    d2 = CalcDistBetweenLines(edge_lines.second, line);
    if (d1 < d2) {
      dist_line_map_1.insert(std::pair<double, cv::Vec4i>(d1, line));
    } else {
      dist_line_map_2.insert(std::pair<double, cv::Vec4i>(d2, line));
    }
  }

  cylinder_line_1 = dist_line_map_1.begin()->second;
  cylinder_line_2 = dist_line_map_2.begin()->second;
  std::cout << "first edge detected: " << cylinder_line_1 << std::endl;
  DrawLine(cropped_image, cylinder_line_1, 0, 0, 255);
  std::cout << "second edge detected: " << cylinder_line_2 << std::endl;
  DrawLine(cropped_image, cylinder_line_2, 0, 0, 255);

  cv::namedWindow("Hough transform", cv::WINDOW_NORMAL);
  cv::resizeWindow("Hough transform", cropped_image.cols / 2,
                   cropped_image.rows / 2);

  cv::Vec4i measured_center_line((cylinder_line_1[0] + cylinder_line_2[0]) / 2,
                                 (cylinder_line_1[1] + cylinder_line_2[1]) / 2,
                                 (cylinder_line_1[2] + cylinder_line_2[2]) / 2,
                                 (cylinder_line_1[3] + cylinder_line_2[3]) / 2);
  double angle;
  if (measured_center_line[0] == measured_center_line[2]) {
    // the center line is horizontal
    angle = M_PI / 2;
  } else if (measured_center_line[1] == measured_center_line[3]) {
    // the center line is vertical
    angle = 0;
  } else {
    angle = std::acos((measured_center_line[2] - measured_center_line[0]) /
                      (measured_center_line[3] - measured_center_line[1]));
  }
  std::cout << "measurement: " << measured_center_line << std::endl;
  DrawLine(cropped_image, measured_center_line, 255, 0, 0);

  measurement_ = Eigen::Vector3d(
      (measured_center_line[0] + measured_center_line[2]) / 2,
      (measured_center_line[1] + measured_center_line[3]) / 2, angle);
  ColorPixelsOnImage(cropped_image,
                     std::vector<Eigen::Vector2d>{
                         Eigen::Vector2d(measurement_(0), measurement_(1))},
                     0, 0, 0, 4);
  measurement_complete_ = true;
  measurement_valid_ = true;

  imshow("Hough transform", cropped_image);

  auto key = cv::waitKey();
  while (key != 121) {
    key = cv::waitKey();
  }
  cv::destroyAllWindows();
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> CamCylExtractor::GetBoundingBox() {
  std::vector<double> u, v;

  std::vector<Eigen::Vector2d> pixels_to_color;

  for (auto point : bounding_points_) {
    auto pixel = CylinderPointToPixel(point);

    if (camera_model_->PixelInImage(pixel)) {
      u.push_back(pixel(0));
      v.push_back(pixel(1));
      pixels_to_color.push_back(pixel);
    }
  }

  if (u.empty() || v.empty()) {
    throw std::runtime_error{
        "Can't find minimum and maximum pixels on the image"};
  }
  const auto min_max_u = std::minmax_element(u.begin(), u.end());
  const auto min_max_v = std::minmax_element(v.begin(), v.end());

  Eigen::Vector2d min_vec(*min_max_u.first, *min_max_v.first);
  Eigen::Vector2d max_vec(*min_max_u.second, *min_max_v.second);

  return std::pair<Eigen::Vector2d, Eigen::Vector2d>(min_vec, max_vec);
}

std::pair<cv::Vec4i, cv::Vec4i> CamCylExtractor::GetCylinderEdges() {
  cv::Vec4i first_corner_line, second_corner_line;

  auto center_line_pixel_buttom =
      CylinderPointToPixel(center_line_points_.first);
  auto center_line_pixel_top = CylinderPointToPixel(center_line_points_.second);

  Eigen::Vector2d top_max_pixel, top_second_max_pixel, pixel;
  double max_dist = 0, second_max_dist = 0, dist;

  for (auto point : top_cylinder_points_) {
    pixel = CylinderPointToPixel(point);
    dist = CalcDistBetweenLineAndPoint(center_line_pixel_top,
                                       center_line_pixel_buttom, pixel);
    if (dist > max_dist) {
      max_dist = dist;
      top_max_pixel = pixel;
    } else if (dist == max_dist || dist > second_max_dist) {
      second_max_dist = dist;
      top_second_max_pixel = pixel;
    }
  }

  Eigen::Vector2d bottom_max_pixel, bottom_second_max_pixel;
  max_dist = 0;
  second_max_dist = 0;
  for (auto point : bottom_cylinder_points_) {
    pixel = CylinderPointToPixel(point);
    dist = CalcDistBetweenLineAndPoint(center_line_pixel_top,
                                       center_line_pixel_buttom, pixel);
    if (dist > max_dist) {
      max_dist = dist;
      bottom_max_pixel = pixel;
    } else if (dist == max_dist || dist > second_max_dist) {
      second_max_dist = dist;
      bottom_second_max_pixel = pixel;
    }
  }

  Eigen::Vector3d angles = T_CAMERA_TARGET_EST_.rotation().eulerAngles(0, 1, 2);
  std::cout << "angles: " << angles << std::endl;

  auto r = round(angles(0) / (M_PI / 2)) * (M_PI / 2);
  r = r * 180 / M_PI;
  r = (int)r % 360;
  r = r < 0 ? r + 360 : r;
  std::cout << "roll: " << r << std::endl;

  if (r == 0 || r == 180 || r == 2 * 360) {
    if ((top_max_pixel(0) < top_second_max_pixel(0) &&
         bottom_max_pixel(0) < bottom_second_max_pixel(0)) ||
        (top_max_pixel(0) > top_second_max_pixel(0) &&
         bottom_max_pixel(0) > bottom_second_max_pixel(0))) {
      first_corner_line[0] = top_max_pixel(0);
      first_corner_line[1] = top_max_pixel(1);
      first_corner_line[2] = bottom_max_pixel(0);
      first_corner_line[3] = bottom_max_pixel(1);

      second_corner_line[0] = top_second_max_pixel(0);
      second_corner_line[1] = top_second_max_pixel(1);
      second_corner_line[2] = bottom_second_max_pixel(0);
      second_corner_line[3] = bottom_second_max_pixel(1);
    } else {
      first_corner_line[0] = top_max_pixel(0);
      first_corner_line[1] = top_max_pixel(1);
      first_corner_line[2] = bottom_second_max_pixel(0);
      first_corner_line[3] = bottom_second_max_pixel(1);

      second_corner_line[0] = top_second_max_pixel(0);
      second_corner_line[1] = top_second_max_pixel(1);
      second_corner_line[2] = bottom_max_pixel(0);
      second_corner_line[3] = bottom_max_pixel(1);
    }
  } else if (r == 90 || r == 270) {
    if ((top_max_pixel(1) < top_second_max_pixel(1) &&
         bottom_max_pixel(1) < bottom_second_max_pixel(1)) ||
        (top_max_pixel(1) > top_second_max_pixel(1) &&
         bottom_max_pixel(1) > bottom_second_max_pixel(1))) {
      first_corner_line[0] = top_max_pixel(0);
      first_corner_line[1] = top_max_pixel(1);
      first_corner_line[2] = bottom_max_pixel(0);
      first_corner_line[3] = bottom_max_pixel(1);

      second_corner_line[0] = top_second_max_pixel(0);
      second_corner_line[1] = top_second_max_pixel(1);
      second_corner_line[2] = bottom_second_max_pixel(0);
      second_corner_line[3] = bottom_second_max_pixel(1);
    } else {
      first_corner_line[0] = top_max_pixel(0);
      first_corner_line[1] = top_max_pixel(1);
      first_corner_line[2] = bottom_second_max_pixel(0);
      first_corner_line[3] = bottom_second_max_pixel(1);

      second_corner_line[0] = top_second_max_pixel(0);
      second_corner_line[1] = top_second_max_pixel(1);
      second_corner_line[2] = bottom_max_pixel(0);
      second_corner_line[3] = bottom_max_pixel(1);
    }
  }

  std::cout << "first edge estimated: " << first_corner_line << std::endl;
  std::cout << "second edge estimated: " << second_corner_line << std::endl;

  return std::pair<cv::Vec4i, cv::Vec4i>(first_corner_line, second_corner_line);
}

std::vector<cv::Vec4i> CamCylExtractor::DetectLines(cv::Mat &orig_img,
                                                    cv::Mat &cropped_img,
                                                    int measurement_num,
                                                    int min_line_length,
                                                    int max_line_gap) {
  /// Load the image
  if (!orig_img.data) {
    throw std::runtime_error{"No image data"};
  }

  cv::Mat gray_img, cropped_gray_img, edge_detected_img;

  /// Convert the image to grayscale
  cv::cvtColor(orig_img, gray_img, CV_BGR2GRAY);

  auto v = Median(gray_img);

  int lower_threshold = (int)std::max(0.0, (1.0 - canny_percent_) * v);
  int upper_threshold = (int)std::min(255.0, (1.0 + canny_percent_) * v);

  cv::cvtColor(cropped_img, cropped_gray_img, CV_BGR2GRAY);
  /// Reduce noise with a kernel 3x3
  cv::blur(cropped_gray_img, edge_detected_img, cv::Size(3, 3));

  /// Canny detector
  cv::Canny(edge_detected_img, edge_detected_img, lower_threshold,
            upper_threshold, 3);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3),
                                             cv::Point(0, 0));
  cv::dilate(edge_detected_img, edge_detected_img, kernel);
  cv::namedWindow("canny image", cv::WINDOW_NORMAL);
  cv::resizeWindow("canny image", edge_detected_img.cols / 2,
                   edge_detected_img.rows / 2);
  cv::imshow("canny image", edge_detected_img);
  // Hough line transform
  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(edge_detected_img, lines, 1, CV_PI / 180, num_intersections_,
                  min_line_length, max_line_gap);

  return lines;
}

cv::Mat CamCylExtractor::CropImage(cv::Mat &image, Eigen::Vector2d min_vector,
                                   Eigen::Vector2d max_vector) {
  if (!camera_model_->PixelInImage(min_vector) ||
      !camera_model_->PixelInImage(max_vector)) {
    return image;
  }
  cv::Mat bounded_img(image.rows, image.cols, image.depth());

  double width = max_vector(0) - min_vector(0);
  double height = max_vector(1) - min_vector(1);

  cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, image.depth());
  mask(cv::Rect(min_vector(0), min_vector(1), width, height)) = 1;
  image.copyTo(bounded_img, mask);

  return bounded_img;
}

double CamCylExtractor::Median(cv::Mat &channel) {
  int depth;

  if (channel.depth() == CV_8U) {
    depth = 8;
  } else if (channel.depth() == CV_16U) {
    depth = 16;
  }

  double m = (channel.rows * channel.cols) / 2;
  int bin = 0;
  double med = -1.0;

  int hist_size = pow(2, depth);
  float range[] = {0, (float)hist_size};
  const float *hist_range = {range};
  bool uniform = true;
  bool accumulate = false;
  cv::Mat hist;
  cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range,
               uniform, accumulate);

  for (int i = 0; i < hist_size && med < 0.0; ++i) {
    bin += cvRound(hist.at<float>(i));
    if (bin > m && med < 0.0)
      med = i;
  }

  return med;
}

void CamCylExtractor::DrawLine(cv::Mat &img, cv::Vec4i line, int r, int g,
                               int b) {
  cv::line(img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]),
           cv::Scalar(b, g, r), 3, CV_AA);
}

cv::Mat CamCylExtractor::ColorPixelsOnImage(cv::Mat &img,
                                            std::vector<Eigen::Vector2d> points,
                                            int r, int g, int b, int radius) {
  for (auto pixel : points) {
    if (camera_model_->PixelInImage(pixel)) {
      cv::circle(img, cv::Point(pixel(0), pixel(1)), radius,
                 cv::Scalar(b, g, r), cv::FILLED, cv::LINE_8);
    }
  }
  return img;
}

double CamCylExtractor::CalcDistBetweenLines(cv::Vec4i line1, cv::Vec4i line2) {
  Eigen::Vector2d line_p1(line1[0], line1[1]), line_p2(line1[2], line1[3]),
      line_p3(line2[0], line2[1]), line_p4(line2[2], line2[3]);

  auto d1 = CalcDistBetweenLineAndPoint(line_p1, line_p2, line_p3);
  auto d2 = CalcDistBetweenLineAndPoint(line_p1, line_p2, line_p4);

  return (d1 + d2) / 2;
}

double CamCylExtractor::CalcDistBetweenLineAndPoint(Eigen::Vector2d line_p1,
                                                    Eigen::Vector2d line_p2,
                                                    Eigen::Vector2d p) {
  double x1 = line_p1(0), y1 = line_p1(1), x2 = line_p2(0), y2 = line_p2(1),
         px = p(0), py = p(1);

  return abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) /
         sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1));
}

Eigen::Vector2d CamCylExtractor::CylinderPointToPixel(Eigen::Vector4d point) {
  Eigen::Vector3d transformed_point;
  Eigen::Vector4d homographic_point;

  homographic_point = T_CAMERA_TARGET_EST_.matrix() * point;

  transformed_point << homographic_point[0] / homographic_point[3],
      homographic_point[1] / homographic_point[3],
      homographic_point[2] / homographic_point[3];

  return camera_model_->ProjectUndistortedPoint(transformed_point);
}

} // end namespace vicon_calibration
