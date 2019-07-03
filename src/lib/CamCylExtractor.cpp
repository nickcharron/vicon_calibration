#include "vicon_calibration/CamCylExtractor.h"
#include "vicon_calibration/utils.hpp"

#include <cmath>
#include <thread>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

CamCylExtractor::CamCylExtractor() { PopulateCylinderPoints(); }

void CamCylExtractor::ConfigureCameraModel(std::string intrinsic_file) {
  camera_model_ = beam_calibration::CameraModel::LoadJSON(intrinsic_file);
  distorted_intrinsics_ = camera_model_->GetIntrinsics();
  undistorted_intrinsics_[0] = 520.6433974864649;
  undistorted_intrinsics_[1] = 520.8029966847124;
  undistorted_intrinsics_[2] = 868.1109733699251;
  undistorted_intrinsics_[3] = 861.7608982000884;
}

void CamCylExtractor::SetCylinderDimension(double radius, double height) {
  radius_ = radius;
  height_ = height;

  PopulateCylinderPoints();
}

void CamCylExtractor::SetThreshold(double threshold) {
  threshold_ = threshold;
  PopulateCylinderPoints();
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
  // Clear all points before populating again
  cylinder_points_.clear();
  center_line_points_.clear();

  Eigen::Vector4d center_line_min_point(-threshold, 0, 0, 1);
  Eigen::Vector4d center_line_max_point(height_ + threshold_, 0, 0, 1);

  center_line_points_.push_back(center_line_min_point);
  center_line_points_.push_back(center_line_max_point);

  Eigen::Vector4d point(0, 0, 0, 1);
  double r = radius_ + threshold_;

  for (double theta = 0; theta < 2 * M_PI; theta += M_PI / 12) {
    point(2) = r * std::cos(theta);

    point(1) = r * std::sin(theta);

    point(0) = -threshold_;
    cylinder_points_.push_back(point);

    point(0) = height_ + threshold_;
    cylinder_points_.push_back(point);
  }
}

void CamCylExtractor::ExtractCylinder(Eigen::Affine3d T_CAMERA_TARGET_EST,
                                      cv::Mat image, int measurement_num) {
  auto undistorted_img = camera_model_->UndistortImage(image);

  auto min_max_vectors = GetBoundingBox(T_CAMERA_TARGET_EST);

  auto cropped_image =
      CropImage(undistorted_img, min_max_vectors[0], min_max_vectors[1]);

  if (cropped_image.depth() == CV_8U) {
    std::cout << "unsigned char image" << std::endl;
  }

  auto lines = DetectLines(cropped_image, measurement_num);

  double first_line_slope, second_line_slope;
  cv::Vec4i first_line, second_line;
  /*
    for (uint8_t i = 0; i < lines.size() - 1; i++) {
      first_line = lines[i];
      first_line_slope =
          (first_line[3] - first_line[1]) / (first_line[2] - first_line[0]);
      for (uint8_t j = i + 1; j < lines.size(); j++) {
        second_line = lines[j];
        second_line_slope =
            (second_line[3] - second_line[1]) / (second_line[2] -
    second_line[0]);
      }
    }
  */

  for (auto l : lines) {
    line(cropped_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
         cv::Scalar(0, 0, 255), 3, CV_AA);
  }

  imshow("Hough transform", cropped_image);

  auto key = cv::waitKey();
  while (key != 121) {
    key = cv::waitKey();
  }
  cv::destroyAllWindows();

  measurement_complete_ = true;
  measurement_valid_ = true;
}

std::vector<Eigen::Vector2d>
CamCylExtractor::GetBoundingBox(Eigen::Affine3d T_CAMERA_TARGET_EST) {
  Eigen::Vector3d transformed_point;
  Eigen::Vector4d homographic_form;

  Eigen::Vector3d transformed_center_point;
  Eigen::Vector4d homographic_center_point;

  std::vector<double> u, v;

  projected_pixels_.clear();

  camera_model_->SetIntrinsics(undistorted_intrinsics_);

  for (int i = 0; i < center_line_points_.size(); i++) {
    homographic_center_point =
        T_CAMERA_TARGET_EST.matrix() * center_line_points_[i];

    transformed_center_point
        << homographic_center_point[0] / homographic_center_point[3],
        homographic_center_point[1] / homographic_center_point[3],
        homographic_center_point[2] / homographic_center_point[3];

    audo center_line_pixel =
        camera_model_->ProjectUndistortedPoint(transformed_center_point);
    center_line_pixels_.push_back(center_line_pixel);
  }

  for (int i = 0; i < cylinder_points_.size(); i++) {
    homographic_point = T_CAMERA_TARGET_EST.matrix() * cylinder_points_[i];

    transformed_point << homographic_point[0] / homographic_point[3],
        homographic_point[1] / homographic_point[3],
        homographic_point[2] / homographic_point[3];

    auto pixel = camera_model_->ProjectUndistortedPoint(transformed_point);

    if (camera_model_->PixelInImage(pixel)) {
      u.push_back(pixel(0));
      v.push_back(pixel(1));
      projected_pixels_.push_back(pixel);
    }
  }

  camera_model_->SetIntrinsics(distorted_intrinsics_);

  if (u.empty() || v.empty()) {
    throw std::runtime_error{
        "Can't find minimum and maximum pixels on the image"};
  }
  const auto min_max_u = std::minmax_element(u.begin(), u.end());
  const auto min_max_v = std::minmax_element(v.begin(), v.end());

  Eigen::Vector2d min_vec(*min_max_u.first, *min_max_v.first);
  Eigen::Vector2d max_vec(*min_max_u.second, *min_max_v.second);

  return std::vector<Eigen::Vector2d>{min_vec, max_vec};
}

cv::Mat CamCylExtractor::CropImage(cv::Mat image, Eigen::Vector2d min_vector,
                                   Eigen::Vector2d max_vector) {
  std::cout << "Min vec: " << std::endl
            << min_vector << std::endl
            << "Max vec: " << std::endl
            << max_vector << std::endl;
  double width = max_vector(0) - min_vector(0);
  double height = max_vector(1) - min_vector(1);

  if (!camera_model_->PixelInImage(min_vector) ||
      !camera_model_->PixelInImage(max_vector)) {
    return image;
  }

  cv::Rect region_of_interest(min_vector(0), min_vector(1), width, height);
  cv::Mat cropped_image = image(region_of_interest);

  return cropped_image;
}

std::vector<cv::Vec4i> CamCylExtractor::DetectLines(cv::Mat &img,
                                                    int measurement_num) {
  /// Load an image
  if (!img.data) {
    throw std::runtime_error{"No image data"};
  }

  cv::Mat gray_img, edge_detected_img;

  /// Convert the image to grayscale
  cv::cvtColor(img, gray_img, CV_BGR2GRAY);

  auto v = Median(gray_img);

  int low_threshold = (int)std::max(0.0, (1.0 - percent_) * v);
  int max_threshold = (int)std::min(255.0, (1.0 + percent_) * v);

  std::string window_name =
      "Edge detected image " + std::to_string(measurement_num);

  /// Create a window
  cv::namedWindow(window_name + " Hough transform", cv::WINDOW_NORMAL);
  cv::resizeWindow(window_name + " Hough transform", img.cols, img.rows);

  /// Reduce noise with a kernel 3x3
  cv::blur(gray_img, edge_detected_img, cv::Size(3, 3));

  /// Canny detector
  cv::Canny(edge_detected_img, edge_detected_img, low_threshold, max_threshold,
            kernel_size);

  // Hough line transform
  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(edge_detected_img, lines, 1, CV_PI / 180, 50, 100, 10);

  distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) /
             sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1));

  return lines;
}

double CamCylExtractor::Median(cv::Mat channel) {
  double m = (channel.rows * channel.cols) / 2;
  int bin = 0;
  double med = -1.0;

  int histSize = 256;
  float range[] = {0, 256};
  const float *histRange = {range};
  bool uniform = true;
  bool accumulate = false;
  cv::Mat hist;
  cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange,
               uniform, accumulate);

  for (int i = 0; i < histSize && med < 0.0; ++i) {
    bin += cvRound(hist.at<float>(i));
    if (bin > m && med < 0.0)
      med = i;
  }

  return med;
}

cv::Mat
CamCylExtractor::ColorPixelsOnImage(cv::Mat &img,
                                    std::vector<Eigen::Vector2d> points) {
  for (auto pixel : points) {
    if (camera_model_->PixelInImage(pixel)) {
      cv::circle(img, cv::Point(pixel(0), pixel(1)), 2, cv::Scalar(0, 0, 255),
                 cv::FILLED, cv::LINE_8);
    }
  }
  return img;
}

} // end namespace vicon_calibration
