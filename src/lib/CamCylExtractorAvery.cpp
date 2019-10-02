#include "vicon_calibration/CamCylExtractor.h"
#include "vicon_calibration/utils.h"

#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <thread>

#include <beam_utils/math.hpp>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

CamCylExtractor::CamCylExtractor() {
  SetDefaultParameters();
  measurement_valid_ = true;
}

void CamCylExtractor::SetTargetParams(
    vicon_calibration::CylinderTgtParams &params) {
  target_params_ = params;
  target_params_set_ = true;

  if (target_params_.radius <= 0 || target_params_.height <= 0) {
    throw std::runtime_error{
        "Invalid cylinder dimentions. Radius: " +
        std::to_string(target_params_.radius) +
        " Height: " + std::to_string(target_params_.height)};
  }

  if (target_params_.radius + image_processing_params_.cropbox_offset <= 0) {
    throw std::runtime_error{
        "Invalid offset value: " +
        std::to_string(image_processing_params_.cropbox_offset)};
  }

  PopulateCylinderPoints();
}

void CamCylExtractor::SetImageProcessingParams(
    vicon_calibration::ImageProcessingParams &params) {
  image_processing_params_ = params;

  if (image_processing_params_.cropbox_offset <= 0) {
    LOG_WARN(
        "Using an offset of %.2f. The bounding box for cropping the "
        "image might be too small and edges might not be detected accurately.",
        image_processing_params_.cropbox_offset);
  }

  if (image_processing_params_.dist_criteria == 0 ||
      image_processing_params_.rot_criteria == 0) {
    LOG_WARN("[WARNING] Using tight criteria for accepting measurements. "
             "Distance criteria of %.2f and rotation criteria of %.2f.",
             image_processing_params_.dist_criteria,
             image_processing_params_.rot_criteria);
  }
}

void CamCylExtractor::SetCameraParams(vicon_calibration::CameraParams &params) {
  camera_params_ = params;
  camera_params_set_ = true;
  camera_model_ = beam_calibration::CameraModel::LoadJSON(params.intrinsics);
}

void CamCylExtractor::ExtractCylinder(Eigen::Affine3d T_CAMERA_TARGET_EST,
                                      cv::Mat &image) {
  measurement_valid_ = true;
  if (!image.data) {
    throw std::invalid_argument{"No image data"};
  }

  if (!beam::IsTransformationMatrix(T_CAMERA_TARGET_EST.matrix())) {
    throw std::invalid_argument{"Invalid transform"};
  }

  if (!target_params_set_) {
    throw std::invalid_argument{"Target parameters not set."};
  }

  if (!camera_params_set_) {
    throw std::invalid_argument{"Camera parameters not set."};
  }

  T_CAMERA_TARGET_EST_ = T_CAMERA_TARGET_EST;
  cv::Mat undistorted_img;
  if (camera_params_.images_distorted) {
    undistorted_img = camera_model_->UndistortImage(image);
  } else {
    undistorted_img = image;
  }

  std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box = GetBoundingBox();

  if (!measurement_valid_) {
    LOG_WARN("Bounding box not within image plane, skipping measurement.");
    if (image_processing_params_.show_measurements) {
      std::cout << "Bounding box [minu, minv, maxu, maxv]:\n"
                << "[" << bounding_box.first[0] << ", "
                << bounding_box.first[1] << ", "
                << bounding_box.second[0] << ", "
                << bounding_box.second[1] << "]\n"
                << "Image Width: " << camera_model_->GetWidth() << "\n"
                << "Image Height: " << camera_model_->GetHeight() << "\n";
      std::cout << "Showing failed measurement\n"
                << "Press [c] to continue with other measurements\n";
      cv::namedWindow("Invalid Measurement", cv::WINDOW_NORMAL);
      cv::resizeWindow("Invalid Measurement", undistorted_img.cols / 2,
                       undistorted_img.rows / 2);
      double scale = 0.3;
      cv::Mat current_image_w_axes = utils::DrawAxis(
          undistorted_img, T_CAMERA_TARGET_EST_, camera_model_, scale);
      cv::imshow("Invalid Measurement", current_image_w_axes);
      auto key = cv::waitKey();
      while (key != 67 && key != 99) {
        key = cv::waitKey();
      }
      cv::destroyAllWindows();
    }

    measurement_ = Eigen::Vector3d(-1, -1, -1);
    measurement_valid_ = false;
    measurement_complete_ = true;
    return;
  }

  auto cropped_image =
      CropImage(undistorted_img, bounding_box.first, bounding_box.second);
  int min_line_length = (bounding_box.second(1) - bounding_box.first(1)) *
                        image_processing_params_.min_length_ratio;
  int max_gap = (bounding_box.second(1) - bounding_box.first(1)) *
                image_processing_params_.max_gap_ratio;
  auto estimated_edge_lines = GetEstimatedEdges();
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
  auto detected_lines =
      DetectLines(undistorted_img, cropped_image, min_line_length, max_gap);

  if (detected_lines.empty()) {
    LOG_WARN("No lines detected in the image. Skipping measurement.");
    if (image_processing_params_.show_measurements) {
      std::cout << "Showing failed measurement\n"
                << "Press [c] to continue with other measurements\n";
      double scale = 0.3;
      cv::Mat current_image_w_axes = utils::DrawAxis(
          undistorted_img, T_CAMERA_TARGET_EST_, camera_model_, scale);
      cv::namedWindow("Invalid Measurement", cv::WINDOW_NORMAL);
      cv::resizeWindow("Invalid Measurement", cropped_image.cols / 2,
                       cropped_image.rows / 2);
      cv::imshow("Invalid Measurement", current_image_w_axes);
      auto key = cv::waitKey();
      while (key != 67 && key != 99) {
        key = cv::waitKey();
      }
      cv::destroyAllWindows();
    }

    measurement_ = Eigen::Vector3d(-1, -1, -1);
    measurement_valid_ = false;
    measurement_complete_ = true;
    return;
  }

  cv::Vec4i detected_line1, detected_line2;
  std::map<double, cv::Vec4i> dist_line_map_1, dist_line_map_2;

  double d1, d2, dist_to_bounding_box;
  Eigen::Vector2d end_point;
  bool is_bounding_line;
  for (auto line : detected_lines) {
    is_bounding_line = false;
    // If the line is a bounding line, ignore that line
    for (auto bounding_line : bounding_lines) {
      if (CalcDistBetweenLines(bounding_line, line) <=
          image_processing_params_.dist_criteria) {
        is_bounding_line = true;
        break;
      }
    }

    if (is_bounding_line)
      continue;

    d1 = CalcDistBetweenLines(estimated_edge_lines.first, line);
    d2 = CalcDistBetweenLines(estimated_edge_lines.second, line);
    if (d1 < d2) {
      dist_line_map_1.insert(std::pair<double, cv::Vec4i>(d1, line));
    } else {
      dist_line_map_2.insert(std::pair<double, cv::Vec4i>(d2, line));
    }
  }

  if (dist_line_map_1.empty() || dist_line_map_2.empty()) {
    std::cout << "No cylinder lines detected" << std::endl;
    std::cout << "Press [c] to continue with other measurements" << std::endl;
    for (auto line : detected_lines) {
      DrawLine(cropped_image, line, 0, 0, 0);
    }
    cv::namedWindow("Invalid Measurement", cv::WINDOW_NORMAL);
    cv::resizeWindow("Invalid Measurement", cropped_image.cols / 2,
                     cropped_image.rows / 2);
    double scale = 0.3;
    cv::Mat current_image_w_axes = utils::DrawAxis(
        cropped_image, T_CAMERA_TARGET_EST_, camera_model_, scale);
    cv::imshow("Invalid Measurement", current_image_w_axes);
    auto key = cv::waitKey();
    while (key != 67 && key != 99) {
      key = cv::waitKey();
    }
    cv::destroyAllWindows();
    measurement_ = Eigen::Vector3d(-1, -1, -1);
    measurement_valid_ = false;
    measurement_complete_ = true;

  } else {
    detected_line1 = dist_line_map_1.begin()->second;
    detected_line2 = dist_line_map_2.begin()->second;
    std::cout << "first detected edge: " << detected_line1 << std::endl;
    std::cout << "second detected edge: " << detected_line2 << std::endl;

    measurement_ = CalcMeasurement(detected_line1, detected_line2);
    auto estimated_measurement = CalcMeasurement(estimated_edge_lines.first,
                                                 estimated_edge_lines.second);
    std::cout << "Estimated Measurement: " << std::endl
              << "  Mid point: " << estimated_measurement(0) << ", "
              << estimated_measurement(1) << std::endl
              << "  Angle (deg): " << estimated_measurement(2) * 180 / M_PI
              << std::endl;

    std::cout << "Measurement: " << std::endl
              << "  Mid point: " << measurement_(0) << ", " << measurement_(1)
              << std::endl
              << "  Angle (deg): " << measurement_(2) * 180 / M_PI << std::endl;

    if (image_processing_params_.show_measurements) {
      LOG_ERROR("Showing measurements!");
      // Show measurement and get user input for accepting the measurement
      DrawLine(cropped_image, detected_line1, 0, 0, 255);
      DrawLine(cropped_image, detected_line2, 0, 0, 255);

      ColorPixelsOnImage(cropped_image,
                         std::vector<Eigen::Vector2d>{
                             Eigen::Vector2d(measurement_(0), measurement_(1))},
                         255, 0, 0, 4);

      cv::namedWindow("Measurement", cv::WINDOW_NORMAL);
      cv::resizeWindow("Measurement", cropped_image.cols / 2,
                       cropped_image.rows / 2);

      std::cout << "-------------------------------" << std::endl
                << "Legend:" << std::endl
                << "  Blue -> Detected edges" << std::endl
                << "  Red -> Measured mid point" << std::endl
                << "Accept measurement? [y/n]" << std::endl;
      double scale = 0.3;
      cv::Mat current_image_w_axes = utils::DrawAxis(
          cropped_image, T_CAMERA_TARGET_EST_, camera_model_, scale);
      imshow("Measurement", current_image_w_axes);

      auto key = cv::waitKey();
      while (key != 121 && key != 89 && key != 110 && key != 78) {
        key = cv::waitKey();
      }
      cv::destroyAllWindows();

      if (key == 121 || key == 89) {
        measurement_valid_ = true;
      } else if (key == 110 || key == 78) {
        measurement_valid_ = false;
      }
      measurement_complete_ = true;
    } else {
      dist_err_ = sqrt((measurement_(0) - estimated_measurement(0)) *
                           (measurement_(0) - estimated_measurement(0)) +
                       (measurement_(1) - estimated_measurement(1)) *
                           (measurement_(1) - estimated_measurement(1)));
      rot_err_ = abs(measurement_(2) - estimated_measurement(2));

      dist_err_ = std::round(dist_err_ * 10000) / 10000;
      rot_err_ = std::round(rot_err_ * 10000) / 10000;
      if (dist_err_ > image_processing_params_.dist_criteria ||
          rot_err_ > image_processing_params_.rot_criteria) {
        measurement_valid_ = false;
      } else {
        measurement_valid_ = true;
      }
      measurement_complete_ = true;
    }
  }
}

std::pair<Eigen::Vector3d, bool> CamCylExtractor::GetMeasurementInfo() {
  if (measurement_complete_) {
    return std::make_pair(measurement_, measurement_valid_);
  } else {
    throw std::runtime_error{"Measurement incomplete. Please run "
                             "ExtractCylinder() before getting the "
                             "measurement information."};
  }
  measurement_complete_ = false;
}

std::pair<double, double> CamCylExtractor::GetErrors() {
  if (measurement_complete_) {
    return std::make_pair(dist_err_, rot_err_);
  } else {
    throw std::runtime_error{"Measurement incomplete. Please run "
                             "ExtractCylinder() before getting the "
                             "measurement error values."};
  }
  measurement_complete_ = false;
}

void CamCylExtractor::SetDefaultParameters() {
  target_params_set_ = false;
  camera_params_set_ = false;
  image_processing_params_.num_intersections = 300;
  image_processing_params_.min_length_ratio = 0.5;
  image_processing_params_.max_gap_ratio = 0.078;
  image_processing_params_.canny_ratio = 0.1;
  image_processing_params_.cropbox_offset = 0.15;
  image_processing_params_.dist_criteria = 5;
  image_processing_params_.rot_criteria = 0.1;
  image_processing_params_.show_measurements = false;
}

void CamCylExtractor::PopulateCylinderPoints() {

  // Populate points for detecting edges
  center_line_points_ = std::make_pair<Eigen::Vector4d, Eigen::Vector4d>(
      Eigen::Vector4d(0, 0, 0, 1),
      Eigen::Vector4d(target_params_.height, 0, 0, 1));
  bottom_cylinder_points_ = GetCylinderPoints(target_params_.radius, 0, 0);
  top_cylinder_points_ =
      GetCylinderPoints(target_params_.radius, target_params_.height, 0);

  // Populate points for cropping the image
  auto bottom_bounding_points = GetCylinderPoints(
      target_params_.radius, 0, -image_processing_params_.cropbox_offset);
  bounding_points_ =
      GetCylinderPoints(target_params_.radius, target_params_.height,
                        image_processing_params_.cropbox_offset);
  bounding_points_.insert(bounding_points_.end(),
                          bottom_bounding_points.begin(),
                          bottom_bounding_points.end());
}

std::vector<Eigen::Vector4d> CamCylExtractor::GetCylinderPoints(double radius,
                                                                double height,
                                                                double offset) {
  Eigen::Vector4d point(0, 0, 0, 1);
  std::vector<Eigen::Vector4d> points;

  double r = radius + offset;

  for (double theta = 0; theta < 2 * M_PI; theta += M_PI / 12) {
    point(2) = r * std::cos(theta);

    point(1) = r * std::sin(theta);

    point(0) = height + offset;
    points.push_back(point);
  }

  return points;
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> CamCylExtractor::GetBoundingBox() {
  std::vector<double> u, v;
  for (auto point : bounding_points_) {
    auto pixel = CylinderPointToPixel(point);
    u.push_back(pixel[0]);
    v.push_back(pixel[1]);
  }
  double minu = *std::min_element(u.begin(), u.end());
  double maxu = *std::max_element(u.begin(), u.end());
  double minv = *std::min_element(v.begin(), v.end());
  double maxv = *std::max_element(v.begin(), v.end());

  if (minu < 0 || maxu > camera_model_->GetWidth() || minv < 0 ||
      maxv > camera_model_->GetHeight()) {
    measurement_valid_ = false;
  }

  Eigen::Vector2d min_vec(minu, minv);
  Eigen::Vector2d max_vec(maxu, maxv);

  return std::pair<Eigen::Vector2d, Eigen::Vector2d>(min_vec, max_vec);
}

std::pair<cv::Vec4i, cv::Vec4i> CamCylExtractor::GetEstimatedEdges() {
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

  auto r = round(angles(0) / (M_PI / 2)) * (M_PI / 2);
  r = r * 180 / M_PI;
  r = (int)r % 360;
  r = r < 0 ? r + 360 : r;
  std::cout << "angle: " << r << std::endl;

  cv::Vec4i estimated_edge_1, estimated_edge_2;

  if (r == 0 || r == 180 || r == 2 * 360) {
    if ((top_max_pixel(0) < top_second_max_pixel(0) &&
         bottom_max_pixel(0) < bottom_second_max_pixel(0)) ||
        (top_max_pixel(0) > top_second_max_pixel(0) &&
         bottom_max_pixel(0) > bottom_second_max_pixel(0))) {
      estimated_edge_1[0] = top_max_pixel(0);
      estimated_edge_1[1] = top_max_pixel(1);
      estimated_edge_1[2] = bottom_max_pixel(0);
      estimated_edge_1[3] = bottom_max_pixel(1);

      estimated_edge_2[0] = top_second_max_pixel(0);
      estimated_edge_2[1] = top_second_max_pixel(1);
      estimated_edge_2[2] = bottom_second_max_pixel(0);
      estimated_edge_2[3] = bottom_second_max_pixel(1);
    } else {
      estimated_edge_1[0] = top_max_pixel(0);
      estimated_edge_1[1] = top_max_pixel(1);
      estimated_edge_1[2] = bottom_second_max_pixel(0);
      estimated_edge_1[3] = bottom_second_max_pixel(1);

      estimated_edge_2[0] = top_second_max_pixel(0);
      estimated_edge_2[1] = top_second_max_pixel(1);
      estimated_edge_2[2] = bottom_max_pixel(0);
      estimated_edge_2[3] = bottom_max_pixel(1);
    }
  } else if (r == 90 || r == 270) {
    if ((top_max_pixel(1) < top_second_max_pixel(1) &&
         bottom_max_pixel(1) < bottom_second_max_pixel(1)) ||
        (top_max_pixel(1) > top_second_max_pixel(1) &&
         bottom_max_pixel(1) > bottom_second_max_pixel(1))) {
      estimated_edge_1[0] = top_max_pixel(0);
      estimated_edge_1[1] = top_max_pixel(1);
      estimated_edge_1[2] = bottom_max_pixel(0);
      estimated_edge_1[3] = bottom_max_pixel(1);

      estimated_edge_2[0] = top_second_max_pixel(0);
      estimated_edge_2[1] = top_second_max_pixel(1);
      estimated_edge_2[2] = bottom_second_max_pixel(0);
      estimated_edge_2[3] = bottom_second_max_pixel(1);
    } else {
      estimated_edge_1[0] = top_max_pixel(0);
      estimated_edge_1[1] = top_max_pixel(1);
      estimated_edge_1[2] = bottom_second_max_pixel(0);
      estimated_edge_1[3] = bottom_second_max_pixel(1);

      estimated_edge_2[0] = top_second_max_pixel(0);
      estimated_edge_2[1] = top_second_max_pixel(1);
      estimated_edge_2[2] = bottom_max_pixel(0);
      estimated_edge_2[3] = bottom_max_pixel(1);
    }
  }

  std::cout << "first edge estimated: " << estimated_edge_1 << std::endl;
  std::cout << "second edge estimated: " << estimated_edge_2 << std::endl;

  return std::pair<cv::Vec4i, cv::Vec4i>(estimated_edge_1, estimated_edge_2);
}

std::vector<cv::Vec4i> CamCylExtractor::DetectLines(cv::Mat &orig_img,
                                                    cv::Mat &cropped_img,
                                                    int min_line_length,
                                                    int max_line_gap) {
  cv::Mat gray_img, cropped_gray_img, edge_detected_img;

  /// Convert the image to grayscale for edge detection
  cv::cvtColor(orig_img, gray_img, CV_BGR2GRAY);

  auto v = Median(gray_img);

  int lower_threshold =
      (int)std::max(0.0, (1.0 - image_processing_params_.canny_ratio) * v);
  int upper_threshold =
      (int)std::min(255.0, (1.0 + image_processing_params_.canny_ratio) * v);

  cv::cvtColor(cropped_img, cropped_gray_img, CV_BGR2GRAY);
  /// Reduce noise with a 3x3 kernel
  cv::blur(cropped_gray_img, edge_detected_img, cv::Size(3, 3));

  /// Canny detector
  cv::Canny(edge_detected_img, edge_detected_img, lower_threshold,
            upper_threshold, 3);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3),
                                             cv::Point(0, 0));
  // Dilate the edge detected image such that the lines are easily detected by
  // Hough line transform
  cv::dilate(edge_detected_img, edge_detected_img, kernel);

  // Hough line transform
  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(edge_detected_img, lines, 1, CV_PI / 180,
                  image_processing_params_.num_intersections, min_line_length,
                  max_line_gap);

  return lines;
}

Eigen::Vector3d CamCylExtractor::CalcMeasurement(cv::Vec4i edge_line1,
                                                 cv::Vec4i edge_line2) {
  cv::Vec4i center_line(
      (edge_line1[0] + edge_line2[0]) / 2, (edge_line1[1] + edge_line2[1]) / 2,
      (edge_line1[2] + edge_line2[2]) / 2, (edge_line1[3] + edge_line2[3]) / 2);

  double angle;
  if (center_line[0] == center_line[2]) {
    // the center line is horizontal
    angle = M_PI / 2;
  } else if (center_line[1] == center_line[3]) {
    // the center line is vertical
    angle = 0;
  } else {
    angle = std::acos((center_line[2] - center_line[0]) /
                      (center_line[3] - center_line[1]));
  }

  return Eigen::Vector3d((center_line[0] + center_line[2]) / 2,
                         (center_line[1] + center_line[3]) / 2, angle);
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

double CamCylExtractor::Median(cv::Mat &image) {
  int depth;

  if (image.depth() == CV_8U) {
    depth = 8;
  } else if (image.depth() == CV_16U) {
    depth = 16;
  }

  double m = (image.rows * image.cols) / 2;
  int bin = 0;
  double med = -1.0;

  int hist_size = pow(2, depth);
  float range[] = {0, (float)hist_size};
  const float *hist_range = {range};
  bool uniform = true;
  bool accumulate = false;
  cv::Mat hist;
  cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range,
               uniform, accumulate);

  for (int i = 0; i < hist_size && med < 0.0; ++i) {
    bin += cvRound(hist.at<float>(i));
    if (bin > m && med < 0.0)
      med = i;
  }

  return med;
}

void CamCylExtractor::DrawLine(cv::Mat &image, cv::Vec4i line, int r, int g,
                               int b) {
  cv::line(image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]),
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
  Eigen::Vector4d homogeneous_point;

  homogeneous_point = T_CAMERA_TARGET_EST_.matrix() * point;

  transformed_point << homogeneous_point[0] / homogeneous_point[3],
      homogeneous_point[1] / homogeneous_point[3],
      homogeneous_point[2] / homogeneous_point[3];

  if (camera_params_.images_distorted) {
    camera_model_->SetUndistortedIntrinsics(camera_model_->GetIntrinsics());
    return camera_model_->ProjectUndistortedPoint(transformed_point);
  } else {
    return camera_model_->ProjectPoint(transformed_point);
  }
}

} // end namespace vicon_calibration
