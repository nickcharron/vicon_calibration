#include <vicon_calibration/measurement_extractors/CylinderCameraExtractor.h>

#include <pcl/surface/convex_hull.h>
#include <pcl/surface/impl/convex_hull.hpp>

#include <vicon_calibration/Utils.h>

namespace vicon_calibration {

void CylinderCameraExtractor::GetKeypoints() {
  CropImage();

  if (!measurement_valid_) {
    measurement_complete_ = true;
    return;
  }

  // Threshold the image
  cv::Mat image_binary;
  cv::inRange(*image_cropped_, color_threshold_min_, color_threshold_max_,
              image_binary);

  // Find all the contours in the thresholded image
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(image_binary, contours, cv::RETR_LIST,
                   cv::CHAIN_APPROX_NONE);

  // extract the largest contour
  double max_area = 0;
  size_t max_area_iter = 0;
  for (size_t i = 0; i < contours.size(); i++) {
    double area = cv::contourArea(contours[i]);
    if (area > max_area) {
      max_area = area;
      max_area_iter = i;
    }
  }
  area_detected_ = max_area;

  if (max_area < 10) {
    LOG_INFO(
        "No target found in image. Try relaxing colour thresholding or check "
        "your initial calibration estimates");
    measurement_valid_ = false;

    // draw contours on binary image
    cv::cvtColor(image_binary, image_binary, cv::COLOR_GRAY2RGB);
    cv::Mat bin_image_annotated = image_binary.clone();
    for (int i = 0; i < contours.size(); i++) {
      if (i == max_area_iter) { continue; }
      cv::drawContours(bin_image_annotated, contours, i, cv::Scalar(0, 0, 255));
    }
    cv::drawContours(bin_image_annotated, contours, max_area_iter,
                     cv::Scalar(255, 0, 0));

    DisplayImagePair(
        *image_in_, bin_image_annotated,
        "Invalid Measurement (no target found after thresholding)",
        "Showing original image (left) and thresholded image with contours "
        "(right). All contours are in red, with the lagest one in blue.",
        false);

    return;
  }
  target_contour_ = contours[max_area_iter];

  if (show_measurements_) {
    *image_annotated_ = *image_cropped_;
    cv::drawContours(*image_annotated_, contours,
                     static_cast<int>(max_area_iter), cv::Scalar(0, 0, 255), 2);
  }

  // convert to pcl point cloud
  keypoints_measured_->points.clear();
  for (uint32_t i = 0; i < target_contour_.size(); i++) {
    pcl::PointXY pixel;
    pixel.x = target_contour_[i].x;
    pixel.y = target_contour_[i].y;
    keypoints_measured_->points.push_back(pixel);
  }
  GetEstimatedArea();
  CheckError();
}

void CylinderCameraExtractor::GetEstimatedArea() {
  // project points
  pcl::PointCloud<pcl::PointXYZ>::Ptr projected_points =
      std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  for (PointCloud::iterator it = target_params_->template_cloud->begin();
       it != target_params_->template_cloud->end(); ++it) {
    bool pix_valid;
    Eigen::Vector2d pix;
    TargetPointToPixel(Eigen::Vector4d(it->x, it->y, it->z, 1), pix, pix_valid);
    if (pix_valid) {
      pcl::PointXYZ point;
      point.x = pix[0];
      point.y = pix[1];
      point.z = 0;
      projected_points->points.push_back(point);
    }
  }

  // calculate convex hull
  pcl::PointCloud<pcl::PointXYZ> hull_points;
  pcl::ConvexHull<pcl::PointXYZ> convex_hull;
  convex_hull.setInputCloud(projected_points);
  convex_hull.reconstruct(hull_points);

  // calculate area
  double area = 0.0;
  for (unsigned int i = 0; i < hull_points.points.size(); ++i) {
    int j = (i + 1) % hull_points.points.size();
    area += 0.5 * (hull_points.points[i].x * hull_points.points[j].y -
                   hull_points.points[j].x * hull_points.points[i].y);
  }
  area_expected_ = std::abs(area);
}

void CylinderCameraExtractor::CheckError() {
  double area_diff = std::abs(area_expected_ - area_detected_) / area_expected_;
  if (area_diff > area_error_allowance_) {
    measurement_valid_ = false;
    if (show_measurements_) {
      LOG_INFO("Measurement invalid because the difference between the "
               "expected area and the detected area is too great. Area "
               "Expected: %.3f, Area Detected: %.3f, Difference: "
               "%.3f, Allowed: %.3f",
               area_expected_, area_detected_, area_diff,
               area_error_allowance_);
    }
    return;
  }
  measurement_valid_ = true;
}

void CylinderCameraExtractor::DrawContourAxis(
    std::shared_ptr<cv::Mat>& img_pointer, const cv::Point& p,
    const cv::Point& q, const cv::Scalar& colour, const float scale = 0.2) {
  cv::Point p_ = p;
  cv::Point q_ = q;

  float angle = std::atan2(p_.y - q_.y, p_.x - q_.x); // angle in radians
  float hypotenuse =
      std::sqrt((p_.y - q_.y) * (p_.y - q_.y) + (p_.x - q_.x) * (p_.x - q_.x));

  // Here we lengthen the arrow by a factor of scale
  q_.x = static_cast<int>(p_.x - scale * hypotenuse * std::cos(angle));
  q_.y = static_cast<int>(p_.y - scale * hypotenuse * std::sin(angle));
  cv::line(*img_pointer, p, q, colour, 1, cv::LINE_AA);
  // create the arrow hooks
  p_.x = static_cast<int>(q_.x + 9 * std::cos(angle + CV_PI / 4));
  p_.y = static_cast<int>(q_.y + 9 * std::sin(angle + CV_PI / 4));
  cv::line(*img_pointer, p, q, colour, 1, cv::LINE_AA);
  p_.x = static_cast<int>(q_.x + 9 * std::cos(angle - CV_PI / 4));
  p_.y = static_cast<int>(q_.y + 9 * std::sin(angle - CV_PI / 4));
  cv::line(*img_pointer, p, q, colour, 1, cv::LINE_AA);
}

void CylinderCameraExtractor::DisplayImagePair(const cv::Mat& img1,
                                               const cv::Mat& img2,
                                               const std::string& display_name,
                                               const std::string& output_text,
                                               bool allow_override) {
  if (!show_measurements_) { return; }

  cv::Mat img1_w_axes = utils::DrawCoordinateFrame(img1, T_CAMERA_TARGET_EST_,
                                                   camera_params_->camera_model,
                                                   axis_plot_scale_);

  cv::Mat img2_w_axes = utils::DrawCoordinateFrame(img2, T_CAMERA_TARGET_EST_,
                                                   camera_params_->camera_model,
                                                   axis_plot_scale_);

  cv::Mat combined_imgs;
  cv::hconcat(img1_w_axes, img2_w_axes, combined_imgs);

  if (allow_override) {
    std::cout << output_text << std::endl
              << "Press [c] to continue with default\n"
              << "Press [y] to accept measurement\n"
              << "Press [n] to reject measurement\n"
              << "Press [s] to stop showing future measurements\n";
    cv::namedWindow(display_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(display_name, combined_imgs.cols / 2,
                     combined_imgs.rows / 2);
    cv::imshow(display_name, combined_imgs);
    auto key = 0;
    while (key != 67 && key != 99 && key != 121 && key != 110 && key != 115) {
      key = cv::waitKey();
      if (key == 121) {
        measurement_valid_ = true;
        std::cout << "Accepted measurement.\n";
      }
      if (key == 110) {
        measurement_valid_ = false;
        std::cout << "Rejected measurement.\n";
      }
      if (key == 115) {
        std::cout << "setting show measurements to false.\n";
        SetShowMeasurements(false);
      }
    }
  } else {
    std::cout << output_text << std::endl
              << "Press [c] to continue with default\n"
              << "Press [s] to stop showing future measurements\n";
    cv::namedWindow(display_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(display_name, combined_imgs.cols / 2,
                     combined_imgs.rows / 2);
    cv::imshow(display_name, combined_imgs);
    auto key = 0;
    while (key != 67 && key != 99 && key != 115) {
      key = cv::waitKey();
      if (key == 115) {
        std::cout << "setting show measurements to false.\n";
        SetShowMeasurements(false);
      }
    }
  }

  cv::destroyAllWindows();
}

} // namespace vicon_calibration
