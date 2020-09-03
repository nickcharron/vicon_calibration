#include "vicon_calibration/measurement_extractors/CylinderCameraExtractor.h"
#include "vicon_calibration/utils.h"

namespace vicon_calibration {

void CylinderCameraExtractor::GetKeypoints() {
  this->CropImage();
  
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
    if (area > max_area && area < 1e5) {
      max_area = area;
      max_area_iter = i;
    }
  }

  if (max_area < 10) {
    LOG_INFO(
        "No target found in image. Try relaxing colour thresholding or check "
        "your initial calibration estimates");
    measurement_valid_ = false;
    this->DisplayImage(
        *image_in_, "Invalid Measurement",
        "Showing original invalid image (no target found after thresholding)",
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
  this->GetMeasuredPose();
  this->GetEstimatedPose();
  this->CheckError();
}

// TODO: redo the error metrics to check for false measurements. I want to have
//       3 error checks:
//       1) check origin point is approximately the same spot.
//       2) check that the angle is approximately equal to estimated
//       3) check that area is approx. equal to the template projected area
void CylinderCameraExtractor::GetMeasuredPose() {
  // Construct a buffer used by the pca analysis
  int sz = static_cast<int>(target_contour_.size());
  cv::Mat data_pts = cv::Mat(sz, 2, CV_64F);
  for (int i = 0; i < data_pts.rows; i++) {
    data_pts.at<double>(i, 0) = target_contour_.at(i).x;
    data_pts.at<double>(i, 1) = target_contour_.at(i).y;
  }

  // Perform PCA analysis
  cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);

  // Store the center of the object
  cv::Point cntr =
      cv::Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                static_cast<int>(pca_analysis.mean.at<double>(0, 1)));

  // Store the eigenvalues and eigenvectors
  std::vector<cv::Point2d> eigen_vecs(2);
  std::vector<double> eigen_val(2);
  for (int i = 0; i < 2; i++) {
    eigen_vecs[i] = cv::Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
    eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
  }

  cv::Point p1 =
      cntr + 0.02 * cv::Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]),
                              static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
  cv::Point p2 =
      cntr - 0.02 * cv::Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]),
                              static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
  double angle =
      std::atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
  target_pose_measured_ = std::make_pair(cntr, angle);

  // Draw the principal components
  if (show_measurements_) {
    cv::circle(*image_annotated_, cntr, 3, cv::Scalar(255, 0, 255), 2);
    this->DrawContourAxis(image_annotated_, cntr, p1, cv::Scalar(0, 255, 0), 1);
    this->DrawContourAxis(image_annotated_, cntr, p2, cv::Scalar(255, 255, 0),
                          5);
  }
}

void CylinderCameraExtractor::GetEstimatedPose() {
  // first, we need to determine the height of the target. Assume coordinate
  // frame is at the top of the template cloud.
  double target_height = 0;
  for (PointCloud::iterator it = target_params_->template_cloud->begin();
       it != target_params_->template_cloud->end(); ++it) {
    double z_coord = it->z;
    if (z_coord > target_height) {
      target_height = z_coord;
    }
  }

  // project axis points to image
  Eigen::Vector4d point_center(target_height / 2, 0, 0, 1);
  Eigen::Vector4d point_origin(0, 0, 0, 1);
  opt<Eigen::Vector2i> pixel_center = this->TargetPointToPixel(point_center);
  opt<Eigen::Vector2i> pixel_origin = this->TargetPointToPixel(point_origin);

  if (!pixel_center.has_value() || !pixel_origin.has_value()) {
    measurement_valid_ = false;
  }

  // save center and angle
  double angle =
      std::atan2((pixel_origin.value()[1] - pixel_center.value()[1]),
                 (pixel_origin.value()[0] - pixel_center.value()[0]));
  cv::Point cv_point_center;
  cv_point_center.x = pixel_center.value()[0];
  cv_point_center.y = pixel_center.value()[1];
  target_pose_estimated_ = std::make_pair(cv_point_center, angle);
}

void CylinderCameraExtractor::CheckError() {
  double x_m = target_pose_measured_.first.x;
  double x_e = target_pose_estimated_.first.x;
  double y_m = target_pose_measured_.first.y;
  double y_e = target_pose_estimated_.first.y;
  double theta_m = target_pose_measured_.second;
  double theta_e = target_pose_estimated_.second;
  dist_err_ = std::sqrt((x_m - x_e) * (x_m - x_e) + (y_m - y_e) * (y_m - y_e));
  rot_err_ = theta_m - theta_e;
  rot_err_ = std::abs(rot_err_);

  // since the vector may be in the opposite direction:
  if (rot_err_ > 1.5708) {
    rot_err_ = rot_err_ - 3.14159;
  }

  if (dist_err_ > dist_acceptance_criteria_) {
    measurement_valid_ = false;
    if (show_measurements_) {
      LOG_INFO("Measurement invalid because the measured distance error is "
               "larger than the specified threshold. Distance error measured: "
               "%.3f, threshold: %.3f",
               dist_err_, dist_acceptance_criteria_);
    }
    return;
  }
  if (rot_err_ > rot_acceptance_criteria_) {
    measurement_valid_ = false;
    if (show_measurements_) {
      LOG_INFO("Measurement invalid because the measured rotation error is "
               "larger than the specified threshold. Rotation error measured: "
               "%.3f, threshold: %.3f",
               rot_err_, rot_acceptance_criteria_);
    }
    return;
  }
  measurement_valid_ = true;
}

void CylinderCameraExtractor::DrawContourAxis(
    std::shared_ptr<cv::Mat> &img_pointer, const cv::Point &p,
    const cv::Point &q, const cv::Scalar &colour, const float scale = 0.2) {
  cv::Point p_ = p;
  cv::Point q_ = q;
  double angle =
      std::atan2((double)p_.y - q_.y, (double)p_.x - q_.x); // angle in radians
  double hypotenuse = std::sqrt((double)(p_.y - q_.y) * (p_.y - q_.y) +
                                (p_.x - q_.x) * (p_.x - q_.x));
  // Here we lengthen the arrow by a factor of scale
  q_.x = (int)(p_.x - scale * hypotenuse * std::cos(angle));
  q_.y = (int)(p_.y - scale * hypotenuse * std::sin(angle));
  cv::line(*img_pointer, p, q, colour, 1, cv::LINE_AA);
  // create the arrow hooks
  p_.x = (int)(q_.x + 9 * std::cos(angle + CV_PI / 4));
  p_.y = (int)(q_.y + 9 * std::sin(angle + CV_PI / 4));
  cv::line(*img_pointer, p, q, colour, 1, cv::LINE_AA);
  p_.x = (int)(q_.x + 9 * std::cos(angle - CV_PI / 4));
  p_.y = (int)(q_.y + 9 * std::sin(angle - CV_PI / 4));
  cv::line(*img_pointer, p, q, colour, 1, cv::LINE_AA);
}

} // namespace vicon_calibration
