#include "vicon_calibration/CamCylExtractor.h"
#include "vicon_calibration/utils.h"

#include <cmath>
#include <iterator>
#include <map>
#include <pcl/io/pcd_io.h>
#include <set>
#include <thread>

#include <beam_utils/math.hpp>

namespace vicon_calibration {

CamCylExtractor::CamCylExtractor() {
  image_in_ = std::make_shared<cv::Mat>();
  image_undistorted_ = std::make_shared<cv::Mat>();
  image_cropped_ = std::make_shared<cv::Mat>();
  image_annotated_ = std::make_shared<cv::Mat>();
  target_contour_ = std::make_shared<std::vector<cv::Point>>();
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
}

void CamCylExtractor::SetImageProcessingParams(
    vicon_calibration::ImageProcessingParams &params) {
  image_processing_params_ = params;

  if (image_processing_params_.dist_criteria == 0 ||
      image_processing_params_.rot_criteria == 0) {
    LOG_WARN("Using tight criteria for accepting measurements. "
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

void CamCylExtractor::ExtractMeasurement(Eigen::Affine3d T_CAMERA_TARGET_EST,
                                         cv::Mat &image) {
  measurement_valid_ = true;
  measurement_complete_ = false;
  T_CAMERA_TARGET_EST_ = T_CAMERA_TARGET_EST;
  *image_in_ = image;
  CheckInputs();
  UndistortImage();
  if (!CropImage()) {
    return;
  }
  double scale = 0.3;
  *image_annotated_ = utils::DrawCoordinateFrame(
      *image_cropped_, T_CAMERA_TARGET_EST_, camera_model_, scale,
      camera_params_.images_distorted);

  GetMeasurementPoints();
  if(!measurement_valid_){
    measurement_complete_ = true;
    return;
  }
  GetMeasuredPose();
  GetEstimatedPose();
  CheckError();
  if (image_processing_params_.show_measurements) {
    DisplayResult();
  }
  measurement_complete_ = true;
}

std::shared_ptr<std::vector<cv::Point>> CamCylExtractor::GetMeasurements() {
  if (measurement_complete_) {
    return target_contour_;
  } else {
    throw std::runtime_error{"Measurement incomplete. Please run "
                             "ExtractMeasurement() before getting the "
                             "measurement information."};
  }
}

bool CamCylExtractor::GetMeasurementsValid() {
  if (measurement_complete_) {
    return measurement_valid_;
  } else {
    throw std::runtime_error{"Measurement incomplete. Please run "
                             "ExtractMeasurement() before getting the "
                             "measurement information."};
    return measurement_valid_;
  }
}

std::pair<double, double> CamCylExtractor::GetErrors() {
  if (measurement_complete_) {
    return std::make_pair(dist_err_, rot_err_);
  } else {
    throw std::runtime_error{"Measurement incomplete. Please run "
                             "ExtractMeasurement() before getting the "
                             "measurement error values."};
  }
  measurement_complete_ = false;
}

void CamCylExtractor::CheckInputs() {
  if (!image_in_->data) {
    throw std::invalid_argument{"No image data"};
  }

  if (!beam::IsTransformationMatrix(T_CAMERA_TARGET_EST_.matrix())) {
    throw std::invalid_argument{"Invalid transform"};
  }

  if (!target_params_set_) {
    throw std::invalid_argument{"Target parameters not set."};
  }

  if (!camera_params_set_) {
    LOG_WARN("Image processing params not set, using default.");
  }
}

void CamCylExtractor::UndistortImage() {
  if (camera_params_.images_distorted) {
    *image_undistorted_ = camera_model_->UndistortImage(*image_in_);
  } else {
    image_undistorted_ = image_in_;
  }
}

bool CamCylExtractor::CropImage() {
  // load target
  PointCloud::Ptr template_cloud = boost::make_shared<PointCloud>();
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(target_params_.template_cloud,
                                          *template_cloud) == -1) {
    LOG_ERROR("Couldn't read template file: %s\n",
              target_params_.template_cloud.c_str());
  }
  Eigen::Vector4d point_target(0, 0, 0, 1);
  std::vector<double> u, v;
  for (PointCloud::iterator it = template_cloud->begin();
       it != template_cloud->end(); ++it) {
    point_target[0] = it->x;
    point_target[1] = it->y;
    point_target[2] = it->z;
    auto pixel = TargetPointToPixel(point_target);
    u.push_back(pixel[0]);
    v.push_back(pixel[1]);
  }
  // Get Cropbox corners
  bool cropbox_in_image = true;
  bool target_in_image = true;
  double minu = *std::min_element(u.begin(), u.end());
  double maxu = *std::max_element(u.begin(), u.end());
  double minv = *std::min_element(v.begin(), v.end());
  double maxv = *std::max_element(v.begin(), v.end());
  Eigen::Vector2d min_vec(minu, minv);
  Eigen::Vector2d max_vec(maxu, maxv);
  Eigen::Vector2d min_vec_buffer(
      minu - image_processing_params_.crop_threshold_u / 2,
      minv - image_processing_params_.crop_threshold_v / 2);
  Eigen::Vector2d max_vec_buffer(
      maxu + image_processing_params_.crop_threshold_u / 2,
      maxv + image_processing_params_.crop_threshold_v / 2);
  if (!camera_model_->PixelInImage(min_vec) ||
      !camera_model_->PixelInImage(max_vec)) {
    target_in_image = false;
  }
  if (!camera_model_->PixelInImage(min_vec_buffer) ||
      !camera_model_->PixelInImage(max_vec_buffer)) {
    cropbox_in_image = false;
  }
  if (!target_in_image) {
    LOG_WARN("Target not in image, skipping measurement.");
    if (image_processing_params_.show_measurements) {
      LOG_INFO("Target corners: [minu, minv, maxu, maxv]: [%d, %d, %d, %d]",
               minu, minv, maxu, maxv);
      LOG_INFO("Image Dimensions: [%d x %d]", camera_model_->GetWidth(),
               camera_model_->GetHeight());
      std::cout << "Showing failed measurement (target not in image)\n"
                << "Press [c] to continue with other measurements\n";
      cv::namedWindow("Invalid Measurement", cv::WINDOW_NORMAL);
      cv::resizeWindow("Invalid Measurement", image_undistorted_->cols / 2,
                       image_undistorted_->rows / 2);
      double scale = 0.3;
      cv::Mat current_image_w_axes = utils::DrawCoordinateFrame(
          *image_undistorted_, T_CAMERA_TARGET_EST_, camera_model_, scale,
          camera_params_.images_distorted);
      cv::imshow("Invalid Measurement", current_image_w_axes);
      auto key = cv::waitKey();
      while (key != 67 && key != 99) {
        key = cv::waitKey();
      }
      cv::destroyAllWindows();
    }
    measurement_valid_ = false;
    measurement_complete_ = true;
  } else if (!cropbox_in_image) {
    LOG_WARN("Target in image, but cropbox is not, you may want to relax your "
             "crop threshold. Skipping measurement.");
    if (image_processing_params_.show_measurements) {
      LOG_INFO("Target corners: [minu, minv, maxu, maxv]: [%d, %d, %d, %d]",
               minu, minv, maxu, maxv);
      LOG_INFO("Cropbox corners: [minu, minv, maxu, maxv]: [%d, %d, %d, %d]",
               min_vec_buffer[0], min_vec_buffer[1], max_vec_buffer[0],
               max_vec_buffer[1]);
      LOG_INFO("Image Dimensions: [%d x %d]", camera_model_->GetWidth(),
               camera_model_->GetHeight());
      std::cout << "Showing failed measurement (cropbox not in image)\n"
                << "Press [c] to continue with other measurements\n";
      cv::namedWindow("Invalid Measurement", cv::WINDOW_NORMAL);
      cv::resizeWindow("Invalid Measurement", image_undistorted_->cols / 2,
                       image_undistorted_->rows / 2);
      double scale = 0.3;
      cv::Mat current_image_w_axes = utils::DrawCoordinateFrame(
          *image_undistorted_, T_CAMERA_TARGET_EST_, camera_model_, scale,
          camera_params_.images_distorted);
      cv::imshow("Invalid Measurement", current_image_w_axes);
      auto key = cv::waitKey();
      while (key != 67 && key != 99) {
        key = cv::waitKey();
      }
      cv::destroyAllWindows();
    }
    measurement_valid_ = false;
    measurement_complete_ = true;
  } else {
    LOG_INFO("Estimated target + cropbox in image.");
    cv::Mat bounded_img(image_undistorted_->rows, image_undistorted_->cols,
                        image_undistorted_->depth());
    double width = max_vec_buffer(0) - min_vec_buffer(0);
    double height = max_vec_buffer(1) - min_vec_buffer(1);
    cv::Mat mask =
        cv::Mat::zeros(image_undistorted_->rows, image_undistorted_->cols,
                       image_undistorted_->depth());
    mask(cv::Rect(min_vec_buffer(0), min_vec_buffer(1), width, height)) = 1;
    image_undistorted_->copyTo(bounded_img, mask);
    *image_cropped_ = bounded_img;
  }
  return cropbox_in_image;
}

void CamCylExtractor::GetMeasurementPoints() {
  // Threshold the image
  cv::Mat image_binary;
  cv::Scalar min_threshold(target_params_.color_threshold_min[0],
                           target_params_.color_threshold_min[1],
                           target_params_.color_threshold_min[2]);
  cv::Scalar max_threshold(target_params_.color_threshold_max[0],
                           target_params_.color_threshold_max[1],
                           target_params_.color_threshold_max[2]);
  cv::inRange(*image_cropped_, min_threshold, max_threshold, image_binary);

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
    LOG_INFO("No target found in image. Try relaxing thresholding or check "
             "your initial calibration estimates");
    measurement_valid_ = false;
    return;
  }
  *target_contour_ = contours[max_area_iter];
  cv::drawContours(*image_annotated_, contours, static_cast<int>(max_area_iter),
                   cv::Scalar(0, 0, 255), 2);
}

void CamCylExtractor::GetMeasuredPose() {
  // Construct a buffer used by the pca analysis
  int sz = static_cast<int>(target_contour_->size());
  cv::Mat data_pts = cv::Mat(sz, 2, CV_64F);
  for (int i = 0; i < data_pts.rows; i++) {
    data_pts.at<double>(i, 0) = target_contour_->at(i).x;
    data_pts.at<double>(i, 1) = target_contour_->at(i).y;
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
  // Draw the principal components
  cv::circle(*image_annotated_, cntr, 3, cv::Scalar(255, 0, 255), 2);
  cv::Point p1 =
      cntr + 0.02 * cv::Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]),
                              static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
  cv::Point p2 =
      cntr - 0.02 * cv::Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]),
                              static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
  this->DrawContourAxis(image_annotated_, cntr, p1, cv::Scalar(0, 255, 0), 1);
  this->DrawContourAxis(image_annotated_, cntr, p2, cv::Scalar(255, 255, 0), 5);
  double angle =
      std::atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
  target_pose_measured_ = std::make_pair(cntr, angle);
}

void CamCylExtractor::GetEstimatedPose() {
  Eigen::Vector4d point_center(target_params_.height / 2, 0, 0, 1);
  Eigen::Vector4d point_origin(0, 0, 0, 1);
  Eigen::Vector2d pixel_center = TargetPointToPixel(point_center);
  Eigen::Vector2d pixel_origin = TargetPointToPixel(point_origin);
  double angle = std::atan2((pixel_origin[1] - pixel_center[1]),
                            (pixel_origin[0] - pixel_center[0]));
  cv::Point cv_point_center;
  cv_point_center.x = pixel_center[0];
  cv_point_center.y = pixel_center[1];
  target_pose_estimated_ = std::make_pair(cv_point_center, angle);
}

void CamCylExtractor::CheckError() {
  double x_m = target_pose_measured_.first.x;
  double x_e = target_pose_estimated_.first.x;
  double y_m = target_pose_measured_.first.y;
  double y_e = target_pose_estimated_.first.y;
  double theta_m = target_pose_measured_.second;
  double theta_e = target_pose_estimated_.second;
  dist_err_ = std::sqrt((x_m - x_e) * (x_m - x_e) + (y_m - y_e) * (y_m - y_e));
  rot_err_ = theta_m - theta_e;
  rot_err_ = std::abs(rot_err_);
  if (dist_err_ > image_processing_params_.dist_criteria) {
    measurement_valid_ = false;
    if (image_processing_params_.show_measurements) {
      std::cout << "Measurement invalid because the measured distance error is "
                   "larger than the specified threshold\n"
                << "Distance error measured: " << dist_err_ << "\n"
                << "Threshold: " << image_processing_params_.dist_criteria
                << "\n";
    }
    return;
  }
  if (rot_err_ > image_processing_params_.rot_criteria) {
    measurement_valid_ = false;
    if (image_processing_params_.show_measurements) {
      std::cout << "Measurement invalid because the measured rotation error is "
                   "larger than the specified threshold\n"
                << "Rotation error measured: " << rot_err_ << "\n"
                << "Threshold: " << image_processing_params_.rot_criteria
                << "\n";
    }
    return;
  }
  measurement_valid_ = true;
}

void CamCylExtractor::DrawContourAxis(std::shared_ptr<cv::Mat> &img_pointer,
                                      cv::Point p, cv::Point q,
                                      cv::Scalar colour,
                                      const float scale = 0.2) {
  double angle =
      std::atan2((double)p.y - q.y, (double)p.x - q.x); // angle in radians
  double hypotenuse =
      std::sqrt((double)(p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
  // Here we lengthen the arrow by a factor of scale
  q.x = (int)(p.x - scale * hypotenuse * std::cos(angle));
  q.y = (int)(p.y - scale * hypotenuse * std::sin(angle));
  cv::line(*img_pointer, p, q, colour, 1, cv::LINE_AA);
  // create the arrow hooks
  p.x = (int)(q.x + 9 * std::cos(angle + CV_PI / 4));
  p.y = (int)(q.y + 9 * std::sin(angle + CV_PI / 4));
  cv::line(*img_pointer, p, q, colour, 1, cv::LINE_AA);
  p.x = (int)(q.x + 9 * std::cos(angle - CV_PI / 4));
  p.y = (int)(q.y + 9 * std::sin(angle - CV_PI / 4));
  cv::line(*img_pointer, p, q, colour, 1, cv::LINE_AA);
}

Eigen::Vector2d CamCylExtractor::TargetPointToPixel(Eigen::Vector4d point) {
  Eigen::Vector3d transformed_point;
  Eigen::Vector4d homogeneous_point;

  homogeneous_point = T_CAMERA_TARGET_EST_.matrix() * point;

  transformed_point << homogeneous_point[0], homogeneous_point[1],
      homogeneous_point[2];

  if (camera_params_.images_distorted) {
    camera_model_->SetUndistortedIntrinsics(camera_model_->GetIntrinsics());
    return camera_model_->ProjectUndistortedPoint(transformed_point);
  } else {
    return camera_model_->ProjectPoint(transformed_point);
  }
}

void CamCylExtractor::DisplayResult() {
  if (measurement_valid_) {
    std::cout << "Showing passed measurement (target not in image)\n"
              << "Press [c] to continue with other measurements\n";
    cv::namedWindow("Valid Measurement", cv::WINDOW_NORMAL);
    cv::resizeWindow("Valid Measurement", image_annotated_->cols / 2,
                     image_annotated_->rows / 2);
    cv::imshow("Valid Measurement", *image_annotated_);
    auto key = cv::waitKey();
    while (key != 67 && key != 99) {
      key = cv::waitKey();
    }
    cv::destroyAllWindows();
  } else {
    std::cout << "Showing failed measurement\n"
              << "Press [c] to continue with other measurements\n";
    cv::namedWindow("Invalid Measurement", cv::WINDOW_NORMAL);
    cv::resizeWindow("Invalid Measurement", image_annotated_->cols / 2,
                     image_annotated_->rows / 2);
    cv::imshow("Invalid Measurement", *image_annotated_);
    auto key = cv::waitKey();
    while (key != 67 && key != 99) {
      key = cv::waitKey();
    }
    cv::destroyAllWindows();
  }
}

} // end namespace vicon_calibration
