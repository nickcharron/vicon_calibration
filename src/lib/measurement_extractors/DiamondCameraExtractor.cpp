#include "vicon_calibration/measurement_extractors/DiamondCameraExtractor.h"
#include "vicon_calibration/utils.h"

namespace vicon_calibration {

// edited
void DiamondCameraExtractor::ExtractKeypoints(
    Eigen::Matrix4d &T_CAMERA_TARGET_EST, cv::Mat &img_in) {
  // initialize member variables
  image_in_ = std::make_shared<cv::Mat>();
  image_undistorted_ = std::make_shared<cv::Mat>();
  image_cropped_ = std::make_shared<cv::Mat>();
  image_annotated_ = std::make_shared<cv::Mat>();
  T_CAMERA_TARGET_EST_ = T_CAMERA_TARGET_EST;
  *image_in_ = img_in;
  measurement_valid_ = true;
  measurement_complete_ = false;
  this->CheckInputs();

  // first check if target is in FOV of camera:
  Eigen::Vector4d point_origin{0, 0, 0, 1};
  Eigen::Vector2d pixel_projected = TargetPointToPixel(point_origin);
  if (!camera_model_->PixelInImage(pixel_projected)) {
    measurement_complete_ = true;
    measurement_valid_ = false;
    return;
  }

  this->UndistortImage();
  if (!this->CropImage()) {
    return;
  }
  *image_annotated_ = utils::DrawCoordinateFrame(
      *image_cropped_, T_CAMERA_TARGET_EST_, camera_model_, axis_plot_scale_,
      camera_params_->images_distorted);

  this->GetMeasurementPoints();

  if (!measurement_valid_) {
    measurement_complete_ = true;
    return;
  }
  measurement_complete_ = true;
  return;
}

// unedited
void DiamondCameraExtractor::CheckInputs() {
  if (!image_in_->data) {
    throw std::invalid_argument{"No image data"};
  }

  if (!utils::IsTransformationMatrix(T_CAMERA_TARGET_EST_)) {
    throw std::invalid_argument{"Invalid transform"};
  }

  if (!target_params_set_) {
    throw std::invalid_argument{"Target parameters not set."};
  }

  if (!camera_params_set_) {
    throw std::invalid_argument{"Camera parameters not set."};
  }
}

// unedited
void DiamondCameraExtractor::DisplayImage(cv::Mat &img,
                                          std::string display_name,
                                          std::string output_text) {
  if (show_measurements_) {
    std::cout << output_text << std::endl
              << "Press [c] to continue with other measurements\n";
    cv::namedWindow(display_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(display_name, img.cols / 2, img.rows / 2);
    cv::imshow(display_name, img);
    auto key = cv::waitKey();
    while (key != 67 && key != 99) {
      key = cv::waitKey();
    }
    cv::destroyAllWindows();
    cv::imshow(display_name, img);
    while (key != 67 && key != 99) {
      key = cv::waitKey();
    }
    cv::destroyAllWindows();
  }
}

// unedited
void DiamondCameraExtractor::UndistortImage() {
  if (camera_params_->images_distorted) {
    *image_undistorted_ = camera_model_->UndistortImage(*image_in_);
  } else {
    image_undistorted_ = image_in_;
  }
}

// edited - we should apply this to all methods though
bool DiamondCameraExtractor::CropImage() {
  // project taget points to image
  Eigen::Vector4d point_target(0, 0, 0, 1);
  std::vector<double> u, v;
  Eigen::Vector2d pixel;
  int iter = 0;
  while (iter < target_params_->template_cloud->size()) {
    point_target[0] = target_params_->template_cloud->at(iter).x;
    point_target[1] = target_params_->template_cloud->at(iter).y;
    point_target[2] = target_params_->template_cloud->at(iter).z;
    pixel = this->TargetPointToPixel(point_target);
    u.push_back(pixel[0]);
    v.push_back(pixel[1]);
    iter = iter + 5;
    // iter++;
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
  double buffer_u = (maxu - minu) * target_params_->crop_image(0, 0) / 100 / 2;
  double buffer_v = (maxv - minv) * target_params_->crop_image(1, 0) / 100 / 2;
  Eigen::Vector2d min_vec_buffer(minu - buffer_u, minv - buffer_v);
  Eigen::Vector2d max_vec_buffer(maxu + buffer_u, maxv + buffer_v);

  if (!camera_model_->PixelInImage(min_vec) ||
      !camera_model_->PixelInImage(max_vec)) {
    target_in_image = false;
  }
  if (!camera_model_->PixelInImage(min_vec_buffer) ||
      !camera_model_->PixelInImage(max_vec_buffer)) {
    cropbox_in_image = false;
  }

  if (!target_in_image) {
    if (show_measurements_) {
      LOG_WARN("Target not in image, skipping measurement.");
      LOG_INFO("Target corners: [minu, minv, maxu, maxv]: [%d, %d, %d, %d]",
               minu, minv, maxu, maxv);
      LOG_INFO("Image Dimensions: [%d x %d]", camera_model_->GetWidth(),
               camera_model_->GetHeight());
      cv::Mat current_image_w_axes = utils::DrawCoordinateFrame(
          *image_undistorted_, T_CAMERA_TARGET_EST_, camera_model_,
          axis_plot_scale_, camera_params_->images_distorted);
      this->DisplayImage(current_image_w_axes, "Invalid Measurement",
                         "Showing failed measurement (target not in image)");
    }
    measurement_valid_ = false;
    measurement_complete_ = true;
  } else if (!cropbox_in_image) {
    LOG_WARN("Target in image, but cropbox is not, you may want to relax your "
             "crop threshold. Skipping measurement.");
    if (show_measurements_) {
      LOG_INFO("Target corners: [minu, minv, maxu, maxv]: [%d, %d, %d, %d]",
               min_vec[0], min_vec[1], max_vec[0], max_vec[1]);
      LOG_INFO("Cropbox corners: [minu, minv, maxu, maxv]: [%d, %d, %d, %d]",
               min_vec_buffer[0], min_vec_buffer[1], max_vec_buffer[0],
               max_vec_buffer[1]);
      LOG_INFO("Image Dimensions: [%d x %d]", camera_model_->GetWidth(),
               camera_model_->GetHeight());
      cv::Mat current_image_w_axes = utils::DrawCoordinateFrame(
          *image_undistorted_, T_CAMERA_TARGET_EST_, camera_model_,
          axis_plot_scale_, camera_params_->images_distorted);
      this->DisplayImage(current_image_w_axes, "Invalid Measurement",
                         "Showing failed measurement (cropbox not in image)");
    }
    measurement_valid_ = false;
    measurement_complete_ = true;
  } else {
    if (show_measurements_) {
      LOG_INFO("Estimated target + cropbox in image.");
    }
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

// edited but will work for both
Eigen::Vector2d
DiamondCameraExtractor::TargetPointToPixel(Eigen::Vector4d point) {
  Eigen::Vector3d transformed_point;
  Eigen::Vector4d homogeneous_point;

  homogeneous_point = T_CAMERA_TARGET_EST_ * point;

  transformed_point << homogeneous_point[0], homogeneous_point[1],
      homogeneous_point[2];

  if (camera_params_->images_distorted) {
    return camera_model_->ProjectPoint(transformed_point);
  } else {
    return camera_model_->ProjectUndistortedPoint(transformed_point);
  }
}

// edited
void DiamondCameraExtractor::GetMeasurementPoints() {

  // find dimensions of checkerboard with [n_dim x m_dim] interior corners
  int n_dim = 0, m_dim = 0;
  double xmax = 0, ymax = 0;
  for (Eigen::Vector3d keypoint : target_params_->keypoints_camera) {
    if (keypoint[0] > xmax) {
      xmax = keypoint[0];
      m_dim++;
    }
    if (keypoint[1] > ymax) {
      ymax = keypoint[1];
      n_dim++;
    }
  }

  // Find checkerboard corners
  std::vector<cv::Point2f> corners;
  bool checkerboard_found = cv::findChessboardCorners(
      *image_cropped_, cv::Size(n_dim, m_dim), corners);

  // check if valid
  if (!checkerboard_found) {
    LOG_INFO("OpenCV could not detect a checkerboard of size: n = %d, m = %d",
             n_dim, m_dim);
    measurement_valid_ = false;
    this->DisplayImage(
        *image_annotated_, "Invalid Measurement",
        "Showing failed measurement (OpenCV could not find checkerboard)");
    return;
  }

  // convert to pcl point cloud
  keypoints_measured_->points.clear();
  for (uint32_t i = 0; i < corners.size(); i++) {
    pcl::PointXY pixel;
    pixel.x = corners[i].x;
    pixel.y = corners[i].y;
    keypoints_measured_->points.push_back(pixel);
  }
  drawChessboardCorners(*image_annotated_, cv::Size(n_dim, m_dim), corners,
                        checkerboard_found);
  this->DisplayImage(*image_annotated_, "Valid Measurement",
                     "Showing passed measurement");
}

// unused
// void DiamondCameraExtractor::GetMeasuredPose()

// unused
// void DiamondCameraExtractor::DrawContourAxis()

// unused
// void DiamondCameraExtractor::GetEstimatedPose()

// unused
// void DiamondCameraExtractor::CheckError()

} // namespace vicon_calibration
