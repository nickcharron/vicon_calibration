#include <vicon_calibration/measurement_extractors/CameraExtractor.h>

#include <boost/make_shared.hpp>

#include <vicon_calibration/Utils.h>

namespace vicon_calibration {

CameraExtractor::CameraExtractor() {
  keypoints_measured_ = std::make_shared<pcl::PointCloud<pcl::PointXY>>();
}

void CameraExtractor::SetCameraParams(
    std::shared_ptr<vicon_calibration::CameraParams>& camera_params) {
  camera_params_ = camera_params;
  camera_params_set_ = true;
}

void CameraExtractor::SetTargetParams(
    std::shared_ptr<vicon_calibration::TargetParams>& target_params) {
  target_params_ = target_params;
  target_params_set_ = true;
}

void CameraExtractor::SetShowMeasurements(const bool& show_measurements) {
  show_measurements_ = show_measurements;
}

bool CameraExtractor::GetShowMeasurements() {
  return show_measurements_;
}

void CameraExtractor::ProcessMeasurement(
    const Eigen::Matrix4d& T_CAMERA_TARGET_EST, const cv::Mat& img_in) {
  // initialize member variables
  image_in_ = std::make_shared<cv::Mat>();
  T_CAMERA_TARGET_EST_ = T_CAMERA_TARGET_EST;
  *image_in_ = img_in;
  measurement_valid_ = true;
  measurement_complete_ = false;
  this->CheckInputs();

  // first check if target is in FOV of camera:
  Eigen::Vector4d point_origin{0, 0, 0, 1};
  opt<Eigen::Vector2d> pixel_projected = this->TargetPointToPixel(point_origin);

  if (!pixel_projected.has_value()) {
    measurement_complete_ = true;
    measurement_valid_ = false;
    return;
  }

  if (show_measurements_) {
    image_annotated_ = std::make_shared<cv::Mat>();
    *image_annotated_ = *image_in_;
  }

  this->GetKeypoints();

  if (show_measurements_) {
    *image_annotated_ = utils::DrawCoordinateFrame(
        *image_annotated_, T_CAMERA_TARGET_EST_, camera_params_->camera_model,
        axis_plot_scale_);
    if (!measurement_valid_) {
      this->DisplayImage(*image_annotated_, "Inalid Measurement",
                         "Showing failed measurement", false);
    } else {
      this->DisplayImage(*image_annotated_, "Valid Measurement",
                         "Showing successfull measurement", true);
    }
  }
  measurement_complete_ = true;
  return;
}

bool CameraExtractor::GetMeasurementValid() {
  if (!measurement_complete_) {
    throw std::invalid_argument{
        "Cannot retrieve measurement, please run ExtractKeypoints before "
        "attempting to retrieve measurement."};
  }
  return measurement_valid_;
}

pcl::PointCloud<pcl::PointXY>::Ptr CameraExtractor::GetMeasurement() {
  if (!measurement_complete_) {
    throw std::invalid_argument{
        "Cannot retrieve measurement, please run ExtractKeypoints before "
        "attempting to retrieve measurement."};
  }
  return keypoints_measured_;
}

void CameraExtractor::CheckInputs() {
  if (!image_in_->data) { throw std::invalid_argument{"No image data"}; }

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

opt<Eigen::Vector2d>
    CameraExtractor::TargetPointToPixel(const Eigen::Vector4d& point) {
  Eigen::Vector4d transformed_point = T_CAMERA_TARGET_EST_ * point;
  return camera_params_->camera_model->ProjectPointPrecise(
          transformed_point.hnormalized());
  // opt<Eigen::Vector2d> tmp =
  //     camera_params_->camera_model->ProjectPointPrecise(
  //         transformed_point.hnormalized());
  // if (tmp.has_value()) {
  //   return opt<Eigen::Vector2d>(tmp.value());
  // } else {
  //   return {};
  // }
}

void CameraExtractor::CropImage() {
  image_cropped_ = std::make_shared<cv::Mat>();

  // project taget points to image and get bounding box
  int iter = 0;
  double maxu{0}, maxv{0}, minu{std::numeric_limits<double>::max()},
      minv{std::numeric_limits<double>::max()};
  while (iter < target_params_->template_cloud->size()) {
    Eigen::Vector4d point_target{target_params_->template_cloud->at(iter).x,
                                 target_params_->template_cloud->at(iter).y,
                                 target_params_->template_cloud->at(iter).z, 1};
    opt<Eigen::Vector2d> pixel = this->TargetPointToPixel(point_target);
    iter = iter + 5;
    if (!pixel.has_value()) { continue; }
    if (pixel.value()[0] > maxu) { maxu = pixel.value()[0]; }
    if (pixel.value()[0] < minu) { minu = pixel.value()[0]; }
    if (pixel.value()[1] > maxv) { maxv = pixel.value()[1]; }
    if (pixel.value()[1] < minv) { minv = pixel.value()[1]; }
  }

  // Get cropbox corners
  Eigen::Vector2d min_vec(minu, minv);
  Eigen::Vector2d max_vec(maxu, maxv);
  double buffer_u = (maxu - minu) * target_params_->crop_image(0, 0) / 100 / 2;
  double buffer_v = (maxv - minv) * target_params_->crop_image(1, 0) / 100 / 2;
  Eigen::Vector2d min_vec_buffer(minu - buffer_u, minv - buffer_v);
  Eigen::Vector2d max_vec_buffer(maxu + buffer_u, maxv + buffer_v);

  // determine if target and/or cropbox is in the image
  bool target_in_image = true;
  if (!camera_params_->camera_model->PixelInImage(min_vec) ||
      !camera_params_->camera_model->PixelInImage(max_vec)) {
    target_in_image = false;
  }

  if (!camera_params_->camera_model->PixelInImage(min_vec_buffer)) {
    min_vec_buffer = Eigen::Vector2d::Zero();
  }
  if (!camera_params_->camera_model->PixelInImage(max_vec_buffer)) {
    max_vec_buffer = Eigen::Vector2d(camera_params_->camera_model->GetWidth(),
                                     camera_params_->camera_model->GetHeight());
  }

  if (target_in_image) {
    measurement_valid_ = true;
  } else {
    measurement_valid_ = false;
    if (show_measurements_) {
      LOG_WARN("Target not in image, skipping measurement.");
    }
  }

  measurement_complete_ = true;

  // create cropped image
  if (measurement_valid_) {
    cv::Mat bounded_img(image_in_->rows, image_in_->cols, image_in_->depth());
    double width = max_vec_buffer(0) - min_vec_buffer(0);
    double height = max_vec_buffer(1) - min_vec_buffer(1);
    cv::Mat mask =
        cv::Mat::zeros(image_in_->rows, image_in_->cols, image_in_->depth());
    mask(cv::Rect(min_vec_buffer(0), min_vec_buffer(1), width, height)) = 1;
    image_in_->copyTo(bounded_img, mask);
    *image_cropped_ = bounded_img;
  } else {
    *image_cropped_ = *image_in_;
  }
  return;
}

void CameraExtractor::DisplayImage(const cv::Mat& img,
                                   const std::string& display_name,
                                   const std::string& output_text,
                                   const bool& allow_override) {
  if (!show_measurements_) { return; }

  cv::Mat current_image_w_axes = utils::DrawCoordinateFrame(
      img, T_CAMERA_TARGET_EST_, camera_params_->camera_model,
      axis_plot_scale_);

  if (allow_override) {
    std::cout << output_text << std::endl
              << "Press [c] to continue with default\n"
              << "Press [y] to accept measurement\n"
              << "Press [n] to reject measurement\n"
              << "Press [s] to stop showing future measurements\n";
    cv::namedWindow(display_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(display_name, img.cols / 2, img.rows / 2);
    cv::imshow(display_name, current_image_w_axes);
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
        this->SetShowMeasurements(false);
      }
    }
  } else {
    std::cout << output_text << std::endl
              << "Press [c] to continue with default\n"
              << "Press [s] to stop showing future measurements\n";
    cv::namedWindow(display_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(display_name, img.cols / 2, img.rows / 2);
    cv::imshow(display_name, current_image_w_axes);
    auto key = 0;
    while (key != 67 && key != 99 && key != 115) {
      key = cv::waitKey();
      if (key == 115) {
        std::cout << "setting show measurements to false.\n";
        this->SetShowMeasurements(false);
      }
    }
  }

  cv::destroyAllWindows();
}

} // namespace vicon_calibration
