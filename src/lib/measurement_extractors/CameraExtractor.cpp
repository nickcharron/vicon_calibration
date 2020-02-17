#include "vicon_calibration/measurement_extractors/CameraExtractor.h"
#include "vicon_calibration/utils.h"
#include <boost/make_shared.hpp>

namespace vicon_calibration {

CameraExtractor::CameraExtractor() {
  keypoints_measured_ = boost::make_shared<pcl::PointCloud<pcl::PointXY>>();
}

void CameraExtractor::SetCameraParams(
    std::shared_ptr<vicon_calibration::CameraParams> &camera_params) {
  camera_params_ = camera_params;
  camera_params_set_ = true;
}

void CameraExtractor::SetTargetParams(
    std::shared_ptr<vicon_calibration::TargetParams> &target_params) {
  target_params_ = target_params;
  target_params_set_ = true;
}

void CameraExtractor::SetShowMeasurements(const bool &show_measurements) {
  show_measurements_ = show_measurements;
}

bool CameraExtractor::GetShowMeasurements() {
  return show_measurements_;
}

void CameraExtractor::ProcessMeasurement(
    const Eigen::Matrix4d &T_CAMERA_TARGET_EST, const cv::Mat &img_in) {
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
  Eigen::Vector2d pixel_projected = this->TargetPointToPixel(point_origin);
  if (!camera_params_->camera_model->PixelInImage(pixel_projected)) {
    measurement_complete_ = true;
    measurement_valid_ = false;
    return;
  }

  this->UndistortImage();
  this->CropImage();

  if (!measurement_valid_) {
    measurement_complete_ = true;
    return;
  }

  *image_annotated_ = utils::DrawCoordinateFrame(
      *image_cropped_, T_CAMERA_TARGET_EST_, camera_params_->camera_model, axis_plot_scale_,
      camera_params_->images_distorted);

  this->GetKeypoints();
  if (show_measurements_) {
  }

  if (!measurement_valid_ && show_measurements_) {
    this->DisplayImage(*image_annotated_, "Inalid Measurement",
                       "Showing failed measurement");
  } else if (measurement_valid_ && show_measurements_) {
    this->DisplayImage(*image_annotated_, "Valid Measurement",
                       "Showing successfull measurement");
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

Eigen::Vector2d
CameraExtractor::TargetPointToPixel(const Eigen::Vector4d &point) {
  Eigen::Vector3d transformed_point;
  Eigen::Vector4d homogeneous_point;

  homogeneous_point = T_CAMERA_TARGET_EST_ * point;

  transformed_point << homogeneous_point[0], homogeneous_point[1],
      homogeneous_point[2];

  if (camera_params_->images_distorted) {
    return camera_params_->camera_model->ProjectPoint(transformed_point);
  } else {
    return camera_params_->camera_model->ProjectUndistortedPoint(transformed_point);
  }
}

void CameraExtractor::CropImage() {
  // project taget points to image and get bounding box
  Eigen::Vector4d point_target(0, 0, 0, 1);
  Eigen::Vector2d pixel;
  int iter = 0;
  double maxu{0}, maxv{0}, minu{1000000}, minv{1000000};
  while (iter < target_params_->template_cloud->size()) {
    point_target[0] = target_params_->template_cloud->at(iter).x;
    point_target[1] = target_params_->template_cloud->at(iter).y;
    point_target[2] = target_params_->template_cloud->at(iter).z;
    pixel = this->TargetPointToPixel(point_target);
    if (pixel[0] > maxu) {
      maxu = pixel[0];
    }
    if (pixel[0] < minu) {
      minu = pixel[0];
    }
    if (pixel[1] > maxv) {
      maxv = pixel[1];
    }
    if (pixel[1] < minv) {
      minv = pixel[1];
    }
    iter = iter + 5;
    // iter++;
  }

  // Get cropbox corners
  Eigen::Vector2d min_vec(minu, minv);
  Eigen::Vector2d max_vec(maxu, maxv);
  double buffer_u = (maxu - minu) * target_params_->crop_image(0, 0) / 100 / 2;
  double buffer_v = (maxv - minv) * target_params_->crop_image(1, 0) / 100 / 2;
  Eigen::Vector2d min_vec_buffer(minu - buffer_u, minv - buffer_v);
  Eigen::Vector2d max_vec_buffer(maxu + buffer_u, maxv + buffer_v);

  // determine if target and/or cropbox is in the image
  bool cropbox_in_image = true;
  bool target_in_image = true;
  if (!camera_params_->camera_model->PixelInImage(min_vec) ||
      !camera_params_->camera_model->PixelInImage(max_vec)) {
    target_in_image = false;
  }
  if (!camera_params_->camera_model->PixelInImage(min_vec_buffer) ||
      !camera_params_->camera_model->PixelInImage(max_vec_buffer)) {
    cropbox_in_image = false;
  }

  // output results and allow the user to accept/decline
  if (!cropbox_in_image) {
    measurement_valid_ = false;
    measurement_complete_ = true;
    if (target_in_image && show_measurements_) {
      LOG_WARN(
          "Target in image, but cropbox is not, you may want to relax your "
          "crop threshold. Skipping measurement.");
    } else if (!target_in_image && show_measurements_) {
      LOG_WARN("Target and cropbox not in image, skipping measurement.");
    }

    if (show_measurements_) {
      LOG_INFO("Target corners: [minu, minv, maxu, maxv]: [%d, %d, %d, %d]",
               min_vec[0], min_vec[1], max_vec[0], max_vec[1]);
      LOG_INFO("Cropbox corners: [minu, minv, maxu, maxv]: [%d, %d, %d, %d]",
               min_vec_buffer[0], min_vec_buffer[1], max_vec_buffer[0],
               max_vec_buffer[1]);
      LOG_INFO("Image Dimensions: [%d x %d]", camera_params_->camera_model->GetWidth(),
               camera_params_->camera_model->GetHeight());
      this->DisplayImage(*image_undistorted_, "Invalid Measurement",
                         "Showing failed measurement (cropbox not in image)");
    }
  } else {
    measurement_valid_ = true;
    measurement_complete_ = true;
    if (show_measurements_) {
      LOG_INFO("Estimated target cropbox in image.");
    }
  }

  // create cropped image
  if (measurement_valid_ && cropbox_in_image) {
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
  } else if(measurement_valid_){
    image_cropped_ = image_undistorted_;
  }
  return;
}

void CameraExtractor::UndistortImage() {
  if (camera_params_->images_distorted) {
    *image_undistorted_ = camera_params_->camera_model->UndistortImage(*image_in_);
  } else {
    image_undistorted_ = image_in_;
  }
}

void CameraExtractor::DisplayImage(const cv::Mat &img,
                                   const std::string &display_name,
                                   const std::string &output_text) {
  if (show_measurements_) {
    cv::Mat current_image_w_axes = utils::DrawCoordinateFrame(
        img, T_CAMERA_TARGET_EST_, camera_params_->camera_model, axis_plot_scale_,
        camera_params_->images_distorted);
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
    cv::destroyAllWindows();
  }
}

} // namespace vicon_calibration
