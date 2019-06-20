#include "vicon_calibration/CamCylExtractor.h"

namespace vicon_calibration {

CamCylExtractor::CamCylExtractor() { PopulateCylinderPoints(); }

void CamCylExtractor::ConfigureCameraModel(std::string intrinsic_file) {
  LOG_INFO("Loading file: %s", intrinsic_file.c_str());

  json J;
  int calibration_counter = 0, value_counter = 0;
  std::ifstream file(intrinsic_file);
  file >> J;

  for (const auto& calibration : J["calibration"]) {
    calibration_counter++;
    value_counter = 0;
    int i = 0, j = 0;

    for (const auto& value : calibration["camera_matrix"]) {
      value_counter++;
      K_(i, j) = value.get<double>();
      if (j == 2) {
        i++;
        j = 0;
      } else {
        j++;
      }
    }
    if (value_counter != 10) {
      LOG_ERROR("Invalid transform matrix in .json file.");
      throw std::invalid_argument{"Invalid transform matrix in .json file."};
      return;
    }
  }
  //camera_model_ = beam_calibration::CameraModel::LoadJSON(intrinsic_file);
}

void CamCylExtractor::SetCylinderDimension(double radius, double height) {
  radius_ = radius;
  height_ = height;

  PopulateCylinderPoints();
}

void CamCylExtractor::SetThreshold(double threshold) { threshold_ = threshold; }

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
  // Populate cylinder_points_ with offsetted points for bigger bounding box
  Eigen::Vector4d point(0, 0, 0, 1);
  for (double theta = 0; theta < M_PI; theta += M_PI / 12) {
    point(0) = radius_ * radius_ * cos(theta) * cos(theta);
    if (point(0) < 0) {
      point(0) -= threshold_;
    } else {
      point(0) += threshold_;
    }

    point(1) = radius_ * radius_ * sin(theta) * sin(theta);
    if (point(1) < 0) {
      point(1) -= threshold_;
    } else {
      point(1) += threshold_;
    }

    point(2) = threshold_;
    cylinder_points_.push_back(point);

    point(2) = -height_ - threshold_;
    cylinder_points_.push_back(point);
  }
}

void CamCylExtractor::ExtractCylinder(Eigen::Affine3d T_CAMERA_TARGET_EST,
                                      cv::Mat image) {
  auto min_max_vectors = GetBoundingBox(T_CAMERA_TARGET_EST);

  auto cropped_image = CropImage(image, min_max_vectors[0], min_max_vectors[1]);

  cv::imshow("Cropped image", cropped_image);
  cv::waitKey(0);

  measurement_complete_ = true;
  measurement_valid_ = true;
}

std::vector<Eigen::Vector2d>
CamCylExtractor::GetBoundingBox(Eigen::Affine3d T_CAMERA_TARGET_EST) {
  Eigen::MatrixXd rotation_and_translation(3,4);
  rotation_and_translation.block<3,3>(0,0) = T_CAMERA_TARGET_EST.rotation();
  rotation_and_translation.block<3,1>(0,3) = T_CAMERA_TARGET_EST.translation();

  Eigen::Vector4d transformed_point;
  Eigen::Vector2d min_vec(camera_model_->GetWidth(),
                          camera_model_->GetHeight());
  Eigen::Vector2d max_vec(0, 0);

  for (int i = 0; i < cylinder_points_.size(); i++) {
    transformed_point = K_ * rotation_and_translation * cylinder_points_[i];
    auto pixel = camera_model_->ProjectPoint(transformed_point);
    if (pixel(0) < min_vec(0))
      min_vec(0) = pixel(0);
    if (pixel(1) < min_vec(1))
      min_vec(1) = pixel(1);
    if (pixel(0) > max_vec(0))
      max_vec(0) = pixel(0);
    if (pixel(1) > max_vec(1))
      max_vec(1) = pixel(1);
  }

  return std::vector<Eigen::Vector2d>{min_vec, max_vec};
}

cv::Mat CamCylExtractor::CropImage(cv::Mat image, Eigen::Vector2d min_vector,
                                   Eigen::Vector2d max_vector) {
  std::cout << "Min vec: " << min_vector << std::endl
            << "Max vec: " << max_vector << std::endl;
  double width = max_vector(0) - min_vector(0);
  double height = max_vector(1) - min_vector(1);
  if (min_vector(0) < 0 || min_vector(1) < 0 ||
      min_vector(0) > camera_model_->GetWidth() ||
      min_vector(1) > camera_model_->GetHeight() ||
      max_vector(0) < 0 || max_vector(1) < 0 ||
      max_vector(0) > camera_model_->GetWidth() ||
      max_vector(1) > camera_model_->GetHeight())
      {
        return image;
      }

  cv::Rect region_of_interest(min_vector(0), min_vector(1), width, height);
  cv::Mat cropped_image = image(region_of_interest);

  return cropped_image;
}

} // end namespace vicon_calibration
