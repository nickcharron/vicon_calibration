#include "vicon_calibration/CamCylExtractor.h"

namespace vicon_calibration {

CamCylExtractor::CamCylExtractor() { PopulateCylinderPoints(); }

void CamCylExtractor::ConfigureCameraModel(std::string intrinsic_file) {
  camera_model_ = beam_calibration::CameraModel::LoadJSON(intrinsic_file);
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
    point(2) = radius_ * radius_ * cos(theta) * cos(theta);
    if (point(2) < 0) {
      point(2) -= threshold_;
    } else {
      point(2) += threshold_;
    }

    point(1) = radius_ * radius_ * sin(theta) * sin(theta);
    if (point(1) < 0) {
      point(1) -= threshold_;
    } else {
      point(1) += threshold_;
    }

    point(0) = -threshold_;
    cylinder_points_.push_back(point);

    point(0) = height_ + threshold_;
    cylinder_points_.push_back(point);
  }
}

void CamCylExtractor::ExtractCylinder(Eigen::Affine3d T_CAMERA_TARGET_EST,
                                      cv::Mat image, int measurement_num) {
  auto min_max_vectors = GetBoundingBox(T_CAMERA_TARGET_EST);
  auto cropped_image = CropImage(image, min_max_vectors[0], min_max_vectors[1]);

  image = ColorPixelsOnImage(image);

  cv::imshow("Original image " + std::to_string(measurement_num), image);
  cv::imshow("Cropped image " + std::to_string(measurement_num), cropped_image);
  cv::waitKey(0);

  cv::destroyAllWindows();

  measurement_complete_ = true;
  measurement_valid_ = true;
}

std::vector<Eigen::Vector2d>
CamCylExtractor::GetBoundingBox(Eigen::Affine3d T_CAMERA_TARGET_EST) {
  Eigen::Vector4d transformed_point;
  std::vector<double> u, v;

  projected_pixels_.clear();

  for (int i = 0; i < cylinder_points_.size(); i++) {
    transformed_point = T_CAMERA_TARGET_EST.matrix() * cylinder_points_[i];
    std::cout << "*************POINT************* " << std::endl << transformed_point << std::endl;
    auto pixel = camera_model_->ProjectPoint(transformed_point);
    std::cout << "*************PIXEL************* " << std::endl << pixel << std::endl;
    u.push_back(pixel(0));
    v.push_back(pixel(1));
    projected_pixels_.push_back(pixel);
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

  if (!CheckPixelWithinRange(min_vector) ||
      !CheckPixelWithinRange(max_vector)) {
    return image;
  }

  cv::Rect region_of_interest(min_vector(0), min_vector(1), width, height);
  cv::Mat cropped_image = image(region_of_interest);

  return cropped_image;
}

bool CamCylExtractor::CheckPixelWithinRange(Eigen::Vector2d pixel) {
  if (pixel(0) < 0 || pixel(1) < 0 || pixel(0) > camera_model_->GetWidth() ||
      pixel(1) > camera_model_->GetHeight()) {
    return false;
  } else {
    return true;
  }
}
cv::Mat CamCylExtractor::ColorPixelsOnImage(cv::Mat &img) {
  for(auto pixel : projected_pixels_) {
    if(CheckPixelWithinRange(pixel)) {
      cv::circle(img, cv::Point(pixel(0), pixel(1)),
                 2, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);
    }
  }
  return img;
}

} // end namespace vicon_calibration
