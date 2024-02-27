#include <vicon_calibration/camera_models/CameraModels.h>

#include <chrono>
#include <ctime>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace vicon_calibration {

std::shared_ptr<CameraModel> CameraModel::Create(std::string& file_location) {
  std::shared_ptr<CameraModel> camera_model;

  // check file exists
  if (!std::filesystem::exists(file_location)) {
    LOG_ERROR("Invalid file path for camera intialization. Input: %s",
              file_location.c_str());
    throw std::runtime_error{"Invalid file path for camera intialization."};
  }

  std::filesystem::path p(file_location);
  if (p.extension().string() == ".json") {
    nlohmann::json J;
    std::ifstream file(file_location);
    file >> J;
    std::string camera_type = J["camera_type"];
    if (camera_type == "KANNALABRANDT") {
      camera_model = std::make_shared<KannalaBrandt>(file_location);
    } else if (camera_type == "DOUBLESPHERE") {
      camera_model = std::make_shared<DoubleSphere>(file_location);
    } else if (camera_type == "RADTAN") {
      camera_model = std::make_shared<Radtan>(file_location);
    } else if (camera_type == "CATADITROPIC") {
      camera_model = std::make_shared<Cataditropic>(file_location);
    } else {
      LOG_ERROR("Invalid camera type read from JSON.");
      throw std::runtime_error{"Invalid camera type read from JSON."};
    }
  } else {
    LOG_ERROR("Invalid file type read for camera intialization.");
    throw std::runtime_error{
        "Invalid file type read for camera intialization."};
  }
  
  return camera_model;
}

void CameraModel::CalculateFOV() {
  fov = 2 * atan(image_width_ / (2 * intrinsics_.at(0)));
}

void CameraModel::SetCameraID(const unsigned int id) {
  cam_id_ = id;
}

void CameraModel::SetFrameID(const std::string& id) {
  frame_id_ = id;
}

void CameraModel::SetCalibrationDate(const std::string& date) {
  calibration_date_ = date;
}

void CameraModel::SetImageDims(const uint32_t height, const uint32_t width) {
  image_width_ = width;
  image_height_ = height;
}

void CameraModel::SetIntrinsics(const std::vector<double>& intrinsics) {
  if (intrinsics.size() != intrinsics_size_[GetType()]) {
    LOG_ERROR("Invalid number of elements in intrinsics vector.");
    throw std::runtime_error{
        "Invalid number of elements in intrinsics vector."};
  } else {
    intrinsics_ = intrinsics;
  }
}

const std::string CameraModel::GetFrameID() const {
  return frame_id_;
}

const std::string CameraModel::GetCalibrationDate() const {
  if (calibration_date_ == "") { LOG_WARN("Calibration date empty."); }
  return calibration_date_;
}

uint32_t CameraModel::GetHeight() const {
  if (image_height_ == 0) { LOG_WARN("Image height not set."); }
  return image_height_;
}

uint32_t CameraModel::GetWidth() const {
  if (image_width_ == 0) { LOG_WARN("Image width not set."); }
  return image_width_;
}

float CameraModel::GetFOV() const {
  if (fov == 0) { LOG_WARN("Image fov not set."); }
  return fov;
}

const std::vector<double>& CameraModel::GetIntrinsics() const {
  return intrinsics_;
}

CameraType CameraModel::GetType() const {
  return type_;
}

bool CameraModel::PixelInImage(const Eigen::Vector2i& pixel) {
  if (pixel[0] < 0 || pixel[1] < 0 ||
      pixel[0] > static_cast<int>(image_width_ - 1) ||
      pixel[1] > static_cast<int>(image_height_ - 1))
    return false;
  return true;
}

bool CameraModel::PixelInImage(const Eigen::Vector2d& pixel) {
  if (pixel[0] < 0 || pixel[1] < 0 ||
      pixel[0] > static_cast<double>(image_width_ - 1) ||
      pixel[1] > static_cast<double>(image_height_ - 1))
    return false;
  return true;
}

void CameraModel::LoadJSON(const std::string& file_location) {
  LOG_INFO("Loading camera model config file: %s", file_location.c_str());

  // load file
  nlohmann::json J;
  std::ifstream file(file_location);
  file >> J;

  // get string repr of class type
  std::string class_type;
  for (std::map<std::string, CameraType>::iterator it =
           intrinsics_types_.begin();
       it != intrinsics_types_.end(); it++) {
    if (intrinsics_types_[it->first] == type_) { class_type = it->first; }
  }

  // check type
  std::string camera_type = J["camera_type"];
  std::map<std::string, CameraType>::iterator it =
      intrinsics_types_.find(camera_type);
  if (it == intrinsics_types_.end()) {
    LOG_ERROR("Invalid camera type read from json. Type read: %s",
              camera_type.c_str());
    OutputCameraTypes();
  } else if (intrinsics_types_[camera_type] != type_) {
    LOG_ERROR("Camera type read from JSON does not match expected type. "
              "Type read: %s, Expected: %s",
              camera_type.c_str(), class_type.c_str());
  }
  // get params
  calibration_date_ = J["date"];
  image_width_ = J["image_width"];
  image_height_ = J["image_height"];
  frame_id_ = J["frame_id"];
  intrinsics_.clear();
  for (const auto& value : J["intrinsics"]) {
    intrinsics_.push_back(value.get<double>());
  }

  if (intrinsics_.size() != intrinsics_size_[type_]) {
    LOG_ERROR("Invalid number of intrinsics read. read: %d, required: %d",
              static_cast<int>(intrinsics_.size()),
              static_cast<int>(intrinsics_size_[type_]));
    throw std::invalid_argument{"Invalid number of instrinsics read."};
  }
}

void CameraModel::WriteJSON(const std::string& file_location,
                            const std::string& method) {
  LOG_INFO("Writing to file: %s", file_location.c_str());

  std::time_t date =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::string cur_date = std::string(std::ctime(&date));
  // load file
  nlohmann::json J;
  J["date"] = cur_date;
  if (method.empty()) {
    J["method"] = std::string("unkown");
  } else {
    J["method"] = method;
  }
  // get string repr of class type
  std::string class_type;
  for (std::map<std::string, CameraType>::iterator it =
           intrinsics_types_.begin();
       it != intrinsics_types_.end(); it++) {
    if (intrinsics_types_[it->first] == type_) { class_type = it->first; }
  }
  J["camera_type"] = class_type;
  J["image_width"] = GetWidth();
  J["image_height"] = GetHeight();
  J["frame_id"] = GetFrameID();
  J["intrinsics"] = intrinsics_;
  std::ofstream out(file_location);
  out << std::setw(4) << J << std::endl;
}

void CameraModel::OutputCameraTypes() {
  std::cout << "Intrinsic type input options:\n";
  for (std::map<std::string, CameraType>::iterator it =
           intrinsics_types_.begin();
       it != intrinsics_types_.end(); it++) {
    std::cout << "    -" << it->first << "\n";
  }
}

} // namespace vicon_calibration
