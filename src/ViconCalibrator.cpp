#include "vicon_calibration/CamCylExtractor.h"
#include "vicon_calibration/GTSAMGraph.h"
#include "vicon_calibration/LidarCylExtractor.h"
#include <string>

std::string bage_file, initial_calibration;
std::vector<std::string> image_topics, image_frames, intrinsics, lidar_topics,
                    lidar_frames;

std::string GetJSONFileName(std::string file_name) {
  std::string file_location = __FILE__;
  file_location.erase(file_location.end() - 23, file_location.end());
  file_location += "data/";
  file_location += file_name;
  return file_location;
}

void LoadJson(std::string file_name){
  nlohmann::json J;
  std::ifstream file(file_name);
  file >> J;

  bage_file = J["bage_file"];
  initial_calibration = J["initial_calibration"];

  for (const auto &topic : J["image_topics"]) {
    image_topics.push_back(topic.get<std::string>());
  }

  for (const auto &topic : J["image_frames"]) {
    image_frames.push_back(topic.get<std::string>());
  }

  for (const auto &topic : J["intrinsics"]) {
    intrinsics.push_back(topic.get<std::string>());
  }

  for (const auto &topic : J["lidar_topics"]) {
    lidar_topics.push_back(topic.get<std::string>());
  }

  for (const auto &topic : J["lidar_frames"]) {
    lidar_frames.push_back(topic.get<std::string>());
  }

}

int main() {
  std::string config_file;
  config_file = GetJSONFileName("ViconCalibrationConfig.json");
  LoadJson(config_file);
  return 0;
}
