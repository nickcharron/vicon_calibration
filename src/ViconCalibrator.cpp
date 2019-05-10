#include "vicon_calibration/CamCylExtractor.h"
#include "vicon_calibration/GTSAMGraph.h"
#include "vicon_calibration/LidarCylExtractor.h"
#include <string>

std::string getJSONFileName(std::string file_name) {
  std::string file_location = __FILE__;
  file_location.erase(file_location.end() - 23, file_location.end());
  file_location += "data/";
  file_location += file_name;
  return file_location;
}

int main() {

  std::string config_file;
  config_file = getJSONFileName("ViconCalibrationConfig.json");

  return 0;
}
