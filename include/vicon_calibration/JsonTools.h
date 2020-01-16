#pragma once

#include "vicon_calibration/params.h"
#include <nlohmann/json.hpp>

namespace vicon_calibration {

class JsonTools {

public:
  /**
   * @brief default constructor
   */
  JsonTools() = default;

  ~JsonTools() = default;

  std::shared_ptr<TargetParams> LoadTargetParams(const std::string &file_name);

  std::shared_ptr<TargetParams> LoadTargetParams(const nlohmann::json &J_in);

  std::shared_ptr<CameraParams> LoadCameraParams(const nlohmann::json &J_in);

  std::shared_ptr<LidarParams> LoadLidarParams(const nlohmann::json &J_in);

  std::shared_ptr<CalibratorConfig> LoadViconCalibratorParams(const std::string &file_name);

private:
  std::string GetJSONFileNameConfig(const std::string &file_name);

  std::string GetJSONFileNameData(const std::string &file_name);

};

} // end namespace vicon_calibration
