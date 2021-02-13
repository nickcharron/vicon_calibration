#pragma once

#include <nlohmann/json.hpp>

#include <vicon_calibration/Params.h>

namespace vicon_calibration {

class JsonTools {

public:
  JsonTools(const CalibratorInputs& inputs);

  ~JsonTools() = default;

  std::shared_ptr<TargetParams> LoadTargetParams(const std::string& target_config_full_path);

  std::shared_ptr<TargetParams> LoadTargetParams(const nlohmann::json &J_in);

  std::shared_ptr<CameraParams> LoadCameraParams(const nlohmann::json &J_in);

  std::shared_ptr<LidarParams> LoadLidarParams(const nlohmann::json &J_in);

  std::shared_ptr<CalibratorConfig> LoadViconCalibratorParams();

private:
  const CalibratorInputs inputs_;
};

} // end namespace vicon_calibration
