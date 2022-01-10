#include <gflags/gflags.h>

#include <vicon_calibration/Gflags.h>
#include <vicon_calibration/ViconCalibrator.h>

std::string config_default_path_ =
    vicon_calibration::utils::GetFilePathConfig("");
std::string data_default_path_ = vicon_calibration::utils::GetFilePathData("");

DEFINE_string(bag, "", "Full path to bag file (Required).");
DEFINE_validator(bag, &vicon_calibration::gflags::ValidateBagFileMustExist);
DEFINE_string(initial_calibration, "",
              "Full path to initial calibration config json. If empty, it will "
              "lookup the calbration from the /tf and /tf_static topics.");
DEFINE_validator(initial_calibration,
                 &vicon_calibration::gflags::ValidateJsonFileMustExistOrEmpty);
DEFINE_string(calibration_config,
              config_default_path_ + "ViconCalibratorConfig.json",
              "Full path to main config for calibrator.");
DEFINE_validator(calibration_config,
                 &vicon_calibration::gflags::ValidateJsonFileMustExist);
DEFINE_string(optimizer_config, config_default_path_ + "OptimizerConfig.json",
              "Full path to optimizer config json. If left empty, it will use OptimizerConfig.json in the default config path.");
DEFINE_validator(optimizer_config,
                 &vicon_calibration::gflags::ValidateJsonFileMustExist);
DEFINE_string(ceres_config, config_default_path_ + "CeresConfig.json",
              "Full path to optimizer config json. If left empty, it will use OptimizerConfig.json in the default config path.");
DEFINE_validator(ceres_config,
                 &vicon_calibration::gflags::ValidateJsonFileMustExist);                 
DEFINE_string(verification_config,
              config_default_path_ + "CalibrationVerification.json",
              "Full path to verification config json. If set to NONE, it will "
              "not run the verification.");
DEFINE_validator(verification_config,
                 &vicon_calibration::gflags::ValidateJsonFileMustExistOrNONE);
DEFINE_string(target_config_path, config_default_path_,
              "Full path to directory containing target config json files.");
DEFINE_validator(target_config_path,
                 &vicon_calibration::gflags::ValidateDirMustExist);
DEFINE_string(
    target_data_path, data_default_path_,
    "Full path to directory containing target data (i.e., template clouds).");
DEFINE_validator(target_data_path,
                 &vicon_calibration::gflags::ValidateDirMustExist);
DEFINE_string(camera_intrinsics_path, data_default_path_,
              "Full path to directory camera intrinsics json files.");
DEFINE_validator(camera_intrinsics_path,
                 &vicon_calibration::gflags::ValidateDirMustExist);
DEFINE_string(output_directory, "/tmp/", "Full path to output directory.");
DEFINE_validator(output_directory,
                 &vicon_calibration::gflags::ValidateDirMustExist);
DEFINE_bool(show_camera_measurements, false,
            "Set to true to show and accept/reject each camera measurement.");
DEFINE_bool(show_lidar_measurements, false,
            "Set to true to show and accept/reject each lidar measurement.");

int main(int argc, char** argv) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);

  vicon_calibration::CalibratorInputs inputs{
      .bag = FLAGS_bag,
      .initial_calibration = FLAGS_initial_calibration,
      .calibration_config = FLAGS_calibration_config,
      .optimizer_config = FLAGS_optimizer_config,
      .ceres_config = FLAGS_ceres_config,
      .target_config_path = FLAGS_target_config_path,
      .target_data_path = FLAGS_target_data_path,
      .camera_intrinsics_path = FLAGS_camera_intrinsics_path,
      .verification_config = FLAGS_verification_config,
      .output_directory = FLAGS_output_directory,
      .show_camera_measurements = FLAGS_show_camera_measurements,
      .show_lidar_measurements = FLAGS_show_lidar_measurements};

  vicon_calibration::ViconCalibrator calibrator(inputs);
  calibrator.RunCalibration();
  return 0;
}
