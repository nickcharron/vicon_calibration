#include "vicon_calibration/ViconCalibrator.h"

#include <gflags/gflags.h>

DEFINE_string(
    calibration_config, "",
    "Full path to main config for calibrator. If empty, it will retrieve the "
    "file from .../vicon_calibration/config/ViconCalibratorConfig.json");
DEFINE_bool(show_camera_measurements, false,
            "Set to true to show and accept/reject each lidar measurement.");
DEFINE_bool(show_lidar_measurements, false,
            "Set to true to show and accept/reject each lidar measurement.");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  vicon_calibration::ViconCalibrator calibrator;
  calibrator.RunCalibration(FLAGS_calibration_config,
                            FLAGS_show_lidar_measurements,
                            FLAGS_show_camera_measurements);
  return 0;
}
