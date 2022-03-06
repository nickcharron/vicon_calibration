/* This exectuable can be used to generate keypoints used for a checkerboard
  target that you will include in your TargetConfig.json config file (see
  CheckerboardTaret.json).

  **Note:** This assumes the coordinate frame is actually at the top left
  "exterior" edge, not the first interior edge which is usually the assumed
  coordinate frame for checkerboard based calibration. The reason for this is
  that for vicon calibration we need to add reflective markers to define the
  coordinate frame and we don't want these to interfere with the checkerboard
  detection.
*/

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <gflags/gflags.h>
#include <nlohmann/json.hpp>

#include <vicon_calibration/Gflags.h>
#include <vicon_calibration/Utils.h>

using namespace vicon_calibration;

DEFINE_int32(
    n_corners_vertical, 0,
    "Number of interior checkerboard corners in the vertical direction.");
DEFINE_int32(
    n_corners_horizontal, 0,
    "Number of interior checkerboard corners in the horizontal direction.");
DEFINE_double(square_size, 0, "Size of the checkerboard squares, in meters.");

DEFINE_string(output_directory, "", "Full path to output directory.");
DEFINE_validator(output_directory,
                 &vicon_calibration::gflags::ValidateDirMustExist);

DEFINE_bool(
    rotate_axes, false,
    "Set to true to change axes from x right and y down, to x down and y "
    "right. The assumed coordinate frame is centered at the top left side of "
    "the checkerboard, not the first interior corner like most checkerboard "
    "detectors assume. The reason for this is that it makes it easier to "
    "install markers directly along the axes of the target so the the vicon "
    "system measures the correct coordinate frame, and if we added the markers "
    "on the interior corners then the detection algorithms may not work.");

int main(int argc, char** argv) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_n_corners_horizontal == 0 || FLAGS_n_corners_vertical == 0 ||
      FLAGS_square_size == 0) {
    LOG_ERROR("Invalid input. Required parameters:  n_corners_vertical, "
              "n_corners_horizontal square_size. Exiting.");
    return 0;
  }

  std::string points = "{ \"keypoints_camera\": [";
  int n_max;
  int m_max;
  if (FLAGS_rotate_axes) {
    n_max = FLAGS_n_corners_vertical;
    m_max = FLAGS_n_corners_horizontal;
  } else {
    m_max = FLAGS_n_corners_vertical;
    n_max = FLAGS_n_corners_horizontal;
  }

  std::vector<double> point{0, 0, 0};
  for (int n = 1; n < n_max + 1; n++) {
    point[1] = point[1] + FLAGS_square_size;
    point[0] = 0;
    for (int m = 1; m < m_max + 1; m++) {
      point[0] = point[0] + FLAGS_square_size;
      points += "{ \"x\": " + std::to_string(point[0]) +
                ", \"y\": " + std::to_string(point[1]) +
                ", \"z\": " + std::to_string(point[2]) + "}";
      if (n != n_max || m != m_max) { points += ","; }
    }
  }
  points += "]}";
  nlohmann::json J = nlohmann::json::parse(points);
  std::string save_path =
      FLAGS_output_directory + "/checkerboard_keypoints.json";
  LOG_INFO("Saving to: %s", save_path.c_str());
  std::ofstream filejson(save_path);
  filejson << std::setw(2) << J << std::endl;
  return 0;
}
