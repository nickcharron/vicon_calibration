/* This exectuable can be used to generate a pcd template cloud for a
  checkerboard.

  **Note:** This assumes the coordinate frame is actually at the top left
  "exterior" edge, not the first interior edge which is usually the assumed
  coordinate frame for checkerboard based calibration. The reason for this is
  that for vicon calibration we need to add reflective markers to define the
  coordinate frame and we don't want these to interfere with the checkerboard
  detection.

*/

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <gflags/gflags.h>

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
DEFINE_double(horizontal_border, 0,
              "Size of border in the horizontal direction, in meters.");
DEFINE_double(vertical_border, 0,
              "Size of border in the vertical direction, in meters.");
DEFINE_double(density, 0.001,
              "Density of the template cloud, this is the distance between "
              "points in meters. Default: 0.001");

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
    LOG_ERROR("Invalid input, exiting.");
    return 0;
  }

  double dim_x = (FLAGS_n_corners_horizontal + 1) * FLAGS_square_size;
  double dim_y = (FLAGS_n_corners_vertical + 1) * FLAGS_square_size;
  double x_start = -FLAGS_horizontal_border;
  double y_start = -FLAGS_vertical_border;
  double x_end = dim_x + FLAGS_horizontal_border;
  double y_end = dim_y + FLAGS_vertical_border;

  if (FLAGS_rotate_axes) {
    std::swap(dim_x, dim_y);
    std::swap(x_start, y_start);
    std::swap(x_end, y_end);
  }

  PointCloud cloud;
  for (double x = x_start; x <= x_end; x += FLAGS_density) {
    for (double y = y_start; y <= y_end; y += FLAGS_density) {
      cloud.push_back(pcl::PointXYZ(x, y, 0));
    }
  }

  std::string save_path = FLAGS_output_directory + "/checkerboard_template.pcd";
  LOG_INFO("Saving template cloud to: %s", save_path.c_str());
  std::string error_type;
  if (!utils::SavePointCloud<pcl::PointXYZ>(
          save_path, cloud, utils::PointCloudFileType::PCDBINARY, error_type)) {
    LOG_ERROR("Cannot save pointcloud, reason: %s", error_type.c_str());
  }

  return 0;
}
