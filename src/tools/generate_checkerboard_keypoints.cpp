/* This exectuable can be used to generate keypoints used for a checkerboard
  target that you will include in your TargetConfig.json config file (see
  DiamondTaret.json).

  Usage: ./path_to_executable/vicon_calibration_generate_checkerboard_keypoints
  n m d where n is the number of interior corners in the vertical direction and
  m is the number of interior corners in the horizontal direction and d is the
  distance between corners (i.e. the size of the squares) in meters.

  **Note:** This assumes the coordinate frame is actually at the top left
  "exterior" edge, not the first interior edge which is usually the assumed
  coordinate frame for checkerboard based calibration. The reason for this is
  that for vicon calibration we need to add reflective markers to define the
  coordinate frame and we don't want these to interfere with the checkerboard
  detection. Also, x is assume to be downwards, y rightwards and z out.

  Output: This will output the file to your current directory
  (named checkerboard_keypoints.json)
*/

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "----------------------------------------------------------\n"
              << "USAGE: " << argv[0] << " n  m  d\n"
              << "Where  n is the number of interior corners in the vertical "
                 "direction and m is the number of interior corners in the "
                 "horizontal direction and d is the distance between corners "
                 "(i.e. the size of the squares) in meters.\n"
              << "----------------------------------------------------------\n";
    return 1;
  }

  int height = std::atof(argv[1]);
  int width = std::atof(argv[2]);
  double d = std::atof(argv[3]);

  std::string points = "{ \"keypoints_camera\": [";

  std::vector<double> point{0,0,0};
  for(int n = 0; n < width; n++){
    point[0] = point[0] + d;
    for(int m = 0; m < height; m++){
      point[1] = point[1] + d;
      points+= "{ \"x\": " + std::to_string(point[0])
                + ", \"y\": " + std::to_string(point[1])
                + ", \"z\": " + std::to_string(point[2])
                + "}";
      if(n+1 < width || m+1 < height){
        points+= ",";
      }
    }
  }
  points+= "]}";
  nlohmann::json J = nlohmann::json::parse(points);
  std::string output_file = "checkerboard_keypoints.json";
  std::ofstream filejson(output_file);
  filejson << std::setw(2) << J << std::endl;
  return 0;
}
