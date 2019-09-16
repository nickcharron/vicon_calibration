#include "vicon_calibration/ViconCalibrator.h"

int main(int argc, char **argv)
{
  vicon_calibration::ViconCalibrator calibrator;
  calibrator.RunCalibration();
  return 0;
}
