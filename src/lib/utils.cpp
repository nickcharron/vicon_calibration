#include "vicon_calibration/utils.h"
#include <unsupported/Eigen/MatrixFunctions>

namespace vicon_calibration {

namespace utils {

Eigen::MatrixXd RoundMatrix(const Eigen::MatrixXd &M, int precision) {
  Eigen::MatrixXd Mround(M.rows(), M.cols());
  for (int i = 0; i < M.rows(); i++) {
    for (int j = 0; j < M.cols(); j++) {
      Mround(i, j) = std::round(M(i, j) * std::pow(10, precision)) /
                     std::pow(10, precision);
    }
  }
  return Mround;
}

bool IsRotationMatrix(const Eigen::Matrix3d R) {
  int precision = 3;
  Eigen::Matrix3d shouldBeIdentity = RoundMatrix(R * R.transpose(), precision);
  double detR = R.determinant();
  double detRRound = std::round(detR * precision) / precision;
  if (shouldBeIdentity.isIdentity() && detRRound == 1) {
    return 1;
  } else {
    return 0;
  }
}

bool IsTransformationMatrix(const Eigen::Matrix4d T) {
  Eigen::Matrix3d R = T.block(0, 0, 3, 3);
  bool homoFormValid, tValid;

  // check translation for infinity or nan
  if (std::isinf(T(0, 3)) || std::isinf(T(1, 3)) || std::isinf(T(2, 3)) ||
      std::isnan(T(0, 3)) || std::isnan(T(1, 3)) || std::isnan(T(2, 3))) {
    tValid = 0;
  } else {
    tValid = 1;
  }

  // check that bottom row is [0 0 0 1]
  if (T(3, 0) == 0 && T(3, 1) == 0 && T(3, 2) == 0 && T(3, 3) == 1) {
    homoFormValid = 1;
  } else {
    homoFormValid = 0;
  }

  if (homoFormValid && tValid && IsRotationMatrix(R)) {
    return 1;
  } else {
    return 0;
  }
}

Eigen::Affine3d PerturbTransform(const Eigen::Affine3d &T_in,
                                 const std::vector<double> &perturbations) {
  Eigen::Vector3d r_perturb(perturbations[3], perturbations[4],
                            perturbations[5]);
  Eigen::Vector3d t_perturb(perturbations[0], perturbations[1],
                            perturbations[2]);
  Eigen::Matrix3d R_in = T_in.rotation();
  Eigen::Vector3d r_in = RToLieAlgebra(R_in);
  Eigen::Matrix3d R_out = LieAlgebraToR(r_in + r_perturb);
  Eigen::Affine3d T_out;
  T_out.matrix().block(0, 3, 3, 1) = T_in.translation() + t_perturb;
  T_out.matrix().block(0, 0, 3, 3) = R_out;
  return T_out;
}

Eigen::Vector3d invSkewTransform(const Eigen::Matrix3d &M) {
  Eigen::Vector3d V;
  V(0) = M(2, 1);
  V(1) = M(0, 2);
  V(2) = M(1, 0);
  return V;
}

Eigen::Matrix3d skewTransform(const Eigen::Vector3d &V) {
  Eigen::Matrix3d M;
  M(0, 0) = 0;
  M(0, 1) = -V(2, 0);
  M(0, 2) = V(1, 0);
  M(1, 0) = V(2, 0);
  M(1, 1) = 0;
  M(1, 2) = -V(0, 0);
  M(2, 0) = -V(1, 0);
  M(2, 1) = V(0, 0);
  M(2, 2) = 0;
  return M;
}

Eigen::Vector3d RToLieAlgebra(const Eigen::Matrix3d &R) {
  return invSkewTransform(R.log());
}

Eigen::Matrix3d LieAlgebraToR(const Eigen::Vector3d &eps) {
  return skewTransform(eps).exp();
}

Eigen::Matrix4d RemoveYaw(const Eigen::Matrix4d &T_in) {
  Eigen::Matrix4d T_out = T_in;
  Eigen::Matrix3d R_in = T_in.block(0, 0, 3, 3);
  Eigen::Vector3d rpy = R_in.eulerAngles(0, 1, 2);
  Eigen::Matrix3d R_out;
  // R_out = Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ()) *
  //         Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
  //         Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX());
  R_out = Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());
  T_out.block(0, 0, 3, 3) = R_out;
  return T_out;
}

Eigen::Matrix4d RemoveYaw2(const Eigen::Matrix4d &T_in) {
  Eigen::Matrix4d T_out = T_in;
  Eigen::Matrix3d R_in = T_in.block(0, 0, 3, 3);
  Eigen::Vector3d rpy = R_in.eulerAngles(2, 1, 0);
  Eigen::Matrix3d R_out;
  R_out = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX());
  T_out.block(0, 0, 3, 3) = R_out;
  return T_out;
}

void OutputTransformInformation(Eigen::Affine3d &T,
                                std::string transform_name) {
  double RAD_TO_DEG = 57.29577951;
  Eigen::Matrix3d R = T.rotation();
  Eigen::Vector3d rpy = R.eulerAngles(0, 1, 2);
  std::cout << transform_name << ":\n"
            << T.matrix() << "\n"
            << "rpy (deg): [" << rpy[0] * RAD_TO_DEG << ", "
            << rpy[1] * RAD_TO_DEG << ", " << rpy[2] * RAD_TO_DEG << "]\n";
}

void OutputLidarMeasurements(
    std::vector<vicon_calibration::LidarMeasurement> &measurements) {
  std::cout << "-------------------------------\n"
            << "Saved a total of " << measurements.size()
            << " lidar measurements \n"
            << "Outputting lidar measurements: \n";
  for (uint32_t i = 0; i < measurements.size(); i++) {
    std::cout << "lidar_id: " << measurements[i].lidar_id << "\n"
              << "target_id: " << measurements[i].target_id << "\n"
              << "lidar_frame: " << measurements[i].lidar_frame << "\n"
              << "target_frame: " << measurements[i].target_frame << "\n"
              << "vicon_base_frame: " << measurements[i].vicon_base_frame
              << "\n"
              << "T_LIDAR_TARGET: \n"
              << measurements[i].T_LIDAR_TARGET << "\n"
              << "T_VICONBASE_TARGET: \n"
              << measurements[i].T_VICONBASE_TARGET << "\n";
  }
}

void OutputCalibrations(
    std::vector<vicon_calibration::CalibrationResult> &calib,
    std::string output_string) {
  std::cout << "----------------------\n" << output_string << "\n";
  for (uint16_t i = 0; i < calib.size(); i++) {
    std::cout << "T_" << calib[i].to_frame << "_" << calib[i].from_frame
              << ":\n"
              << calib[i].transform << "\n";
  }
}
} // namespace utils

} // end namespace vicon_calibration