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

Eigen::Matrix4d PerturbTransform(const Eigen::Matrix4d &T_in,
                                 const Eigen::VectorXd &perturbations) {
  Eigen::Vector3d r_perturb = perturbations.block(0,0,3,1);
  Eigen::Vector3d t_perturb = perturbations.block(3,0,3,1);
  Eigen::Matrix3d R_in = T_in.block(0,0,3,3);
  Eigen::Vector3d r_in = RToLieAlgebra(R_in);
  Eigen::Matrix3d R_out = LieAlgebraToR(r_in + r_perturb);
  Eigen::Matrix4d T_out;
  T_out.block(0, 3, 3, 1) = T_in.block(0,3,3,1) + t_perturb;
  T_out.block(0, 0, 3, 3) = R_out;
  return T_out;
}

Eigen::Vector3d InvSkewTransform(const Eigen::Matrix3d &M) {
  Eigen::Vector3d V;
  V(0) = M(2, 1);
  V(1) = M(0, 2);
  V(2) = M(1, 0);
  return V;
}

Eigen::Matrix3d SkewTransform(const Eigen::Vector3d &V) {
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
  return InvSkewTransform(R.log());
}

Eigen::Matrix3d LieAlgebraToR(const Eigen::Vector3d &eps) {
  return SkewTransform(eps).exp();
}

Eigen::Matrix4d InvertTransform(const Eigen::Matrix4d &T){
  Eigen::Matrix4d T_inv;
  T_inv.setIdentity();
  T_inv.block(0,0,3,3) = T.block(0,0,3,3).transpose();
  T_inv.block(0,3,3,1) = - T.block(0,0,3,3).transpose() * T.block(0,3,3,1);
  return T_inv;
}

cv::Mat
DrawCoordinateFrame(cv::Mat &img_in, Eigen::MatrixXd &T_cam_frame,
                    std::shared_ptr<beam_calibration::CameraModel> camera_model,
                    double &scale, bool images_distorted = true) {
  cv::Mat img_out;
  img_out = img_in.clone();
  Eigen::Vector4d origin(0, 0, 0, 1), x_end(scale, 0, 0, 1),
      y_end(0, scale, 0, 1), z_end(0, 0, scale, 1);
  Eigen::Vector4d o_trans = T_cam_frame * origin, x_trans = T_cam_frame * x_end,
                  y_trans = T_cam_frame * y_end, z_trans = T_cam_frame * z_end;

  Eigen::Vector3d o(o_trans(0), o_trans(1), o_trans(2)),
      x(x_trans(0), x_trans(1), x_trans(2)),
      y(y_trans(0), y_trans(1), y_trans(2)),
      z(z_trans(0), z_trans(1), z_trans(2));

  Eigen::Vector2d start_pixel;
  Eigen::Vector2d end_pixel_x;
  Eigen::Vector2d end_pixel_y;
  Eigen::Vector2d end_pixel_z;
  if(images_distorted){
    start_pixel = camera_model->ProjectPoint(o);
    end_pixel_x = camera_model->ProjectPoint(x);
    end_pixel_y = camera_model->ProjectPoint(y);
    end_pixel_z = camera_model->ProjectPoint(z);
  } else {
    start_pixel = camera_model->ProjectUndistortedPoint(o);
    end_pixel_x = camera_model->ProjectUndistortedPoint(x);
    end_pixel_y = camera_model->ProjectUndistortedPoint(y);
    end_pixel_z = camera_model->ProjectUndistortedPoint(z);
  }

  cv::Point start, end_x, end_y, end_z;
  start.x = start_pixel(0);
  start.y = start_pixel(1);
  end_x.x = end_pixel_x(0);
  end_x.y = end_pixel_x(1);
  end_y.x = end_pixel_y(0);
  end_y.y = end_pixel_y(1);
  end_z.x = end_pixel_z(0);
  end_z.y = end_pixel_z(1);

  cv::Scalar colourX(0, 0, 255); // BGR
  cv::Scalar colourY(0, 255, 0);
  cv::Scalar colourZ(255, 0, 0);
  int thickness = 3;

  cv::line(img_out, start, end_x, colourX, thickness);
  cv::line(img_out, start, end_y, colourY, thickness);
  cv::line(img_out, start, end_z, colourZ, thickness);

  // std::cout << "T_cam_frame': \n" << T_cam_frame.matrix() << "\n"
  //           << "Origin: \n" << o << "\n"
  //           << "Origin Projected: [" << start.x << " ," << start.y << "]\n";
  return img_out;
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
