#pragma once

#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"
#include <beam_calibration/CameraModel.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/slam/BetweenFactor.h>

namespace vicon_calibration {

class CameraLidarFactor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
  std::shared_ptr<beam_calibration::CameraModel> camera_model_;
  Eigen::Vector2i pixel_detected_;
  Eigen::Vector3d point_detected_, P_T_ci_, P_T_li_;
  double Fx_, Fy_, Cx_, Cy_;

public:
  CameraLidarFactor(
      gtsam::Key lid_key, gtsam::Key cam_key, Eigen::Vector2i pixel_detected,
      Eigen::Vector3d point_detected, Eigen::Vector3d P_T_ci,
      Eigen::Vector3d P_T_li,
      std::shared_ptr<beam_calibration::CameraModel> &camera_model,
      const gtsam::SharedNoiseModel &model)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model, lid_key,
                                                             cam_key),
        pixel_detected_(pixel_detected), point_detected_(point_detected),
        P_T_ci_(P_T_ci), P_T_li_(P_T_li), camera_model_(camera_model) {}

  /** destructor */
  ~CameraLidarFactor() {}

  gtsam::Vector
  evaluateError(const gtsam::Pose3 &_T_VL, const gtsam::Pose3 &_T_VC,
                boost::optional<gtsam::Matrix &> HL = boost::none,
                boost::optional<gtsam::Matrix &> HC = boost::none) const {
    Eigen::Matrix4d T_VC = _T_VC.matrix();
    Eigen::Matrix4d T_VL = _T_VL.matrix();
    Eigen::Matrix3d R_VC = T_VC.block(0, 0, 3, 3);
    Eigen::Vector3d t_VC = T_VC.block(0, 3, 3, 1);
    Eigen::Matrix3d R_VL = T_VL.block(0, 0, 3, 3);
    Eigen::Vector3d t_VL = T_VL.block(0, 3, 3, 1);
    Eigen::Vector3d tmp_point = point_detected_ + P_T_ci_ - P_T_li_;
    Eigen::Vector3d point_transformed =
        R_VC.transpose() * (R_VL * tmp_point + t_VL - t_VC);
    Eigen::MatrixXd dfdg(2, 3);
    opt<Eigen::Vector2i> projected_point =
        camera_model_->ProjectPoint(point_transformed, dfdg);
    if (!projected_point.has_value()) {
      throw std::runtime_error{"Cannot project point."};
    }

    gtsam::Vector error =
        (pixel_detected_ - projected_point.value()).cast<double>();

    // Assume e(R,t) = measured_pixel - f(g(R,t))
    // -> de/d(R,t) = [de/df * df/dg * dg/dR , de/df * df/dg * dg/dt]
    if (HL) {
      Eigen::MatrixXd H_(2, 6);
      Eigen::MatrixXd dgdR(3, 3);
      Eigen::MatrixXd dgdt(3, 3);
      Eigen::MatrixXd dedf(2, 2);
      dgdR = R_VC.transpose() * R_VL * utils::SkewTransform(-1 * tmp_point);
      dgdt = R_VC.transpose();
      dedf.setIdentity();
      dedf = -1 * dedf;
      H_.block(0, 0, 2, 3) = dedf * dfdg * dgdR;
      H_.block(0, 3, 2, 3) = dedf * dfdg * dgdt;
      (*HL) = H_;
    }
    if (HC) {
      Eigen::MatrixXd H_(2, 6);
      Eigen::MatrixXd dgdR(3, 3);
      Eigen::MatrixXd dgdt(3, 3);
      Eigen::MatrixXd dedf(2, 2);
      dgdR = utils::SkewTransform(R_VC.transpose() *
                                  (R_VL * tmp_point + t_VL - t_VC));
      dgdt = -1 * R_VC.transpose();
      dedf.setIdentity();
      dedf = -1 * dedf;
      H_.block(0, 0, 2, 3) = dedf * dfdg * dgdR;
      H_.block(0, 3, 2, 3) = dedf * dfdg * dgdt;
      (*HC) = H_;
    }
    return error;
  }

}; // CameraLidarFactor

} // end namespace vicon_calibration
