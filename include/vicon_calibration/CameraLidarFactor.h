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
  Eigen::Matrix4d T_VICONBASE_TARGET_;
  Eigen::Vector2d pixel_detected_;
  Eigen::Vector3d point_detected_, P_T_ci_, P_T_li_;

public:
  CameraLidarFactor(
      gtsam::Key i, gtsam::Key j, Eigen::Vector2d pixel_detected,
      Eigen::Vector3d point_detected, Eigen::Vector3d P_T_ci,
      Eigen::Vector3d P_T_li,
      std::shared_ptr<beam_calibration::CameraModel> &camera_model,
      const gtsam::SharedNoiseModel &model)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model, i, j),
        pixel_detected_(pixel_detected), point_detected_(point_detected),
        P_T_ci_(P_T_ci), P_T_li_(P_T_li), camera_model_(camera_model) {}

  /** destructor */
  ~CameraLidarFactor() {}

  gtsam::Vector
  evaluateError(const gtsam::Pose3 &T_B_L, const gtsam::Pose3 &T_B_C,
                boost::optional<gtsam::Matrix &> HL = boost::none,
                boost::optional<gtsam::Matrix &> HC = boost::none) const {
    Eigen::Matrix4d T_B_C_eig = T_B_C.matrix();
    Eigen::Matrix4d T_B_L_eig = T_B_L.matrix();
    Eigen::Vector4d tmp_point =
        utils::InvertTransform(T_B_C_eig) * T_B_L_eig *
        utils::PointToHomoPoint(point_detected_ + P_T_ci_ + P_T_li_);
    Eigen::Vector2d projected_point = camera_model_->ProjectPoint(tmp_point);
    gtsam::Vector error = pixel_detected_ - projected_point;

    if (HL) {
      Eigen::Matrix3d K = camera_model_->GetCameraMatrix();
      Eigen::Matrix3d R_B_L = T_B_L_eig.block(0, 0, 3, 3);
      Eigen::Matrix3d R_C_B = utils::InvertTransform(T_B_C_eig).block(0, 0, 3, 3);
      Eigen::Vector3d t_C_B = utils::InvertTransform(T_B_C_eig).block(0, 3, 3, 1);
      Eigen::Matrix3d tmp =
          R_B_L * utils::SkewTransform(-(point_detected_ + P_T_ci_ + P_T_li_));
      Eigen::MatrixXd HL_(3, 6);
      HL_.block(0, 0, 3, 3) = - K * (R_C_B * tmp);
      HL_.block(0, 3, 3, 3) =
          -K * (utils::InvertTransform(T_B_C_eig).block(0, 0, 3, 3) +
                utils::InvertTransform(T_B_C_eig).block(0, 3, 3, 1));
      // TODO: Not sure if this is correct. We usually normalize but in this
      // case it isn't a 3 x 1 that we can normalize
      (*HL) = HL_.block(0, 0, 2, 6);
    }
    if (HC) {
      Eigen::Matrix3d K = camera_model_->GetCameraMatrix();
      Eigen::MatrixXd HC_(3, 6);
      Eigen::Matrix3d R_B_L = T_B_L_eig.block(0, 0, 3, 3);
      Eigen::Matrix3d R_B_C = T_B_C_eig.block(0, 0, 3, 3);
      Eigen::Vector3d t_B_L = T_B_L_eig.block(0, 3, 3, 1);
      Eigen::Vector3d t_B_C = T_B_C_eig.block(0, 3, 3, 1);
      Eigen::Vector3d tmp1 =
          R_B_C.transpose() * R_B_L * (point_detected_ + P_T_ci_ + P_T_li_) +
          t_B_L;
      Eigen::Vector3d tmp2 = R_B_C.transpose() * t_B_C;
      HC_.block(0, 0, 3, 3) =
          -K * (utils::SkewTransform(tmp1) - utils::SkewTransform(tmp2));
      HC_.block(0, 3, 3, 3) = -K * R_B_C.transpose();
      // TODO: Not sure if this is correct. We usually normalize but in this
      // case it isn't a 3 x 1 that we can normalize
      (*HC) = HC_.block(0, 0, 2, 6);
    }

    return error;
  }

}; // CameraLidarFactor

} // end namespace vicon_calibration
