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
  Eigen::Vector2d pixel_detected_;
  Eigen::Vector3d point_detected_, P_T_ci_, P_T_li_;
  double Fx_, Fy_, Cx_, Cy_;

public:
  CameraLidarFactor(
      gtsam::Key lid_key, gtsam::Key cam_key, Eigen::Vector2d pixel_detected,
      Eigen::Vector3d point_detected, Eigen::Vector3d P_T_ci,
      Eigen::Vector3d P_T_li,
      std::shared_ptr<beam_calibration::CameraModel>& camera_model,
      const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model, lid_key,
                                                             cam_key),
        pixel_detected_(pixel_detected),
        point_detected_(point_detected),
        P_T_ci_(P_T_ci),
        P_T_li_(P_T_li),
        camera_model_(camera_model) {}

  /** destructor */
  ~CameraLidarFactor() {}

  gtsam::Vector
      evaluateError(const gtsam::Pose3& _T_LV, const gtsam::Pose3& _T_CV,
                    boost::optional<gtsam::Matrix&> HL = boost::none,
                    boost::optional<gtsam::Matrix&> HC = boost::none) const {
    Eigen::Matrix4d T_CV = _T_CV.matrix();
    Eigen::Matrix4d T_LV = _T_LV.matrix();
    Eigen::Matrix3d R_CV = T_CV.block(0, 0, 3, 3);
    Eigen::Vector3d t_CV = T_CV.block(0, 3, 3, 1);
    Eigen::Matrix3d R_LV = T_LV.block(0, 0, 3, 3);
    Eigen::Vector3d t_LV = T_LV.block(0, 3, 3, 1);

    Eigen::Vector3d P_L_C = point_detected_ + P_T_ci_ - P_T_li_;
    Eigen::Vector3d P_CAM =
        R_CV * (R_LV.transpose() * P_L_C - R_LV * t_LV) + t_CV;

    Eigen::MatrixXd dfdg(2, 3);
    opt<Eigen::Vector2d> projected_point =
        camera_model_->ProjectPointPrecise(P_CAM);
    camera_model_->ProjectPoint(P_CAM, dfdg);

    gtsam::Vector error{Eigen::Vector2d::Zero()};
    if (projected_point.has_value()) {
      error = (pixel_detected_ - projected_point.value()).cast<double>();
    }

    // TODO: do we leave the error as zero if it doesn't project to the image
    // plane? Do we need to change the jacobian?

    if (HL) {
      Eigen::MatrixXd H_(2, 6);
      Eigen::MatrixXd dgdR(3, 3);
      Eigen::MatrixXd dgdt(3, 3);

      dgdR = -R_CV * R_LV * utils::SkewTransform(-t_LV) +
             R_CV * utils::SkewTransform(R_LV.transpose() * P_L_C);
      dgdt = -R_CV * R_LV;

      H_.block(0, 0, 2, 3) = -dfdg * dgdR;
      H_.block(0, 3, 2, 3) = -dfdg * dgdt;
      (*HL) = H_;
    }
    if (HC) {
      Eigen::MatrixXd H_(2, 6);
      Eigen::MatrixXd dgdR(3, 3);
      Eigen::MatrixXd dgdt(3, 3);

      dgdR =
          R_CV * utils::SkewTransform(-R_LV.transpose() * P_L_C + R_LV * t_LV);
      dgdt = Eigen::Matrix3d::Identity();

      H_.block(0, 0, 2, 3) = -dfdg * dgdR;
      H_.block(0, 3, 2, 3) = -dfdg * dgdt;
      (*HC) = H_;
    }
    return error;
  }

}; // CameraLidarFactor

} // end namespace vicon_calibration
