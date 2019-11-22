#pragma once

#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"
#include <beam_calibration/CameraModel.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/slam/BetweenFactor.h>

namespace vicon_calibration {

class CameraFactor
    : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
  std::shared_ptr<beam_calibration::CameraModel> camera_model_;
  Eigen::Matrix4d T_VICONBASE_TARGET_;
  Eigen::Vector2d measured_pixel_;
  Eigen::Vector3d corresponding_point_;

public:
  CameraFactor(
      gtsam::Key i, Eigen::Vector2d measured_pixel,
      Eigen::Vector3d corresponding_point,
      std::shared_ptr<beam_calibration::CameraModel> &camera_model,
      Eigen::Matrix4d &T_VICONBASE_TARGET,const gtsam::SharedNoiseModel &model)
      :gtsam::NoiseModelFactor1<gtsam::Pose3>(model, i),
        measured_pixel_(measured_pixel),
        corresponding_point_(corresponding_point), camera_model_(camera_model),
        T_VICONBASE_TARGET_(T_VICONBASE_TARGET) {}

  /** destructor */
  ~CameraFactor() {}

  gtsam::Vector
  evaluateError(const gtsam::Pose3 &q,
                boost::optional<gtsam::Matrix &> H = boost::none) const {
    Eigen::Matrix3d R_VT, R_op;
    Eigen::Vector3d t_VT, t_op;
    R_VT = T_VICONBASE_TARGET_.block(0, 0, 3, 3);
    t_VT = T_VICONBASE_TARGET_.block(0, 3, 3, 1);
    Eigen::Matrix4d T_op = q.matrix();
    R_op = T_op.block(0, 0, 3, 3);
    t_op = T_op.block(0, 3, 3, 1);
    Eigen::Vector4d point_homo = utils::PointToHomoPoint(corresponding_point_);
    Eigen::Vector4d point_transformed =
        T_op * T_VICONBASE_TARGET_ * point_homo;
    Eigen::Vector2d projected_point =
        camera_model_->ProjectPoint(point_transformed);
    gtsam::Vector error = measured_pixel_ - projected_point;

    if (H) {
      Eigen::Matrix3d K = camera_model_->GetCameraMatrix();
      Eigen::MatrixXd H_(3, 6);
      H_.block(0, 0, 3, 3) =
          -K * R_op * utils::SkewTransform(-R_VT * corresponding_point_ - t_VT);
      H_.block(0, 3, 3, 3) = -K;
      // TODO: Not sure if this is correct. We usually normalize but in this
      // case it isn't a 3 x 1 that we can normalize
      (*H) = H_.block(0, 0, 2, 6);
    }
    return error;
  }

}; // CameraFactor

} // end namespace vicon_calibration
