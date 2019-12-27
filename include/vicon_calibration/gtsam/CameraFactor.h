#pragma once

#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"
#include <beam_calibration/CameraModel.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/slam/BetweenFactor.h>

namespace vicon_calibration {

class CameraFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
  std::shared_ptr<beam_calibration::CameraModel> camera_model_;
  Eigen::Matrix4d T_VICONBASE_TARGET_;
  Eigen::Vector2d measured_pixel_;
  Eigen::Vector3d corresponding_point_;
  double Fx_, Fy_, Cx_, Cy_;

public:
  CameraFactor(const gtsam::Key i, const Eigen::Vector2d measured_pixel,
               const Eigen::Vector3d corresponding_point,
               const std::shared_ptr<beam_calibration::CameraModel> &camera_model,
               const Eigen::Matrix4d &T_VICONBASE_TARGET,
               const gtsam::SharedNoiseModel &model)
      : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, i),
        measured_pixel_(measured_pixel),
        corresponding_point_(corresponding_point), camera_model_(camera_model),
        T_VICONBASE_TARGET_(T_VICONBASE_TARGET), Fx_(camera_model->GetFx()),
        Fy_(camera_model->GetFy()), Cx_(camera_model->GetCx()),
        Cy_(camera_model->GetCy()) {}

  /** destructor */
  ~CameraFactor() {}

  gtsam::Vector
  evaluateError(const gtsam::Pose3 &q,
                boost::optional<gtsam::Matrix &> H = boost::none) const {
    Eigen::Matrix4d T_op;
    Eigen::Matrix3d R_VT, R_op;
    Eigen::Vector3d t_VT, t_op;
    R_VT = T_VICONBASE_TARGET_.block(0, 0, 3, 3);
    t_VT = T_VICONBASE_TARGET_.block(0, 3, 3, 1);
    T_op = q.matrix();
    R_op = T_op.block(0, 0, 3, 3);
    t_op = T_op.block(0, 3, 3, 1);
    Eigen::Vector3d point_transformed =
        R_op.transpose() * (R_VT * corresponding_point_ + t_VT - t_op);
    Eigen::Vector2d projected_point;
    projected_point[0] =
        Cx_ + point_transformed[0] * Fx_ / point_transformed[2];
    projected_point[1] =
        Cy_ + point_transformed[1] * Fy_ / point_transformed[2];
    // Eigen::Vector2d projected_point =
    //     camera_model_->ProjectUndistortedPoint(point_transformed);
    gtsam::Vector error = measured_pixel_ - projected_point;

    if (H) {
      // Assume e(R,t) = measured_pixel - f(g(R,t))
      // -> de/d(R,t) = - [df/dg * dg/dR , df/dg * dg/dt]
      Eigen::MatrixXd H_(2, 6), dfdg(2, 3), dgdR(3, 3), dgdt(3, 3);
      dfdg(0, 0) = Fx_ / point_transformed[2];
      dfdg(1, 0) = 0;
      dfdg(0, 1) = 0;
      dfdg(1, 1) = Fy_ / point_transformed[2];
      dfdg(0, 2) = -point_transformed[0] * Fx_ /
                   ((point_transformed[2]) * (point_transformed[2]));
      dfdg(1, 2) = -point_transformed[1] * Fy_ /
                   ((point_transformed[2]) * (point_transformed[2]));
      dgdR = utils::SkewTransform(R_op.transpose() *
                                  (R_VT * corresponding_point_ + t_VT - t_op));
      dgdt = -R_op.transpose();
      H_.block(0, 0, 2, 3) = -dfdg * dgdR;
      H_.block(0, 3, 2, 3) = -dfdg * dgdt;
      (*H) = H_;
    }
    return error;
  }

}; // CameraFactor

} // end namespace vicon_calibration