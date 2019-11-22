#pragma once

#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"
#include <beam_calibration/CameraModel.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/slam/BetweenFactor.h>

namespace vicon_calibration {

class LidarFactor
    : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
  Eigen::Matrix4d T_VICONBASE_TARGET_;
  Eigen::Vector3d measured_point_;
  Eigen::Vector3d corresponding_point_;

public:
  LidarFactor(
      const gtsam::Key i,
      const Eigen::Vector3d measured_point,
      const Eigen::Vector3d corresponding_point,
      const Eigen::Matrix4d &T_VICONBASE_TARGET,
      const gtsam::SharedNoiseModel &model)
      :gtsam::NoiseModelFactor1<gtsam::Pose3>(model, i),
        measured_point_(measured_point),
        corresponding_point_(corresponding_point),
        T_VICONBASE_TARGET_(T_VICONBASE_TARGET) {}

  /** destructor */
  ~LidarFactor() {}

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
    gtsam::Vector error = measured_point_ - utils::HomoPointToPoint(point_transformed);

    if (H) {
      Eigen::MatrixXd H_(3, 6);
      H_.block(0, 0, 3, 3) =
          - R_op * utils::SkewTransform(-R_VT * corresponding_point_ - t_VT);
      Eigen::Matrix3d I3;
      I3.setIdentity();
      H_.block(0, 3, 3, 3) = - I3;
      (*H) = H_;
    }
    return error;
  }

}; // LidarFactor

} // end namespace vicon_calibration
