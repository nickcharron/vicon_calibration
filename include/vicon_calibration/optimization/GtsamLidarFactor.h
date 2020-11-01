#pragma once

#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"
#include <beam_calibration/CameraModel.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/slam/BetweenFactor.h>

namespace vicon_calibration {

class LidarFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
  Eigen::Matrix4d T_VICONBASE_TARGET_;
  Eigen::Vector3d measured_point_;
  Eigen::Vector3d corresponding_point_;

public:
  LidarFactor(const gtsam::Key i, const Eigen::Vector3d measured_point,
              const Eigen::Vector3d corresponding_point,
              const Eigen::Matrix4d& T_VICONBASE_TARGET,
              const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, i),
        measured_point_(measured_point),
        corresponding_point_(corresponding_point),
        T_VICONBASE_TARGET_(T_VICONBASE_TARGET) {}

  /** destructor */
  ~LidarFactor() {}

  gtsam::Vector
      evaluateError(const gtsam::Pose3& q,
                    boost::optional<gtsam::Matrix&> H = boost::none) const {
    Eigen::Matrix3d R_VICONBASE_TARGET = T_VICONBASE_TARGET_.block(0, 0, 3, 3);
    Eigen::Vector3d t_VICONBASE_TARGET = T_VICONBASE_TARGET_.block(0, 3, 3, 1);
    Eigen::Matrix4d T_LIDAR_VICONBASE = q.matrix();
    Eigen::Vector3d P_VICONBASE =
        (T_VICONBASE_TARGET_ * corresponding_point_.homogeneous())
            .hnormalized();
    Eigen::Vector3d P_LIDAR =
        (T_LIDAR_VICONBASE * P_VICONBASE.homogeneous()).hnormalized();
    gtsam::Vector error = measured_point_ - P_LIDAR;

    if (H) {
      Eigen::MatrixXd H_(3, 6);
      H_.block(0, 0, 3, 3) =
          -T_LIDAR_VICONBASE.block(0, 0, 3, 3) * utils::SkewTransform(-P_LIDAR);
      H_.block(0, 3, 3, 3) = -Eigen::Matrix3d::Identity();
      (*H) = H_;
    }
    return error;
  }

}; // LidarFactor

} // end namespace vicon_calibration
