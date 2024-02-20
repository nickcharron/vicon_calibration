#include <ceres/autodiff_cost_function.h>
#include <ceres/rotation.h>

struct CeresLidarCostFunction {
  CeresLidarCostFunction(Eigen::Vector3d point_detected,
                         Eigen::Vector3d P_Robot)
      : point_detected_(point_detected), P_Robot_(P_Robot) {}

  // T_LR is [qw, qx, qy, qz, tx, ty, tz]
  template <typename T>
  bool operator()(const T* const T_LR, T* residuals) const {
    T P_Robot[3];
    P_Robot[0] = P_Robot_.cast<T>()[0];
    P_Robot[1] = P_Robot_.cast<T>()[1];
    P_Robot[2] = P_Robot_.cast<T>()[2];

    // rotate and translate point
    T P_Lidar[3];
    ceres::QuaternionRotatePoint(T_LR, P_Robot, P_Lidar);
    P_Lidar[0] += T_LR[4];
    P_Lidar[1] += T_LR[5];
    P_Lidar[2] += T_LR[6];

    residuals[0] = point_detected_.cast<T>()[0] - P_Lidar[0];
    residuals[1] = point_detected_.cast<T>()[1] - P_Lidar[1];
    residuals[2] = point_detected_.cast<T>()[2] - P_Lidar[2];

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Vector3d point_detected,
                                     const Eigen::Vector3d P_Robot) {
    return (new ceres::AutoDiffCostFunction<CeresLidarCostFunction, 3, 7>(
        new CeresLidarCostFunction(point_detected, P_Robot)));
  }

  Eigen::Vector3d point_detected_;
  Eigen::Vector3d P_Robot_;
};