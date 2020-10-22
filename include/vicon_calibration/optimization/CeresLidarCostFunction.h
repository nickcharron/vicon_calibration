#include <ceres/autodiff_cost_function.h>
#include <ceres/rotation.h>

struct CeresLidarCostFunction {
  CeresLidarCostFunction(Eigen::Vector3d point_detected,
                         Eigen::Vector3d P_VICONBASE)
      : point_detected_(point_detected), P_VICONBASE_(P_VICONBASE) {}

  // T_LR is [qw, qx, qy, qz, tx, ty, tz]
  template <typename T>
  bool operator()(const T* const T_LR, T* residuals) const {
    T P_VICONBASE[3];
    P_VICONBASE[0] = P_VICONBASE_.cast<T>()[0];
    P_VICONBASE[1] = P_VICONBASE_.cast<T>()[1];
    P_VICONBASE[2] = P_VICONBASE_.cast<T>()[2];

    // rotate and translate point
    T P_LIDAR[3];
    ceres::QuaternionRotatePoint(T_LR, P_VICONBASE, P_LIDAR);
    P_LIDAR[0] += T_LR[4];
    P_LIDAR[1] += T_LR[5];
    P_LIDAR[2] += T_LR[6];

    residuals[0] = point_detected_.cast<T>()[0] - P_LIDAR[0];
    residuals[1] = point_detected_.cast<T>()[1] - P_LIDAR[1];
    residuals[2] = point_detected_.cast<T>()[2] - P_LIDAR[2];
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Vector3d point_detected,
                                     const Eigen::Vector3d P_VICONBASE) {
    return (new ceres::AutoDiffCostFunction<CeresLidarCostFunction, 3, 7>(
        new CeresLidarCostFunction(point_detected, P_VICONBASE)));
  }

  Eigen::Vector3d point_detected_;
  Eigen::Vector3d P_VICONBASE_;
};