#include <ceres/autodiff_cost_function.h>
#include <ceres/rotation.h>

struct CeresLidarCostFunction {
  CeresLidarCostFunction(const Eigen::Vector3d& P_LidarDetected,
                         const Eigen::Vector3d& P_Target,
                         const Eigen::Matrix4d& T_Robot_Target)
      : P_LidarDetected_(P_LidarDetected),
        P_Target_(P_Target),
        T_Robot_Target_(T_Robot_Target) {}

  // T_LR is [qw, qx, qy, qz, tx, ty, tz]
  template <typename T>
  bool operator()(const T* const T_Lidar_Robot,
                  const T* const T_TargetCorrected_Target, T* residuals) const {
    // transform point to corrected target frame
    T P_Target[3] = {T(P_Target_[0]), T(P_Target_[1]), T(P_Target_[2])};
    T P_TargetCorrected[3];
    ceres::QuaternionRotatePoint(T_TargetCorrected_Target, P_Target,
                                 P_TargetCorrected);
    P_TargetCorrected[0] += T_TargetCorrected_Target[4];
    P_TargetCorrected[1] += T_TargetCorrected_Target[5];
    P_TargetCorrected[2] += T_TargetCorrected_Target[6];

    // transform point to robot frame
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> P_TargetCorrected_Eig(
        P_TargetCorrected);
    Eigen::Matrix<T, 4, 1> P_Robot_Eig =
        T_Robot_Target_.cast<T>() * P_TargetCorrected_Eig.homogeneous();
    T P_Robot[3] = {P_Robot_Eig[0], P_Robot_Eig[1], P_Robot_Eig[2]};

    // transform point to lidar frame
    T P_Lidar[3];
    ceres::QuaternionRotatePoint(T_Lidar_Robot, P_Robot, P_Lidar);
    P_Lidar[0] += T_Lidar_Robot[4];
    P_Lidar[1] += T_Lidar_Robot[5];
    P_Lidar[2] += T_Lidar_Robot[6];

    residuals[0] = T(P_LidarDetected_[0]) - P_Lidar[0];
    residuals[1] = T(P_LidarDetected_[1]) - P_Lidar[1];
    residuals[2] = T(P_LidarDetected_[2]) - P_Lidar[2];

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Vector3d& P_LidarDetected,
                                     const Eigen::Vector3d& P_Target,
                                     const Eigen::Matrix4d& T_Robot_Target) {
    return (new ceres::AutoDiffCostFunction<CeresLidarCostFunction, 3, 7, 7>(
        new CeresLidarCostFunction(P_LidarDetected, P_Target, T_Robot_Target)));
  }

  Eigen::Vector3d P_LidarDetected_;
  Eigen::Vector3d P_Target_;
  Eigen::Matrix4d T_Robot_Target_;
};