#include <ceres/autodiff_cost_function.h>
#include <ceres/rotation.h>

#include <vicon_calibration/camera_models/CameraModel.h>

struct CameraProjectionFunctor {
  CameraProjectionFunctor(
      const std::shared_ptr<vicon_calibration::CameraModel>& camera_model)
      : camera_model_(camera_model) {}

  bool operator()(const double* P, double* pixel) const {
    Eigen::Vector3d P_Camera_eig{P[0], P[1], P[2]};
    bool projection_valid;
    Eigen::Vector2d pixel_projected;
    camera_model_->ProjectPoint(P_Camera_eig, pixel_projected,
                                projection_valid);
    if (!projection_valid) { return false; }
    pixel[0] = pixel_projected[0];
    pixel[1] = pixel_projected[1];
    return true;
  }

  std::shared_ptr<vicon_calibration::CameraModel> camera_model_;
};

struct CeresCameraCostFunction {
  CeresCameraCostFunction(
      const Eigen::Vector2d& pixel_detected, const Eigen::Vector3d& P_Target,
      const Eigen::Matrix4d& T_Robot_Target,
      const std::shared_ptr<vicon_calibration::CameraModel>& camera_model)
      : pixel_detected_(pixel_detected),
        P_Target_(P_Target),
        T_Robot_Target_(T_Robot_Target),
        camera_model_(camera_model) {
    compute_projection.reset(new ceres::CostFunctionToFunctor<2, 3>(
        new ceres::NumericDiffCostFunction<CameraProjectionFunctor,
                                           ceres::CENTRAL, 2, 3>(
            new CameraProjectionFunctor(camera_model_))));
  }

  template <typename T>
  bool operator()(const T* const T_Camera_Robot,
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

    // transform point to camera frame
    T P_Camera[3];
    ceres::QuaternionRotatePoint(T_Camera_Robot, P_Robot, P_Camera);
    P_Camera[0] += T_Camera_Robot[4];
    P_Camera[1] += T_Camera_Robot[5];
    P_Camera[2] += T_Camera_Robot[6];

    const T* P_Camera_Const = &(P_Camera[0]);

    T pixel_projected[2];
    (*compute_projection)(P_Camera_Const, &(pixel_projected[0]));

    residuals[0] = pixel_detected_.cast<T>()[0] - pixel_projected[0];
    residuals[1] = pixel_detected_.cast<T>()[1] - pixel_projected[1];
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(
      const Eigen::Vector2d& pixel_detected, const Eigen::Vector3d& P_Target,
      const Eigen::Matrix4d& T_Robot_Target,
      const std::shared_ptr<vicon_calibration::CameraModel>& camera_model) {
    return (new ceres::AutoDiffCostFunction<CeresCameraCostFunction, 2, 7, 7>(
        new CeresCameraCostFunction(pixel_detected, P_Target, T_Robot_Target,
                                    camera_model)));
  }

  Eigen::Vector2d pixel_detected_;
  Eigen::Vector3d P_Target_;
  Eigen::Matrix4d T_Robot_Target_;
  std::shared_ptr<vicon_calibration::CameraModel> camera_model_;
  std::unique_ptr<ceres::CostFunctionToFunctor<2, 3>> compute_projection;
};