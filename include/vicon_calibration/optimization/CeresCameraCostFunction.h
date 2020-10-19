#include <ceres/autodiff_cost_function.h>
#include <ceres/rotation.h>

struct CameraProjectionFunctor {
  CameraProjectionFunctor(
      const std::shared_ptr<beam_calibration::CameraModel>& camera_model)
      : camera_model_(camera_model) {}

  bool operator()(const double* P, double* pixel) const {
    Eigen::Vector3d P_CAMERA_eig{P[0], P[1], P[2]};
    opt<Eigen::Vector2d> pixel_projected =
        camera_model_->ProjectPointPrecise(P_CAMERA_eig);
    if (!pixel_projected.has_value()) { return false; }
    pixel[0] = pixel_projected.value()[0];
    pixel[1] = pixel_projected.value()[1];
    return true;
  }

  std::shared_ptr<beam_calibration::CameraModel> camera_model_;
};

struct CeresCameraCostFunction {
  CeresCameraCostFunction(
      Eigen::Vector2d pixel_detected, Eigen::Vector3d P_VICONBASE,
      std::shared_ptr<beam_calibration::CameraModel> camera_model)
      : pixel_detected_(pixel_detected),
        P_VICONBASE_(P_VICONBASE),
        camera_model_(camera_model) {
    compute_projection.reset(new ceres::CostFunctionToFunctor<2, 3>(
        new ceres::NumericDiffCostFunction<CameraProjectionFunctor,
                                           ceres::CENTRAL, 2, 3>(
            new CameraProjectionFunctor(camera_model_))));
  }

  template <typename T>
  bool operator()(const T* const T_CR, T* residuals) const {
    // T_CR[0,1,2] are the angle-axis rotation.
    T P_VICONBASE[3];
    P_VICONBASE[0] = P_VICONBASE_.cast<T>()[0];
    P_VICONBASE[1] = P_VICONBASE_.cast<T>()[1];
    P_VICONBASE[2] = P_VICONBASE_.cast<T>()[2];

    T P_CAMERA[3];

    ceres::AngleAxisRotatePoint(T_CR, P_VICONBASE, P_CAMERA);
    // T_CR[3,4,5] are the translation.
    P_CAMERA[0] += T_CR[3];
    P_CAMERA[1] += T_CR[4];
    P_CAMERA[2] += T_CR[5];

    const T P_CAMERA_const = *P_CAMERA;

    T pixel_projected[2];
    (*compute_projection)(&P_CAMERA_const, &pixel_projected[0]);

    residuals[0] = pixel_detected_.cast<T>()[0] - pixel_projected[0];
    residuals[1] = pixel_detected_.cast<T>()[1] - pixel_projected[1];
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(
      const Eigen::Vector2d pixel_detected, const Eigen::Vector3d P_VICONBASE,
      const std::shared_ptr<beam_calibration::CameraModel> camera_model) {
    return (new ceres::AutoDiffCostFunction<CeresCameraCostFunction, 2, 6>(
        new CeresCameraCostFunction(pixel_detected, P_VICONBASE,
                                    camera_model)));
  }

  Eigen::Vector2d pixel_detected_;
  Eigen::Vector3d P_VICONBASE_;
  std::shared_ptr<beam_calibration::CameraModel> camera_model_;
  std::unique_ptr<ceres::CostFunctionToFunctor<2, 3>> compute_projection;
};