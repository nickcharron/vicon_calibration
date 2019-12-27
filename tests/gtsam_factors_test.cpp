#define CATCH_CONFIG_MAIN

#include "vicon_calibration/gtsam/CameraFactor.h"
#include "vicon_calibration/gtsam/CameraLidarFactor.h"
#include "vicon_calibration/gtsam/LidarFactor.h"
#include "vicon_calibration/utils.h"
#include <Eigen/Geometry>
#include <beam_calibration/CameraModel.h>
#include <catch2/catch.hpp>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>

Eigen::Vector2d EvaluateCameraError(
    Eigen::Matrix4d T_op, Eigen::Matrix4d T_VT,
    Eigen::Vector3d corresponding_point, Eigen::Vector2d measured_pixel,
    std::shared_ptr<beam_calibration::CameraModel> camera_model) {
  Eigen::Matrix3d R_VT, R_op;
  Eigen::Vector3d t_VT, t_op, transformed_point;
  R_VT = T_VT.block(0, 0, 3, 3);
  t_VT = T_VT.block(0, 3, 3, 1);
  R_op = T_op.block(0, 0, 3, 3);
  t_op = T_op.block(0, 3, 3, 1);
  transformed_point =
      R_op.transpose() * (R_VT * corresponding_point + t_VT - t_op);
  Eigen::Vector2d projected_point;
  projected_point[0] = camera_model->GetCx() + transformed_point[0] *
                                                   camera_model->GetFx() /
                                                   transformed_point[2];
  projected_point[1] = camera_model->GetCy() + transformed_point[1] *
                                                   camera_model->GetFy() /
                                                   transformed_point[2];
  return Eigen::Vector2d(measured_pixel - projected_point);
}

Eigen::MatrixXd EvaluateCameraJacobian(
    Eigen::Matrix4d T_op, Eigen::Matrix4d T_VT,
    Eigen::Vector3d corresponding_point, Eigen::Vector2d measured_pixel,
    std::shared_ptr<beam_calibration::CameraModel> camera_model) {
  Eigen::Matrix3d R_VT, R_op;
  Eigen::Vector3d t_VT, t_op, transformed_point;
  R_VT = T_VT.block(0, 0, 3, 3);
  t_VT = T_VT.block(0, 3, 3, 1);
  R_op = T_op.block(0, 0, 3, 3);
  t_op = T_op.block(0, 3, 3, 1);
  transformed_point =
      R_op.transpose() * (R_VT * corresponding_point + t_VT - t_op);
  Eigen::Vector2d projected_point;
  projected_point[0] = camera_model->GetCx() + transformed_point[0] *
                                                   camera_model->GetFx() /
                                                   transformed_point[2];
  projected_point[1] = camera_model->GetCy() + transformed_point[1] *
                                                   camera_model->GetFy() /
                                                   transformed_point[2];
  Eigen::MatrixXd H(2, 6), dfdg(2, 3), dgdR(3, 3), dgdt(3, 3);
  dfdg(0, 0) = camera_model->GetFx() / transformed_point[2];
  dfdg(1, 0) = 0;
  dfdg(0, 1) = 0;
  dfdg(1, 1) = camera_model->GetFy() / transformed_point[2];
  dfdg(0, 2) = -transformed_point[0] * camera_model->GetFx() /
               ((transformed_point[2]) * (transformed_point[2]));
  dfdg(1, 2) = -transformed_point[1] * camera_model->GetFy() /
               ((transformed_point[2]) * (transformed_point[2]));
  dgdR = vicon_calibration::utils::SkewTransform(
      R_op.transpose() * (R_VT * corresponding_point + t_VT - t_op));
  dgdt = -R_op.transpose();
  H.block(0, 0, 2, 3) = -dfdg * dgdR;
  H.block(0, 3, 2, 3) = -dfdg * dgdt;
  return H;
}

Eigen::Vector3d EvaluateLidarError(Eigen::Matrix4d T_op, Eigen::Matrix4d T_VT,
                                   Eigen::Vector3d corresponding_point,
                                   Eigen::Vector3d measured_point) {
  Eigen::Matrix3d R_VT, R_op;
  Eigen::Vector3d t_VT, t_op, transformed_point;
  R_VT = T_VT.block(0, 0, 3, 3);
  t_VT = T_VT.block(0, 3, 3, 1);
  R_op = T_op.block(0, 0, 3, 3);
  t_op = T_op.block(0, 3, 3, 1);
  transformed_point =
      R_op.transpose() * (R_VT * corresponding_point + t_VT - t_op);
  return Eigen::Vector3d(measured_point - transformed_point);
}

Eigen::MatrixXd EvaluateLidarJacobian(Eigen::Matrix4d T_op,
                                      Eigen::Matrix4d T_VT,
                                      Eigen::Vector3d corresponding_point,
                                      Eigen::Vector3d measured_point) {
  Eigen::Matrix3d R_VT, R_op;
  Eigen::Vector3d t_VT, t_op, transformed_point;
  R_VT = T_VT.block(0, 0, 3, 3);
  t_VT = T_VT.block(0, 3, 3, 1);
  R_op = T_op.block(0, 0, 3, 3);
  t_op = T_op.block(0, 3, 3, 1);
  transformed_point =
      R_op.transpose() * (R_VT * corresponding_point + t_VT - t_op);
  Eigen::MatrixXd H(3, 6);
  H.block(0, 0, 3, 3) =
      -vicon_calibration::utils::SkewTransform(transformed_point);
  H.block(0, 3, 3, 3) = R_op.transpose();
  return H;
}

Eigen::Vector2d EvaluateCameraLidarError(
    Eigen::Matrix4d T_VL, Eigen::Matrix4d T_VC, Eigen::Vector3d P_T_ci,
    Eigen::Vector3d P_T_li, Eigen::Vector3d measured_point,
    Eigen::Vector2d measured_pixel,
    std::shared_ptr<beam_calibration::CameraModel> camera_model) {
  Eigen::Matrix3d R_VC = T_VC.block(0, 0, 3, 3);
  Eigen::Vector3d t_VC = T_VC.block(0, 3, 3, 1);
  Eigen::Matrix3d R_VL = T_VL.block(0, 0, 3, 3);
  Eigen::Vector3d t_VL = T_VL.block(0, 3, 3, 1);

  Eigen::Vector3d tmp_point = measured_point + P_T_ci - P_T_li;
  Eigen::Vector3d point_transformed =
      R_VC.transpose() * (R_VL * tmp_point + t_VL - t_VC);
  Eigen::Vector2d projected_point;
  projected_point[0] = camera_model->GetCx() + point_transformed[0] *
                                                   camera_model->GetFx() /
                                                   point_transformed[2];
  projected_point[1] = camera_model->GetCy() + point_transformed[1] *
                                                   camera_model->GetFy() /
                                                   point_transformed[2];
  return Eigen::Vector2d(measured_pixel - projected_point);
}

Eigen::MatrixXd EvaluateCameraLidarJacobianL(
    Eigen::Matrix4d T_VL, Eigen::Matrix4d T_VC, Eigen::Vector3d P_T_ci,
    Eigen::Vector3d P_T_li, Eigen::Vector3d measured_point,
    Eigen::Vector2d measured_pixel,
    std::shared_ptr<beam_calibration::CameraModel> camera_model) {

  Eigen::Matrix3d R_VC = T_VC.block(0, 0, 3, 3);
  Eigen::Vector3d t_VC = T_VC.block(0, 3, 3, 1);
  Eigen::Matrix3d R_VL = T_VL.block(0, 0, 3, 3);
  Eigen::Vector3d t_VL = T_VL.block(0, 3, 3, 1);

  Eigen::Vector3d tmp_point = measured_point + P_T_ci - P_T_li;
  Eigen::Vector3d point_transformed =
      R_VC.transpose() * (R_VL * tmp_point + t_VL - t_VC);
  Eigen::Vector2d projected_point;
  projected_point[0] = camera_model->GetCx() + point_transformed[0] *
                                                   camera_model->GetFx() /
                                                   point_transformed[2];
  projected_point[1] = camera_model->GetCy() + point_transformed[1] *
                                                   camera_model->GetFy() /
                                                   point_transformed[2];

  Eigen::MatrixXd H(2, 6), dfdg(2, 3), dgdR(3, 3), dgdt(3, 3), dedf(2, 2);
  dfdg(0, 0) = camera_model->GetFx() / point_transformed[2];
  dfdg(1, 0) = 0;
  dfdg(0, 1) = 0;
  dfdg(1, 1) = camera_model->GetFy() / point_transformed[2];
  dfdg(0, 2) = -point_transformed[0] * camera_model->GetFx() /
               ((point_transformed[2]) * (point_transformed[2]));
  dfdg(1, 2) = -point_transformed[1] * camera_model->GetFy() /
               ((point_transformed[2]) * (point_transformed[2]));

  dgdR = R_VC.transpose() * R_VL *
         vicon_calibration::utils::SkewTransform(-1 * tmp_point);
  dgdt = R_VC.transpose();
  dedf.setIdentity();
  dedf = -1 * dedf;
  H.block(0, 0, 2, 3) = dedf * dfdg * dgdR;
  H.block(0, 3, 2, 3) = dedf * dfdg * dgdt;
  return H;
}

Eigen::MatrixXd EvaluateCameraLidarJacobianC(
    Eigen::Matrix4d T_VL, Eigen::Matrix4d T_VC, Eigen::Vector3d P_T_ci,
    Eigen::Vector3d P_T_li, Eigen::Vector3d measured_point,
    Eigen::Vector2d measured_pixel,
    std::shared_ptr<beam_calibration::CameraModel> camera_model) {

  Eigen::Matrix3d R_VC = T_VC.block(0, 0, 3, 3);
  Eigen::Vector3d t_VC = T_VC.block(0, 3, 3, 1);
  Eigen::Matrix3d R_VL = T_VL.block(0, 0, 3, 3);
  Eigen::Vector3d t_VL = T_VL.block(0, 3, 3, 1);

  Eigen::Vector3d tmp_point = measured_point + P_T_ci - P_T_li;
  Eigen::Vector3d point_transformed =
      R_VC.transpose() * (R_VL * tmp_point + t_VL - t_VC);
  Eigen::Vector2d projected_point;
  projected_point[0] = camera_model->GetCx() + point_transformed[0] *
                                                   camera_model->GetFx() /
                                                   point_transformed[2];
  projected_point[1] = camera_model->GetCy() + point_transformed[1] *
                                                   camera_model->GetFy() /
                                                   point_transformed[2];

  Eigen::MatrixXd H(2, 6), dfdg(2, 3), dgdR(3, 3), dgdt(3, 3), dedf(2, 2);
  dfdg(0, 0) = camera_model->GetFx() / point_transformed[2];
  dfdg(1, 0) = 0;
  dfdg(0, 1) = 0;
  dfdg(1, 1) = camera_model->GetFy() / point_transformed[2];
  dfdg(0, 2) = -point_transformed[0] * camera_model->GetFx() /
               ((point_transformed[2]) * (point_transformed[2]));
  dfdg(1, 2) = -point_transformed[1] * camera_model->GetFy() /
               ((point_transformed[2]) * (point_transformed[2]));

  dgdR = vicon_calibration::utils::SkewTransform(
      R_VC.transpose() * (R_VL * tmp_point + t_VL - t_VC));
  dgdt = -1 * R_VC.transpose();
  dedf.setIdentity();
  dedf = -1 * dedf;
  H.block(0, 0, 2, 3) = dedf * dfdg * dgdR;
  H.block(0, 3, 2, 3) = dedf * dfdg * dgdt;
  return H;
}

TEST_CASE("Test Camera Factor Error and Jacobian") {
  // Create intrinsics
  std::string camera_model_location = __FILE__;
  camera_model_location.erase(camera_model_location.end() - 22,
                              camera_model_location.end());
  camera_model_location += "data/CamFactorIntrinsics.json";
  std::shared_ptr<beam_calibration::CameraModel> camera_model =
      beam_calibration::CameraModel::LoadJSON(camera_model_location);

  // Create Transforms
  Eigen::Matrix4d T_VC, T_CV, T_CT, T_VT;
  T_VC.setIdentity();
  T_CT.setIdentity();
  T_VC.block(0, 3, 3, 1) = Eigen::Vector3d(1, 1, 0);
  T_CT.block(0, 3, 3, 1) = Eigen::Vector3d(0.2, 0.2, 3);
  T_CV = T_VC.inverse();
  T_VT = T_VC * T_CT;

  // Create Points
  Eigen::Vector4d point_homo(0.2, 0.2, 1, 1);
  Eigen::Vector4d point_transformed_homo = T_CT * point_homo;
  Eigen::Vector3d point_transformed =
      vicon_calibration::utils::HomoPointToPoint(point_transformed_homo);
  Eigen::Vector3d point =
      vicon_calibration::utils::HomoPointToPoint(point_homo);

  Eigen::Vector2d pixel_measured;
  pixel_measured[0] = camera_model->GetCx() + point_transformed[0] *
                                                  camera_model->GetFx() /
                                                  point_transformed[2];
  pixel_measured[1] = camera_model->GetCy() + point_transformed[1] *
                                                  camera_model->GetFy() /
                                                  point_transformed[2];

  // calculate numerical Jacobian
  double eps = std::sqrt(1e-15);
  Eigen::MatrixXd p(6, 6);
  p.setIdentity();
  p = p * eps;
  Eigen::MatrixXd J_numerical(2, 6);
  Eigen::Matrix4d T_perturb;

  for (int i = 0; i < 6; i++) {
    T_perturb =
        vicon_calibration::utils::PerturbTransform(T_VC, p.block(0, i, 6, 1));
    J_numerical.block(0, i, 2, 1) =
        EvaluateCameraError(T_perturb, T_VT, point, pixel_measured,
                            camera_model) /
        eps;
  }

  // calculate analytical jacobian
  Eigen::MatrixXd J_analytical(2, 6);
  J_analytical = EvaluateCameraJacobian(T_perturb, T_VT, point, pixel_measured,
                                        camera_model);
  REQUIRE(vicon_calibration::utils::RoundMatrix(J_numerical, 4) ==
          vicon_calibration::utils::RoundMatrix(J_analytical, 4));
}

TEST_CASE("Test Lidar Factor Error and Jacobian") {
  // Create Transforms
  Eigen::Matrix4d T_VL, T_LV, T_LT, T_VT;
  T_VL.setIdentity();
  T_LT.setIdentity();
  T_VL.block(0, 3, 3, 1) = Eigen::Vector3d(1.1, 1.2, 1.3);
  T_LT.block(0, 3, 3, 1) = Eigen::Vector3d(2, 3, 4);
  T_LV = T_VL.inverse();
  T_VT = T_VL * T_LT;

  // Create Points
  Eigen::Vector4d point_homo(0.3, 0.2, 0.1, 1);
  Eigen::Vector4d point_measured_homo = T_LT * point_homo;
  Eigen::Vector3d point =
      vicon_calibration::utils::HomoPointToPoint(point_homo);
  Eigen::Vector3d point_measured =
      vicon_calibration::utils::HomoPointToPoint(point_measured_homo);

  // calculate numerical Jacobian
  double eps = std::sqrt(1e-15);
  Eigen::MatrixXd p(6, 6);
  p.setIdentity();
  p = p * eps;
  Eigen::MatrixXd J_numerical(3, 6);
  Eigen::Matrix4d T_perturb;

  for (int i = 0; i < 6; i++) {
    T_perturb =
        vicon_calibration::utils::PerturbTransform(T_VL, p.block(0, i, 6, 1));
    J_numerical.block(0, i, 3, 1) =
        EvaluateLidarError(T_perturb, T_VT, point, point_measured) / eps;
  }

  // calculate analytical jacobian
  Eigen::MatrixXd J_analytical(3, 6);
  J_analytical = EvaluateLidarJacobian(T_perturb, T_VT, point, point_measured);
  REQUIRE(vicon_calibration::utils::RoundMatrix(J_numerical, 6) ==
          vicon_calibration::utils::RoundMatrix(J_analytical, 6));
}

TEST_CASE("Test Camera-Lidar Factor Error and Jacobian") {
  // Create intrinsics
  std::string camera_model_location = __FILE__;
  camera_model_location.erase(camera_model_location.end() - 22,
                              camera_model_location.end());
  camera_model_location += "data/CamFactorIntrinsics.json";
  std::shared_ptr<beam_calibration::CameraModel> camera_model =
      beam_calibration::CameraModel::LoadJSON(camera_model_location);

  // Create Transforms
  Eigen::Matrix4d T_VC, T_CV, T_VL, T_LV, T_CT, T_VT;
  T_VC.setIdentity();
  T_CT.setIdentity();
  T_VL.setIdentity();
  T_VC.block(0, 3, 3, 1) = Eigen::Vector3d(1, 1, 0);
  T_CT.block(0, 3, 3, 1) = Eigen::Vector3d(0.2, 0.2, 3);
  T_VL.block(0, 3, 3, 1) =
      T_VC.block(0, 3, 3, 1) + Eigen::Vector3d(0.2, 0.2, 0);
  T_CV = T_VC.inverse();
  T_LV = T_VL.inverse();
  T_VT = T_VC * T_CT;

  // Create Points
  Eigen::Vector4d P_T_homo(0.2, 0.2, 1, 1);
  Eigen::Vector4d P_C_homo = T_CT * P_T_homo;
  Eigen::Vector4d P_L_homo = T_LV * T_VT * P_T_homo;
  Eigen::Vector3d P_T = vicon_calibration::utils::HomoPointToPoint(P_T_homo);
  Eigen::Vector3d P_C = vicon_calibration::utils::HomoPointToPoint(P_C_homo);
  Eigen::Vector3d P_L = vicon_calibration::utils::HomoPointToPoint(P_L_homo);
  Eigen::Vector3d point_measured = P_L;
  Eigen::Vector2d pixel_measured;
  pixel_measured[0] =
      camera_model->GetCx() + P_C[0] * camera_model->GetFx() / P_C[2];
  pixel_measured[1] =
      camera_model->GetCy() + P_C[1] * camera_model->GetFy() / P_C[2];

  // calculate numerical Jacobian
  double eps = std::sqrt(1e-15);
  Eigen::MatrixXd p(6, 6);
  p.setIdentity();
  p = p * eps;
  Eigen::MatrixXd JC_numerical(2, 6);
  Eigen::MatrixXd JL_numerical(2, 6);
  Eigen::Matrix4d T_VC_perturb, T_VL_perturb;

  for (int i = 0; i < 6; i++) {
    T_VC_perturb =
        vicon_calibration::utils::PerturbTransform(T_VC, p.block(0, i, 6, 1));
    T_VL_perturb =
        vicon_calibration::utils::PerturbTransform(T_VL, p.block(0, i, 6, 1));
    JC_numerical.block(0, i, 2, 1) =
        EvaluateCameraLidarError(T_VL, T_VC_perturb, P_T, P_T, point_measured,
                                 pixel_measured, camera_model) / eps;
    JL_numerical.block(0, i, 2, 1) =
        EvaluateCameraLidarError(T_VL_perturb, T_VC, P_T, P_T, point_measured,
                                 pixel_measured, camera_model) / eps;
  }

  // calculate analytical jacobian
  Eigen::MatrixXd JL_analytical(2, 6), JC_analytical(2, 6);
  JC_analytical = EvaluateCameraLidarJacobianC(
      T_VL, T_VC, P_T, P_T, point_measured, pixel_measured, camera_model);
  JL_analytical = EvaluateCameraLidarJacobianL(
      T_VL, T_VC, P_T, P_T, point_measured, pixel_measured, camera_model);

  REQUIRE(vicon_calibration::utils::RoundMatrix(JC_numerical, 4) ==
          vicon_calibration::utils::RoundMatrix(JC_analytical, 4));
  REQUIRE(vicon_calibration::utils::RoundMatrix(JL_numerical, 4) ==
          vicon_calibration::utils::RoundMatrix(JL_analytical, 4));
}

TEST_CASE("Test camera factor in Optimization") {
  // create keypoints
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      points;
  double max_distance_x = 2, max_distance_y = 2, max_distance_z = 4;
  for (int i = 0; i < 20; i++) {
    double x = ((double)std::rand() / (RAND_MAX)-0.5) * 2 * max_distance_x;
    double y = ((double)std::rand() / (RAND_MAX)-0.5) * 2 * max_distance_y;
    double z = ((double)std::rand() / (RAND_MAX)-0) * 1 * max_distance_z;
    Eigen::Vector4d point(x, y, z, 1);
    points.push_back(point);
  }

  // Create intrinsics
  std::string camera_model_location = __FILE__;
  camera_model_location.erase(camera_model_location.end() - 22,
                              camera_model_location.end());
  camera_model_location += "data/CamFactorIntrinsics.json";
  std::shared_ptr<beam_calibration::CameraModel> camera_model =
      beam_calibration::CameraModel::LoadJSON(camera_model_location);

  // Create Transforms
  Eigen::Matrix4d T_VC, T_CV, T_CT, T_VT;
  T_VC.setIdentity();
  T_CT.setIdentity();
  T_VC.block(0, 3, 3, 1) = Eigen::Vector3d(1, 1, 0);
  T_CT.block(0, 3, 3, 1) = Eigen::Vector3d(0.2, 0.2, 3);
  T_CV = T_VC.inverse();
  T_VT = T_VC * T_CT;

  // create perturbed initial
  Eigen::Matrix4d T_VC_pert;
  Eigen::VectorXd perturbation(6, 1);
  perturbation << 0.3, -0.3, 0.3, 0.5, -0.5, 0.3;
  T_VC_pert = vicon_calibration::utils::PerturbTransform(T_VC, perturbation);

  // create projected (detected) points - no noise
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      pixels(points.size());
  std::vector<bool> pixels_valid(points.size());
  Eigen::Vector4d point_transformed;
  Eigen::Vector2d pixel;
  int count = 0;
  for (int i = 0; i < points.size(); i++) {
    point_transformed = T_CV * T_VT * points[i];
    pixel = camera_model->ProjectUndistortedPoint(
        vicon_calibration::utils::HomoPointToPoint(point_transformed));
    // pixel[0] = camera_model->GetCx() + point_transformed[0] *
    //                                        camera_model->GetFx() /
    //                                        point_transformed[2];
    // pixel[1] = camera_model->GetCy() + point_transformed[1] *
    //                                        camera_model->GetFy() /
    //                                        point_transformed[2];
    pixels[i] = pixel;
    if (camera_model->PixelInImage(pixel)) {
      pixels_valid[i] = true;
      count++;
    } else {
      pixels_valid[i] = false;
    }
  }

  // build graph
  gtsam::NonlinearFactorGraph graph;

  // add factors
  gtsam::Key key = gtsam::Symbol('C', 1);
  gtsam::Vector2 noise_vec;
  noise_vec << 1, 1;
  gtsam::noiseModel::Diagonal::shared_ptr ImageNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  Eigen::Vector3d pti;
  for (int i = 0; i < points.size(); i++) {
    if (pixels_valid[i]) {
      pti = vicon_calibration::utils::HomoPointToPoint(points[i]);
      graph.emplace_shared<vicon_calibration::CameraFactor>(
          key, pixels[i], pti, camera_model, T_VT, ImageNoise);
    }
  }
  gtsam::LevenbergMarquardtParams params;
  params.setVerbosity("SILENT");
  params.absoluteErrorTol = 1e-9;
  params.relativeErrorTol = 1e-9;
  params.setlambdaUpperBound(1e5);
  gtsam::KeyFormatter key_formatter = gtsam::DefaultKeyFormatter;

  // solve with perfect initials
  gtsam::Values initials_exact, results1;
  gtsam::Pose3 initial_pose1(T_VC);
  initials_exact.insert(key, initial_pose1);
  gtsam::LevenbergMarquardtOptimizer optimizer1(graph, initials_exact, params);
  results1 = optimizer1.optimize();
  Eigen::Matrix4d T_VC_opt1 =
      results1.at<gtsam::Pose3>(gtsam::Symbol('C', 1)).matrix();
  // std::cout << "\n-------------------------\n";
  // std::cout << "PRINTING EXACT INITIALS: \n";
  // initials_exact.print();
  // graph.print();
  // std::cout << "\nPRINTING RESULTS: \n";
  // results1.print();

  // solve with perturbed initials
  gtsam::Values initials_pert, results2;
  gtsam::Pose3 initial_pose2(T_VC_pert);
  initials_pert.insert(key, initial_pose2);
  gtsam::LevenbergMarquardtOptimizer optimizer2(graph, initials_pert, params);
  results2 = optimizer2.optimize();
  Eigen::Matrix4d T_VC_opt2 =
      results2.at<gtsam::Pose3>(gtsam::Symbol('C', 1)).matrix();
  // std::cout << "\n----------------------------\n";
  // std::cout << "PRINTING PERTURBED INITIALS : \n ";
  // initials_pert.print();
  // graph.print();
  // std::cout << "\nPRINTING RESULTS: \n";
  // results2.print();

  REQUIRE(T_VC == vicon_calibration::utils::RoundMatrix(T_VC_opt1, 5));
  REQUIRE(T_VC == vicon_calibration::utils::RoundMatrix(T_VC_opt2, 5));
}

TEST_CASE("Test lidar factor in Optimization") {
  // create keypoints
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      points;
  double x, y, z, max_distance = 3;
  for (int i = 0; i < 30; i++) {
    x = ((double)std::rand() / (RAND_MAX)-0.5) * 2 * max_distance;
    y = ((double)std::rand() / (RAND_MAX)-0.5) * 2 * max_distance;
    z = ((double)std::rand() / (RAND_MAX)-0.5) * 2 * max_distance;
    Eigen::Vector4d point(x, y, z, 1);
    points.push_back(point);
  }

  // Create Transforms
  Eigen::Matrix4d T_VL, T_LV, T_LT, T_VT;
  T_VL.setIdentity();
  T_LT.setIdentity();
  T_VL.block(0, 3, 3, 1) = Eigen::Vector3d(1.1, 1.2, 1.3);
  T_LT.block(0, 3, 3, 1) = Eigen::Vector3d(2, 3, 4);
  T_LV = T_VL.inverse();
  T_VT = T_VL * T_LT;

  // create perturbed initial
  Eigen::Matrix4d T_VL_pert;
  Eigen::VectorXd perturbation(6, 1);
  perturbation << 0.3, -0.3, 0.3, 0.5, -0.5, 0.3;
  T_VL_pert = vicon_calibration::utils::PerturbTransform(T_VL, perturbation);

  // create transformed (detected) points - no noise
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      points_measured(points.size());
  Eigen::Vector4d point_transformed;
  for (int i = 0; i < points.size(); i++) {
    points_measured[i] = T_LV * T_VT * points[i];
  }

  // build graph
  gtsam::NonlinearFactorGraph graph;

  // add factors
  gtsam::Key key = gtsam::Symbol('L', 1);
  gtsam::Vector3 noise_vec;
  noise_vec << 0.01, 0.01, 0.01;
  gtsam::noiseModel::Diagonal::shared_ptr ScanNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  Eigen::Vector3d pti, ptmi;
  for (int i = 0; i < points.size(); i++) {
    pti = vicon_calibration::utils::HomoPointToPoint(points[i]);
    ptmi = vicon_calibration::utils::HomoPointToPoint(points_measured[i]);
    graph.emplace_shared<vicon_calibration::LidarFactor>(key, ptmi, pti, T_VT,
                                                         ScanNoise);
  }
  gtsam::LevenbergMarquardtParams params;
  params.setVerbosity("SILENT");
  params.absoluteErrorTol = 1e-9;
  params.relativeErrorTol = 1e-9;
  params.setlambdaUpperBound(1e10);
  gtsam::KeyFormatter key_formatter = gtsam::DefaultKeyFormatter;

  // solve with perfect initials
  gtsam::Values initials_exact, results1;
  gtsam::Pose3 initial_pose1(T_VL);
  initials_exact.insert(key, initial_pose1);
  gtsam::LevenbergMarquardtOptimizer optimizer1(graph, initials_exact, params);
  results1 = optimizer1.optimize();
  Eigen::Matrix4d T_VL_opt1 =
      results1.at<gtsam::Pose3>(gtsam::Symbol('L', 1)).matrix();

  // std::cout << "\n-------------------------\n";
  // std::cout << "PRINTING EXACT INITIALS: \n";
  // initials_exact.print();
  // graph.print();
  // std::cout << "\nPRINTING RESULTS: \n";
  // results1.print();

  // solve with perturbed initials
  gtsam::Values initials_pert, results2;
  gtsam::Pose3 initial_pose2(T_VL_pert);
  initials_pert.insert(key, initial_pose2);
  gtsam::LevenbergMarquardtOptimizer optimizer2(graph, initials_pert, params);
  results2 = optimizer2.optimize();
  Eigen::Matrix4d T_VL_opt2 =
      results2.at<gtsam::Pose3>(gtsam::Symbol('L', 1)).matrix();

  // std::cout << "\n----------------------------\n";
  // std::cout << "PRINTING PERTURBED INITIALS: \n";
  // initials_pert.print();
  // graph.print();
  // std::cout << "\nPRINTING RESULTS: \n";
  // results2.print();

  REQUIRE(T_VL == vicon_calibration::utils::RoundMatrix(T_VL_opt1, 5));
  REQUIRE(T_VL == vicon_calibration::utils::RoundMatrix(T_VL_opt2, 5));
}

TEST_CASE("Test Camera-Lidar factor in Optimization") {
  // create keypoints
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      points;
  double max_distance_x = 3, max_distance_y = 3, max_distance_z = 4;
  for (int i = 0; i < 30; i++) {
    double x = ((double)std::rand() / (RAND_MAX)-0.5) * 2 * max_distance_x;
    double y = ((double)std::rand() / (RAND_MAX)-0.5) * 2 * max_distance_y;
    double z = ((double)std::rand() / (RAND_MAX)-0) * 1 * max_distance_z;
    Eigen::Vector4d point(x, y, z, 1);
    points.push_back(point);
  }

  // Create intrinsics
  std::string camera_model_location = __FILE__;
  camera_model_location.erase(camera_model_location.end() - 22,
                              camera_model_location.end());
  camera_model_location += "data/CamFactorIntrinsics.json";
  std::shared_ptr<beam_calibration::CameraModel> camera_model =
      beam_calibration::CameraModel::LoadJSON(camera_model_location);
  double Fx = camera_model->GetFx();
  double Fy = camera_model->GetFy();
  double Cx = camera_model->GetCx();
  double Cy = camera_model->GetCy();

  // Create Transforms
  Eigen::Matrix4d T_VC, T_CV, T_VL, T_LV, T_CT, T_VT;
  T_VC.setIdentity();
  T_CT.setIdentity();
  T_VL.setIdentity();
  T_VC.block(0, 3, 3, 1) = Eigen::Vector3d(1, 1, 0);
  T_CT.block(0, 3, 3, 1) = Eigen::Vector3d(0.2, 0.2, 3);
  T_VL.block(0, 3, 3, 1) =
      T_VC.block(0, 3, 3, 1) + Eigen::Vector3d(0.2, 0.2, 0);
  T_CV = T_VC.inverse();
  T_LV = T_VL.inverse();
  T_VT = T_VC * T_CT;

  // create perturbed initial
  Eigen::Matrix4d T_VC_pert, T_VL_pert;
  Eigen::VectorXd perturb_cam(6, 1), perturb_lid(6, 1);
  perturb_cam << 0.3, -0.3, 0.3, 0.5, -0.5, 0.3;
  perturb_lid << -0.3, 0.3, -0.3, -0.5, 0.5, -0.3;
  T_VC_pert = vicon_calibration::utils::PerturbTransform(T_VC, perturb_cam);
  T_VL_pert = vicon_calibration::utils::PerturbTransform(T_VL, perturb_lid);

  // create measured pixels and measured lidar points - no noise
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      pixels_measured(points.size());
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      points_measured(points.size());
  std::vector<bool> pixels_valid(points.size());

  Eigen::Vector2d pixel;
  for (int i = 0; i < points.size(); i++) {
    Eigen::Vector4d point_transformed = T_CT * points[i];
    pixel[0] = Cx + point_transformed[0] * Fx / point_transformed[2];
    pixel[1] = Cy + point_transformed[1] * Fy / point_transformed[2];
    pixels_measured[i] = pixel;
    points_measured[i] = T_LV * T_VT * points[i];
    if (camera_model->PixelInImage(pixel)) {
      pixels_valid[i] = true;
    } else {
      pixels_valid[i] = false;
    }
  }

  // build graph
  gtsam::NonlinearFactorGraph graph;

  // add binary factors between lidar and camera
  gtsam::Key camera_key = gtsam::Symbol('C', 1);
  gtsam::Key lidar_key = gtsam::Symbol('L', 1);
  gtsam::Vector2 camera_noise_vec;
  camera_noise_vec << 1, 1;
  gtsam::noiseModel::Diagonal::shared_ptr ImageNoise =
      gtsam::noiseModel::Diagonal::Sigmas(camera_noise_vec);

  Eigen::Vector3d pti, ptmi;
  for (int i = 0; i < points.size(); i++) {
    if (pixels_valid[i]) {
      pti = vicon_calibration::utils::HomoPointToPoint(points[i]);
      ptmi = vicon_calibration::utils::HomoPointToPoint(points_measured[i]);
      graph.emplace_shared<vicon_calibration::CameraLidarFactor>(
          lidar_key, camera_key, pixels_measured[i], ptmi, pti, pti,
          camera_model, ImageNoise);
    }
  }

  // Add unary factors on lidar where points were not projected to image plane
  // We need unary factors on one of the frames to fully constrain the problem
  gtsam::Vector3 lidar_noise_vec;
  lidar_noise_vec << 0.01, 0.01, 0.01;
  gtsam::noiseModel::Diagonal::shared_ptr ScanNoise =
      gtsam::noiseModel::Diagonal::Sigmas(lidar_noise_vec);
  for (int i = 0; i < points.size(); i++) {
    if (!pixels_valid[i]) {
      pti = vicon_calibration::utils::HomoPointToPoint(points[i]);
      ptmi = vicon_calibration::utils::HomoPointToPoint(points_measured[i]);
      graph.emplace_shared<vicon_calibration::LidarFactor>(lidar_key, ptmi, pti,
                                                           T_VT, ScanNoise);
    }
  }

  gtsam::LevenbergMarquardtParams params;
  params.setVerbosity("SILENT");
  params.absoluteErrorTol = 1e-9;
  params.relativeErrorTol = 1e-9;
  params.setlambdaUpperBound(1e5);
  gtsam::KeyFormatter key_formatter = gtsam::DefaultKeyFormatter;

  // solve with perfect initials
  gtsam::Values initials_exact, results1;
  gtsam::Pose3 initial_pose_cam1(T_VC), initial_pose_lid1(T_VL);
  initials_exact.insert(camera_key, initial_pose_cam1);
  initials_exact.insert(lidar_key, initial_pose_lid1);
  gtsam::LevenbergMarquardtOptimizer optimizer1(graph, initials_exact, params);
  results1 = optimizer1.optimize();
  Eigen::Matrix4d T_VC_opt1 = results1.at<gtsam::Pose3>(camera_key).matrix();
  Eigen::Matrix4d T_VL_opt1 = results1.at<gtsam::Pose3>(lidar_key).matrix();
  // std::cout << "\n-------------------------\n";
  // std::cout << "PRINTING EXACT INITIALS: \n";
  // initials_exact.print();
  // std::string output_file = "/home/nick/gtsam_results/graph.xdot";
  // std::ofstream graph_file(output_file);
  // graph.saveGraph(graph_file);
  // graph.print();
  // std::cout << "\nPRINTING RESULTS: \n";
  // results1.print();

  // solve with perturbed initials
  gtsam::Values initials_pert, results2;
  gtsam::Pose3 initial_pose_cam2(T_VC_pert), initial_pose_lid2(T_VL_pert);
  initials_pert.insert(camera_key, initial_pose_cam2);
  initials_pert.insert(lidar_key, initial_pose_lid2);
  gtsam::LevenbergMarquardtOptimizer optimizer2(graph, initials_pert, params);
  results2 = optimizer2.optimize();
  Eigen::Matrix4d T_VC_opt2 = results1.at<gtsam::Pose3>(camera_key).matrix();
  Eigen::Matrix4d T_VL_opt2 = results1.at<gtsam::Pose3>(lidar_key).matrix();
  // std::cout << "\n----------------------------\n";
  // std::cout << "PRINTING PERTURBED INITIALS : \n ";
  // initials_pert.print();
  // graph.print();
  // std::cout << "\nPRINTING RESULTS: \n";
  // results2.print();

  REQUIRE(T_VC == vicon_calibration::utils::RoundMatrix(T_VC_opt1, 5));
  REQUIRE(T_VC == vicon_calibration::utils::RoundMatrix(T_VC_opt2, 5));
  REQUIRE(T_VL == vicon_calibration::utils::RoundMatrix(T_VL_opt1, 5));
  REQUIRE(T_VL == vicon_calibration::utils::RoundMatrix(T_VL_opt2, 5));
}
