#define CATCH_CONFIG_MAIN

#include "vicon_calibration/optimization/GtsamCameraFactor.h"
#include "vicon_calibration/optimization/GtsamCameraFactorInv.h"
#include "vicon_calibration/optimization/GtsamCameraLidarFactor.h"
#include "vicon_calibration/optimization/GtsamLidarFactor.h"
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

using AlignVec2d = Eigen::aligned_allocator<Eigen::Vector2d>;

void OutputJacobians(const Eigen::MatrixXd& J_analytical,
                     const Eigen::MatrixXd& J_numerical,
                     const std::string& test_case) {
  std::cout << "Outputting jacobians for test case: " << test_case << "\n"
            << "J_analytical: \n"
            << J_analytical << "\n"
            << "J_numerical: \n"
            << J_numerical << "\n";
  std::cout << "Norm error: " << std::setprecision(10)
            << (J_analytical - J_numerical).norm() << "\n";
}

TEST_CASE("Test Camera Factor Error and Jacobian") {
  // Create intrinsics
  std::string camera_model_location = __FILE__;
  camera_model_location.erase(camera_model_location.end() - 22,
                              camera_model_location.end());
  camera_model_location += "data/CamFactorIntrinsics.json";
  std::shared_ptr<beam_calibration::CameraModel> camera_model =
      beam_calibration::CameraModel::Create(camera_model_location);

  // Create Transforms
  Eigen::Matrix4d T_VC = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_CT = Eigen::Matrix4d::Identity();
  T_VC.block(0, 3, 3, 1) = Eigen::Vector3d(1, 1, 0);
  T_CT.block(0, 3, 3, 1) = Eigen::Vector3d(0.2, 0.2, 3);
  Eigen::Matrix4d T_CV = T_VC.inverse();
  Eigen::Matrix4d T_VT = T_VC * T_CT;

  // Create Points
  Eigen::Vector3d point(0.2, 0.2, 1);
  Eigen::Vector4d point_transformed = T_CT * point.homogeneous();
  opt<Eigen::Vector2d> pixel_measured =
      camera_model->ProjectPointPrecise(point_transformed.hnormalized());

  // create factor
  gtsam::Key dummy_key(gtsam::symbol('C', 0));
  gtsam::Vector2 noise_vec(10, 10);
  gtsam::noiseModel::Diagonal::shared_ptr ImageNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  gtsam::Pose3 pose(T_CV);
  vicon_calibration::CameraFactor factor(dummy_key, pixel_measured.value(),
                                         point, camera_model, T_VT, ImageNoise);

  // calculate numerical Jacobian
  double eps = std::sqrt(1e-8);
  Eigen::MatrixXd p(6, 6);
  p.setIdentity();
  p = p * eps;
  Eigen::MatrixXd J_numerical(2, 6);
  for (int i = 0; i < 6; i++) {
    Eigen::Matrix4d T_perturb = vicon_calibration::utils::PerturbTransformRadM(
        T_CV, p.block(0, i, 6, 1));
    gtsam::Pose3 pose_perturbed(T_perturb);
    J_numerical.block(0, i, 2, 1) =
        (factor.evaluateError(pose_perturbed) - factor.evaluateError(pose)) /
        eps;
  }

  // calculate analytical jacobian
  gtsam::Matrix J_analytical;
  boost::optional<gtsam::Matrix> J_opt(J_analytical);
  factor.evaluateError(pose, J_analytical);
  REQUIRE((J_numerical - J_analytical).norm() < 5e-4);
}

TEST_CASE("Test Camera Factor Inv Error and Jacobian") {
  // Create intrinsics
  std::string camera_model_location = __FILE__;
  camera_model_location.erase(camera_model_location.end() - 22,
                              camera_model_location.end());
  camera_model_location += "data/CamFactorIntrinsics.json";
  std::shared_ptr<beam_calibration::CameraModel> camera_model =
      beam_calibration::CameraModel::Create(camera_model_location);

  // Create Transforms
  Eigen::Matrix4d T_VC = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_CT = Eigen::Matrix4d::Identity();
  T_VC.block(0, 3, 3, 1) = Eigen::Vector3d(1, 1, 0);
  T_CT.block(0, 3, 3, 1) = Eigen::Vector3d(0.2, 0.2, 3);
  Eigen::Matrix4d T_CV = T_VC.inverse();
  Eigen::Matrix4d T_VT = T_VC * T_CT;

  // Create Points
  Eigen::Vector3d point(0.2, 0.2, 1);
  Eigen::Vector4d point_transformed = T_CT * point.homogeneous();
  opt<Eigen::Vector2d> pixel_measured =
      camera_model->ProjectPointPrecise(point_transformed.hnormalized());

  // create factor
  gtsam::Key dummy_key(gtsam::symbol('C', 0));
  gtsam::Vector2 noise_vec(10, 10);
  gtsam::noiseModel::Diagonal::shared_ptr ImageNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  gtsam::Pose3 pose(T_VC);
  vicon_calibration::CameraFactorInv factor(
      dummy_key, pixel_measured.value(), point, camera_model, T_VT, ImageNoise);

  // calculate numerical Jacobian
  double eps = std::sqrt(1e-8);
  Eigen::MatrixXd p(6, 6);
  p.setIdentity();
  p = p * eps;
  Eigen::MatrixXd J_numerical(2, 6);
  for (int i = 0; i < 6; i++) {
    Eigen::Matrix4d T_perturb = vicon_calibration::utils::PerturbTransformRadM(
        T_VC, p.block(0, i, 6, 1));
    gtsam::Pose3 pose_perturbed(T_perturb);
    J_numerical.block(0, i, 2, 1) =
        (factor.evaluateError(pose_perturbed) - factor.evaluateError(pose)) /
        eps;
  }

  // calculate analytical jacobian
  gtsam::Matrix J_analytical;
  boost::optional<gtsam::Matrix> J_opt(J_analytical);
  factor.evaluateError(pose, J_analytical);
  REQUIRE((J_analytical - J_numerical).norm() < 5e-4);
}

TEST_CASE("Test Lidar Factor Error and Jacobian") {
  // Create Transforms
  Eigen::Matrix4d T_VL = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_LT = Eigen::Matrix4d::Identity();
  T_VL.block(0, 3, 3, 1) = Eigen::Vector3d(1.1, 1.2, 1.3);
  T_LT.block(0, 3, 3, 1) = Eigen::Vector3d(2, 3, 4);
  Eigen::Matrix4d T_LV = T_VL.inverse();
  Eigen::Matrix4d T_VT = T_VL * T_LT;

  // Create Points
  Eigen::Vector3d point(0.3, 0.2, 0.1);
  Eigen::Vector4d point_measured = T_LT * point.homogeneous();

  // create factor
  gtsam::Key dummy_key(gtsam::symbol('L', 0));
  gtsam::Vector3 noise_vec(0.01, 0.01, 0.01);
  gtsam::noiseModel::Diagonal::shared_ptr LidarNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  gtsam::Pose3 pose(T_VL);
  vicon_calibration::LidarFactor factor(dummy_key, point_measured.hnormalized(),
                                        point, T_VT, LidarNoise);

  // calculate numerical Jacobian
  double eps = std::sqrt(1e-8);
  Eigen::MatrixXd p(6, 6);
  p.setIdentity();
  p = p * eps;
  Eigen::MatrixXd J_numerical(3, 6);

  for (int i = 0; i < 6; i++) {
    Eigen::Matrix4d T_perturb = vicon_calibration::utils::PerturbTransformRadM(
        T_VL, p.block(0, i, 6, 1));
    gtsam::Pose3 pose_perturbed(T_perturb);
    J_numerical.block(0, i, 3, 1) =
        (factor.evaluateError(pose_perturbed) - factor.evaluateError(pose)) /
        eps;
  }

  // calculate analytical jacobian
  gtsam::Matrix J_analytical;
  boost::optional<gtsam::Matrix> J_opt(J_analytical);
  factor.evaluateError(pose, J_analytical);
  REQUIRE((J_analytical - J_numerical).norm() < 5e-4);
}

TEST_CASE("Test Camera-Lidar Factor Error and Jacobian") {
  // Create intrinsics
  std::string camera_model_location = __FILE__;
  camera_model_location.erase(camera_model_location.end() - 22,
                              camera_model_location.end());
  camera_model_location += "data/CamFactorIntrinsics.json";
  std::shared_ptr<beam_calibration::CameraModel> camera_model =
      beam_calibration::CameraModel::Create(camera_model_location);

  // Create Transforms
  Eigen::Matrix4d T_VC = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_VL = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_CT = Eigen::Matrix4d::Identity();
  T_VC.block(0, 3, 3, 1) = Eigen::Vector3d(1, 1, 0);
  T_CT.block(0, 3, 3, 1) = Eigen::Vector3d(0.2, 0.2, 3);
  T_VL.block(0, 3, 3, 1) =
      T_VC.block(0, 3, 3, 1) + Eigen::Vector3d(0.2, 0.2, 0);
  Eigen::Matrix4d T_CV = T_VC.inverse();
  Eigen::Matrix4d T_LV = T_VL.inverse();
  Eigen::Matrix4d T_VT = T_VC * T_CT;

  // Create Points
  Eigen::Vector4d P_T(0.2, 0.2, 1, 1);
  Eigen::Vector4d P_C = T_CT * P_T;
  Eigen::Vector4d P_L = T_LV * T_VT * P_T;
  Eigen::Vector3d point_measured = P_L.hnormalized();
  opt<Eigen::Vector2d> pixel_measured =
      camera_model->ProjectPointPrecise(P_C.hnormalized());

  // create factor
  gtsam::Key dummy_keyL(gtsam::symbol('L', 0));
  gtsam::Key dummy_keyC(gtsam::symbol('C', 0));
  gtsam::Vector2 noise_vec(10, 10);
  gtsam::noiseModel::Diagonal::shared_ptr noise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  gtsam::Pose3 camera_pose(T_VC);
  gtsam::Pose3 lidar_pose(T_VL);
  vicon_calibration::CameraLidarFactor factor(
      dummy_keyL, dummy_keyC, pixel_measured.value(), point_measured,
      P_T.hnormalized(), P_T.hnormalized(), camera_model, noise);

  // calculate numerical Jacobian
  double eps = std::sqrt(1e-12);
  Eigen::MatrixXd p(6, 6);
  p.setIdentity();
  p = p * eps;
  Eigen::MatrixXd JC_numerical(2, 6);
  Eigen::MatrixXd JL_numerical(2, 6);

  for (int i = 0; i < 6; i++) {
    Eigen::Matrix4d T_VC_perturb =
        vicon_calibration::utils::PerturbTransformRadM(T_VC,
                                                       p.block(0, i, 6, 1));
    Eigen::Matrix4d T_VL_perturb =
        vicon_calibration::utils::PerturbTransformRadM(T_VL,
                                                       p.block(0, i, 6, 1));
    gtsam::Pose3 camera_pose_perturbed(T_VC_perturb);
    gtsam::Pose3 lidar_pose_perturbed(T_VL_perturb);
    JC_numerical.block(0, i, 2, 1) =
        (factor.evaluateError(lidar_pose, camera_pose_perturbed) -
         factor.evaluateError(lidar_pose, camera_pose)) /
        eps;
    JL_numerical.block(0, i, 2, 1) =
        (factor.evaluateError(lidar_pose_perturbed, camera_pose) -
         factor.evaluateError(lidar_pose, camera_pose)) /
        eps;
  }

  // calculate analytical jacobian
  gtsam::Matrix JL_analytical;
  gtsam::Matrix JC_analytical;
  factor.evaluateError(lidar_pose, camera_pose, JL_analytical, JC_analytical);

  REQUIRE((JC_analytical - JC_numerical).norm() < 5e-4);
  REQUIRE((JL_analytical - JL_numerical).norm() < 5e-4);
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
  Eigen::Matrix4d T_VL = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_LT = Eigen::Matrix4d::Identity();
  T_VL.block(0, 3, 3, 1) = Eigen::Vector3d(1.1, 1.2, 1.3);
  T_LT.block(0, 3, 3, 1) = Eigen::Vector3d(2, 3, 4);
  Eigen::Matrix4d T_LV = T_VL.inverse();
  Eigen::Matrix4d T_VT = T_VL * T_LT;

  // create perturbed initial
  Eigen::Matrix4d T_VL_pert;
  Eigen::VectorXd perturbation(6, 1);
  perturbation << 0.3, -0.3, 0.3, 0.5, -0.5, 0.3;
  T_VL_pert =
      vicon_calibration::utils::PerturbTransformRadM(T_VL, perturbation);

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
    pti = points[i].hnormalized();
    ptmi = points_measured[i].hnormalized();
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

  // solve with perturbed initials
  gtsam::Values initials_pert, results2;
  gtsam::Pose3 initial_pose2(T_VL_pert);
  initials_pert.insert(key, initial_pose2);
  gtsam::LevenbergMarquardtOptimizer optimizer2(graph, initials_pert, params);
  results2 = optimizer2.optimize();
  Eigen::Matrix4d T_VL_opt2 =
      results2.at<gtsam::Pose3>(gtsam::Symbol('L', 1)).matrix();

  REQUIRE(T_VL == vicon_calibration::utils::RoundMatrix(T_VL_opt1, 5));
  REQUIRE(T_VL == vicon_calibration::utils::RoundMatrix(T_VL_opt2, 5));
}

TEST_CASE("Test camera factor in Optimization") {
  // create keypoints
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      points;
  double max_distance_x = 2, max_distance_y = 2, max_distance_z = 4;
  for (int i = 0; i < 80; i++) {
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
      beam_calibration::CameraModel::Create(camera_model_location);

  // Create Transforms
  Eigen::Matrix4d T_VC = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_CT = Eigen::Matrix4d::Identity();
  T_VC.block(0, 3, 3, 1) = Eigen::Vector3d(1, 1, 0);
  T_CT.block(0, 3, 3, 1) = Eigen::Vector3d(0.2, 0.2, 3);
  Eigen::Matrix4d T_CV = T_VC.inverse();
  Eigen::Matrix4d T_VT = T_VC * T_CT;

  // create perturbed initial
  Eigen::Matrix4d T_CV_pert;
  Eigen::VectorXd perturbation(6, 1);
  perturbation << 0.3, -0.3, 0.3, 0.5, -0.5, 0.3;
  T_CV_pert =
      vicon_calibration::utils::PerturbTransformDegM(T_CV, perturbation);

  // create projected (detected) points - no noise
  std::vector<Eigen::Vector2d, AlignVec2d> pixels(points.size());
  std::vector<bool> pixels_valid(points.size());
  for (int i = 0; i < points.size(); i++) {
    Eigen::Vector4d point_transformed = T_CV * T_VT * points[i];
    opt<Eigen::Vector2d> pixel =
        camera_model->ProjectPointPrecise(point_transformed.hnormalized());
    if (pixel.has_value()) {
      pixels_valid[i] = true;
      pixels[i] = pixel.value();
    } else {
      pixels_valid[i] = false;
    }
  }

  // build graph
  gtsam::NonlinearFactorGraph graph;

  // add factors
  gtsam::Key key = gtsam::Symbol('C', 1);
  gtsam::Vector2 noise_vec{Eigen::Vector2d{1, 1}};
  gtsam::noiseModel::Diagonal::shared_ptr ImageNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  for (int i = 0; i < points.size(); i++) {
    if (pixels_valid[i]) {
      graph.emplace_shared<vicon_calibration::CameraFactor>(
          key, pixels[i], points[i].hnormalized(), camera_model, T_VT,
          ImageNoise);
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
  gtsam::Pose3 initial_pose1(T_CV);
  initials_exact.insert(key, initial_pose1);
  gtsam::LevenbergMarquardtOptimizer optimizer1(graph, initials_exact, params);
  results1 = optimizer1.optimize();
  Eigen::Matrix4d T_CV_opt1 =
      results1.at<gtsam::Pose3>(gtsam::Symbol('C', 1)).matrix();

  // solve with perturbed initials
  gtsam::Values initials_pert, results2;
  gtsam::Pose3 initial_pose2(T_CV_pert);
  initials_pert.insert(key, initial_pose2);
  gtsam::LevenbergMarquardtOptimizer optimizer2(graph, initials_pert, params);
  results2 = optimizer2.optimize();
  Eigen::Matrix4d T_CV_opt2 =
      results2.at<gtsam::Pose3>(gtsam::Symbol('C', 1)).matrix();

  REQUIRE(T_CV == vicon_calibration::utils::RoundMatrix(T_CV_opt1, 5));
  REQUIRE(T_CV == vicon_calibration::utils::RoundMatrix(T_CV_opt2, 5));
}

TEST_CASE("Test camera factor inv in Optimization") {
  // create keypoints
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      points;
  double max_distance_x = 2, max_distance_y = 2, max_distance_z = 4;
  for (int i = 0; i < 80; i++) {
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
      beam_calibration::CameraModel::Create(camera_model_location);

  // Create Transforms
  Eigen::Matrix4d T_VC = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_CT = Eigen::Matrix4d::Identity();
  T_VC.block(0, 3, 3, 1) = Eigen::Vector3d(1, 1, 0);
  T_CT.block(0, 3, 3, 1) = Eigen::Vector3d(0.2, 0.2, 3);
  Eigen::Matrix4d T_CV = T_VC.inverse();
  Eigen::Matrix4d T_VT = T_VC * T_CT;

  // create perturbed initial
  Eigen::Matrix4d T_VC_pert;
  Eigen::VectorXd perturbation(6, 1);
  perturbation << 0.3, -0.3, 0.3, 0.5, -0.5, 0.3;
  T_VC_pert =
      vicon_calibration::utils::PerturbTransformDegM(T_VC, perturbation);

  // create projected (detected) points - no noise
  std::vector<Eigen::Vector2d, AlignVec2d> pixels(points.size());
  std::vector<bool> pixels_valid(points.size());
  for (int i = 0; i < points.size(); i++) {
    Eigen::Vector4d point_transformed = T_CV * T_VT * points[i];
    opt<Eigen::Vector2d> pixel =
        camera_model->ProjectPointPrecise(point_transformed.hnormalized());
    if (pixel.has_value()) {
      pixels_valid[i] = true;
      pixels[i] = pixel.value();
    } else {
      pixels_valid[i] = false;
    }
  }

  // build graph
  gtsam::NonlinearFactorGraph graph;

  // add factors
  gtsam::Key key = gtsam::Symbol('C', 1);
  gtsam::Vector2 noise_vec{Eigen::Vector2d{1, 1}};
  gtsam::noiseModel::Diagonal::shared_ptr ImageNoise =
      gtsam::noiseModel::Diagonal::Sigmas(noise_vec);
  for (int i = 0; i < points.size(); i++) {
    if (pixels_valid[i]) {
      graph.emplace_shared<vicon_calibration::CameraFactorInv>(
          key, pixels[i], points[i].hnormalized(), camera_model, T_VT,
          ImageNoise);
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

  // solve with perturbed initials
  gtsam::Values initials_pert, results2;
  gtsam::Pose3 initial_pose2(T_VC_pert);
  initials_pert.insert(key, initial_pose2);
  gtsam::LevenbergMarquardtOptimizer optimizer2(graph, initials_pert, params);
  results2 = optimizer2.optimize();
  Eigen::Matrix4d T_VC_opt2 =
      results2.at<gtsam::Pose3>(gtsam::Symbol('C', 1)).matrix();

  REQUIRE(T_VC == vicon_calibration::utils::RoundMatrix(T_VC_opt1, 5));
  REQUIRE(T_VC == vicon_calibration::utils::RoundMatrix(T_VC_opt2, 5));
}

/*
TEST_CASE("Test Camera-Lidar factor in Optimization") {
  // create keypoints
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      points;
  double max_distance_x = 3, max_distance_y = 3, max_distance_z = 4;
  for (int i = 0; i < 80; i++) {
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
      beam_calibration::CameraModel::Create(camera_model_location);

  // Create Transforms
  Eigen::Matrix4d T_VC = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_VL = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_CT = Eigen::Matrix4d::Identity();
  T_VC.block(0, 3, 3, 1) = Eigen::Vector3d(1, 1, 0);
  T_CT.block(0, 3, 3, 1) = Eigen::Vector3d(0.2, 0.2, 3);
  T_VL.block(0, 3, 3, 1) =
      T_VC.block(0, 3, 3, 1) + Eigen::Vector3d(0.2, 0.2, 0);
  Eigen::Matrix4d T_CV = T_VC.inverse();
  Eigen::Matrix4d T_LV = T_VL.inverse();
  Eigen::Matrix4d T_VT = T_VC * T_CT;

  // create perturbed initial
  Eigen::Matrix4d T_VC_pert, T_VL_pert;
  Eigen::VectorXd perturb_cam(6, 1), perturb_lid(6, 1);
  perturb_cam << 0.3, -0.3, 0.3, 0.5, -0.5, 0.3;
  perturb_lid << -0.3, 0.3, -0.3, -0.5, 0.5, -0.3;
  T_VC_pert = vicon_calibration::utils::PerturbTransformRadM(T_VC, perturb_cam);
  T_VL_pert = vicon_calibration::utils::PerturbTransformRadM(T_VL, perturb_lid);

  // create measured pixels and measured lidar points - no noise
  std::vector<Eigen::Vector2d, AlignVec2d> pixels_measured(points.size());
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      points_measured(points.size());
  std::vector<bool> pixels_valid(points.size());

  for (int i = 0; i < points.size(); i++) {
    Eigen::Vector4d point_transformed = T_CT * points[i];
    opt<Eigen::Vector2d> pixel =
        camera_model->ProjectPointPrecise(point_transformed.hnormalized());
    if (pixel.has_value()) {
      pixels_valid[i] = true;
      pixels_measured[i] = pixel.value();
      points_measured[i] = T_LV * T_VT * points[i];
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
      pti = points[i].hnormalized();
      ptmi = points_measured[i].hnormalized();
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
      pti = points[i].hnormalized();
      ptmi = points_measured[i].hnormalized();
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

  // solve with perturbed initials
  gtsam::Values initials_pert, results2;
  gtsam::Pose3 initial_pose_cam2(T_VC_pert), initial_pose_lid2(T_VL_pert);
  initials_pert.insert(camera_key, initial_pose_cam2);
  initials_pert.insert(lidar_key, initial_pose_lid2);
  gtsam::LevenbergMarquardtOptimizer optimizer2(graph, initials_pert, params);
  results2 = optimizer2.optimize();
  Eigen::Matrix4d T_VC_opt2 = results1.at<gtsam::Pose3>(camera_key).matrix();
  Eigen::Matrix4d T_VL_opt2 = results1.at<gtsam::Pose3>(lidar_key).matrix();

  REQUIRE(T_VC == vicon_calibration::utils::RoundMatrix(T_VC_opt1, 5));
  REQUIRE(T_VC == vicon_calibration::utils::RoundMatrix(T_VC_opt2, 5));
  REQUIRE(T_VL == vicon_calibration::utils::RoundMatrix(T_VL_opt1, 5));
  REQUIRE(T_VL == vicon_calibration::utils::RoundMatrix(T_VL_opt2, 5));
}
*/