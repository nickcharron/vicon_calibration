#define CATCH_CONFIG_MAIN

#include <Eigen/Geometry>
#include <catch2/catch.hpp>
#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/solver.h>
#include <ceres/types.h>

#include <vicon_calibration/Utils.h>
#include <vicon_calibration/optimization/CeresLidarCostFunction.h>

using namespace vicon_calibration;

ceres::Solver::Options ceres_solver_options_;
std::unique_ptr<ceres::LossFunction> loss_function_;
std::unique_ptr<ceres::LocalParameterization> se3_parameterization_;
bool output_results_{true};

std::string GetFileLocationData(const std::string& name) {
  std::string full_path = __FILE__;
  std::string this_filename = "ceres_lidar_tests.cpp";
  full_path.erase(full_path.end() - this_filename.length(), full_path.end());
  full_path += "data/";
  full_path += name;
  return full_path;
}

std::shared_ptr<ceres::Problem> SetupCeresProblem() {
  // set ceres solver params
  ceres_solver_options_.minimizer_progress_to_stdout = false;
  ceres_solver_options_.max_num_iterations = 50;
  ceres_solver_options_.max_solver_time_in_seconds = 1e6;
  ceres_solver_options_.function_tolerance = 1e-8;
  ceres_solver_options_.gradient_tolerance = 1e-10;
  ceres_solver_options_.parameter_tolerance = 1e-8;
  ceres_solver_options_.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_solver_options_.preconditioner_type = ceres::SCHUR_JACOBI;

  // set ceres problem options
  ceres::Problem::Options ceres_problem_options;

  // if we want to manage our own data for these, we can set these flags:
  ceres_problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres_problem_options.local_parameterization_ownership =
      ceres::DO_NOT_TAKE_OWNERSHIP;

  std::shared_ptr<ceres::Problem> problem =
      std::make_shared<ceres::Problem>(ceres_problem_options);

  loss_function_ =
      std::unique_ptr<ceres::LossFunction>(new ceres::HuberLoss(1.0));

  std::unique_ptr<ceres::LocalParameterization> quat_parameterization(
      new ceres::QuaternionParameterization());
  std::unique_ptr<ceres::LocalParameterization> identity_parameterization(
      new ceres::IdentityParameterization(3));
  se3_parameterization_ = std::unique_ptr<ceres::LocalParameterization>(
      new ceres::ProductParameterization(quat_parameterization.release(),
                                         identity_parameterization.release()));

  return problem;
}

void SolveProblem(const std::shared_ptr<ceres::Problem>& problem,
                  bool output_results) {
  ceres::Solver::Summary ceres_summary;
  ceres::Solve(ceres_solver_options_, problem.get(), &ceres_summary);
  if (output_results) {
    LOG_INFO("Done.");
    LOG_INFO("Outputting ceres summary:");
    std::string report = ceres_summary.FullReport();
    std::cout << report << "\n";
  }
}

TEST_CASE("Test lidar optimization") {
  // create keypoints
  std::vector<Eigen::Vector4d> points;
  double x, y, z, max_distance = 3;
  for (int i = 0; i < 30; i++) {
    x = ((double)std::rand() / (RAND_MAX)-0.5) * 2 * max_distance;
    y = ((double)std::rand() / (RAND_MAX)-0.5) * 2 * max_distance;
    z = ((double)std::rand() / (RAND_MAX)-0.5) * 2 * max_distance;
    Eigen::Vector4d point(x, y, z, 1);
    points.push_back(point);
  }

  // Create Transforms
  Eigen::Matrix4d T_VL =
      utils::BuildTransformEulerDegM(90, 10, -5, 0.1, -0.4, 0.2);
  Eigen::Matrix4d T_LT = utils::BuildTransformEulerDegM(10, -20, 5, 1, 1.5, 0);
  Eigen::Matrix4d T_LV = utils::InvertTransform(T_VL);
  Eigen::Matrix4d T_VT = T_VL * T_LT;

  // create perturbed initial
  Eigen::Matrix4d T_LV_pert;
  Eigen::VectorXd perturbation(6, 1);
  perturbation << 0.3, -0.3, 0.3, 0.5, -0.5, 0.3;
  T_LV_pert = utils::PerturbTransformRadM(T_LV, perturbation);

  // create transformed (detected) points - no noise
  std::vector<Eigen::Vector4d> points_measured(points.size());
  Eigen::Vector4d point_transformed;
  for (int i = 0; i < points.size(); i++) {
    points_measured[i] = T_LV * T_VT * points[i];
  }

  // create values to optimize

  // ------------------------------------
  // THIS METHOD CAUSES SEG FAULT AT LINE 163
  // std::vector<double> results_perfect_init =
  //     utils::TransformMatrixToQuaternionAndTranslation(T_LV);
  // std::vector<double> results_perturbed_init =
  //     utils::TransformMatrixToQuaternionAndTranslation(
  //         T_LV_pert);
  // ------------------------------------
  // THIS METHOD DOES NOT
  Eigen::Matrix3d R1 = T_LV.block(0, 0, 3, 3);
  Eigen::Quaternion<double> q1 = Eigen::Quaternion<double>(R1);
  std::vector<double> results_perfect_init{
      q1.w(), q1.x(), q1.y(), q1.z(), T_LV(0, 3), T_LV(1, 3), T_LV(2, 3)};
  Eigen::Matrix3d R2 = T_LV_pert.block(0, 0, 3, 3);
  Eigen::Quaternion<double> q2 = Eigen::Quaternion<double>(R2);
  std::vector<double> results_perturbed_init{
      q2.w(),          q2.x(),          q2.y(),         q2.z(),
      T_LV_pert(0, 3), T_LV_pert(1, 3), T_LV_pert(2, 3)};
  // ------------------------------------

  // build problems
  std::shared_ptr<ceres::Problem> problem1 = SetupCeresProblem();
  std::shared_ptr<ceres::Problem> problem2 = SetupCeresProblem();

  problem1->AddParameterBlock(&(results_perfect_init[0]), 7,
                              se3_parameterization_.get());
  problem2->AddParameterBlock(&(results_perturbed_init[0]), 7,
                              se3_parameterization_.get());

  for (int i = 0; i < points.size(); i++) {
    Eigen::Vector3d point_measured = points_measured[i].hnormalized();

    // add residuals for perfect init
    std::unique_ptr<ceres::CostFunction> cost_function1(
        CeresLidarCostFunction::Create(point_measured, points[i].hnormalized(),
                                       T_VT));
    problem1->AddResidualBlock(cost_function1.release(), loss_function_.get(),
                               &(results_perfect_init[0]));

    // add residuals for perturbed init
    std::unique_ptr<ceres::CostFunction> cost_function2(
        CeresLidarCostFunction::Create(point_measured, points[i].hnormalized(),
                                       T_VT));

    problem2->AddResidualBlock(cost_function2.release(), loss_function_.get(),
                               &results_perturbed_init[0]);
  }

  LOG_INFO("TESTING WITH PERFECT INITIALIZATION");
  SolveProblem(problem1, output_results_);
  Eigen::Matrix4d T_LV_opt1 =
      utils::QuaternionAndTranslationToTransformMatrix(results_perfect_init);

  LOG_INFO("TESTING WITH PERTURBED INITIALIZATION");
  SolveProblem(problem2, output_results_);
  Eigen::Matrix4d T_LV_opt2 =
      utils::QuaternionAndTranslationToTransformMatrix(results_perturbed_init);

  REQUIRE(utils::RoundMatrix(T_LV, 5) == utils::RoundMatrix(T_LV_opt1, 5));
  REQUIRE(utils::RoundMatrix(T_LV, 5) == utils::RoundMatrix(T_LV_opt2, 5));
}

/*
TEST_CASE("Test Camera-Lidar factor in Optimization") {
  // create keypoints
  std::vector<Eigen::Vector4d> points;
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
  std::shared_ptr<vicon_calibration::CameraModel> camera_model =
      vicon_calibration::CameraModel::Create(camera_model_location);

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
  T_VC_pert = utils::PerturbTransformRadM(T_VC, perturb_cam);
  T_VL_pert = utils::PerturbTransformRadM(T_VL, perturb_lid);

  // create measured pixels and measured lidar points - no noise
  std::vector<Eigen::Vector2d> pixels_measured(points.size());
  std::vector<Eigen::Vector4d>
      points_measured(points.size());
  std::vector<bool> pixels_valid(points.size());

  for (int i = 0; i < points.size(); i++) {
    Eigen::Vector4d point_transformed = T_CT * points[i];
    bool pixel_valid;
    Eigen::Vector2d pixel;
    camera_model->ProjectPoint(point_transformed.hnormalized(), pixel,
pixel_valid); pixels_valid[i] = pixel_valid; if (pixel_valid) {
      pixels_measured[i] = pixel;
      points_measured[i] = T_LV * T_VT * points[i];
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
      graph.emplace_shared<CameraLidarFactor>(
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
      graph.emplace_shared<LidarFactor>(lidar_key, ptmi, pti,
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

  REQUIRE(T_VC == utils::RoundMatrix(T_VC_opt1, 5));
  REQUIRE(T_VC == utils::RoundMatrix(T_VC_opt2, 5));
  REQUIRE(T_VL == utils::RoundMatrix(T_VL_opt1, 5));
  REQUIRE(T_VL == utils::RoundMatrix(T_VL_opt2, 5));
}
*/