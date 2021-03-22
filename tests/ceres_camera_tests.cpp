#define CATCH_CONFIG_MAIN

#include <Eigen/Geometry>
#include <catch2/catch.hpp>
#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/solver.h>
#include <ceres/types.h>

#include <vicon_calibration/optimization/CeresCameraCostFunction.h>
#include <vicon_calibration/Utils.h>
#include <vicon_calibration/camera_models/CameraModels.h>

using namespace vicon_calibration;

ceres::Solver::Options ceres_solver_options_;
std::unique_ptr<ceres::LossFunction> loss_function_;
std::unique_ptr<ceres::LocalParameterization> se3_parameterization_;
bool output_results_{true};

std::string GetFileLocationData(const std::string& name) {
  std::string full_path = __FILE__;
  std::string this_filename = "ceres_camera_tests.cpp";
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

TEST_CASE("Test camera optimization") {
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

  // // Create intrinsics
  std::string camera_model_location =
      GetFileLocationData("CamFactorIntrinsics.json");
  std::shared_ptr<vicon_calibration::CameraModel> camera_model =
      vicon_calibration::CameraModel::Create(camera_model_location);

  // Create Transforms
  Eigen::Matrix4d T_VC =
      utils::BuildTransformEulerDegM(90, 10, -5, 0.1, -0.4, 0.2);
  Eigen::Matrix4d T_CT = utils::BuildTransformEulerDegM(10, -20, 5, 1, 1.5, 0);
  Eigen::Matrix4d T_CV = utils::InvertTransform(T_VC);
  Eigen::Matrix4d T_VT = T_VC * T_CT;

  // create perturbed initial
  Eigen::VectorXd perturbation(6, 1);
  perturbation << 0.3, -0.3, 0.3, 0.5, -0.5, 0.3;
  Eigen::Matrix4d T_CV_pert = utils::PerturbTransformDegM(T_CV, perturbation);

  // create projected (detected) points - no noise
  std::vector<Eigen::Vector2d, AlignVec2d> pixels(points.size());
  std::vector<bool> pixels_valid(points.size());
  for (int i = 0; i < points.size(); i++) {
    Eigen::Vector4d point_transformed = T_CV * T_VT * points[i];
    vicon_calibration::opt<Eigen::Vector2d> pixel =
        camera_model->ProjectPointPrecise(point_transformed.hnormalized());
    if (pixel.has_value()) {
      pixels_valid[i] = true;
      pixels[i] = pixel.value();
    } else {
      pixels_valid[i] = false;
    }
  }

  // create values to optimize
  // ----------------------------------------------
  // THIS OPTION CAUSES A SEG FAULT FOR SOME REASON
  // std::vector<double> results_perfect_init =
  //     utils::TransformMatrixToQuaternionAndTranslation(T_CV);
  // std::vector<double> results_perturbed_init =
  //     utils::TransformMatrixToQuaternionAndTranslation(
  //         T_CV_pert);
  // ----------------------------------------------
  Eigen::Matrix3d R1 = T_CV.block(0, 0, 3, 3);
  Eigen::Quaternion<double> q1 = Eigen::Quaternion<double>(R1);
  std::vector<double> results_perfect_init{
      q1.w(), q1.x(), q1.y(), q1.z(), T_CV(0, 3), T_CV(1, 3), T_CV(2, 3)};
  Eigen::Matrix3d R2 = T_CV_pert.block(0, 0, 3, 3);
  Eigen::Quaternion<double> q2 = Eigen::Quaternion<double>(R2);
  std::vector<double> results_perturbed_init{
      q2.w(),          q2.x(),          q2.y(),         q2.z(),
      T_CV_pert(0, 3), T_CV_pert(1, 3), T_CV_pert(2, 3)};
  // ----------------------------------------------

  // build problems
  std::shared_ptr<ceres::Problem> problem1 = SetupCeresProblem();
  std::shared_ptr<ceres::Problem> problem2 = SetupCeresProblem();

  problem1->AddParameterBlock(&(results_perfect_init[0]), 7,
                              se3_parameterization_.get());
  problem2->AddParameterBlock(&(results_perturbed_init[0]), 7,
                              se3_parameterization_.get());

  for (int i = 0; i < points.size(); i++) {
    if (pixels_valid[i]) {
      Eigen::Vector3d P_VICONBASE = (T_VT * points[i]).hnormalized();

      // add residuals for perfect init
      std::unique_ptr<ceres::CostFunction> cost_function1(
          CeresCameraCostFunction::Create(pixels[i], P_VICONBASE,
                                          camera_model));

      problem1->AddResidualBlock(cost_function1.release(), loss_function_.get(),
                                 &(results_perfect_init[0]));

      // add residuals for perturbed init
      std::unique_ptr<ceres::CostFunction> cost_function2(
          CeresCameraCostFunction::Create(pixels[i], P_VICONBASE,
                                          camera_model));
      problem2->AddResidualBlock(cost_function2.release(), loss_function_.get(),
                                 &(results_perturbed_init[0]));

      // Check that the inputs are correct:
      double P_C[3];
      ceres::QuaternionRotatePoint(&(results_perfect_init[0]),
                                   P_VICONBASE.data(), P_C);
      Eigen::Vector3d point_transformed(P_C[0] + results_perfect_init[4],
                                        P_C[1] + results_perfect_init[5],
                                        P_C[2] + results_perfect_init[6]);
      vicon_calibration::opt<Eigen::Vector2d> pixels_projected =
          camera_model->ProjectPointPrecise(point_transformed);
      REQUIRE(pixels_projected.value().isApprox(pixels[i], 1e-5));
    }
  }

  LOG_INFO("TESTING WITH PERFECT INITIALIZATION");
  SolveProblem(problem1, output_results_);
  Eigen::Matrix4d T_CV_opt1 =
      utils::QuaternionAndTranslationToTransformMatrix(results_perfect_init);

  LOG_INFO("TESTING WITH PERTURBED INITIALIZATION");
  SolveProblem(problem2, output_results_);
  Eigen::Matrix4d T_CV_opt2 =
      utils::QuaternionAndTranslationToTransformMatrix(results_perturbed_init);

  REQUIRE(utils::RoundMatrix(T_CV, 5) == utils::RoundMatrix(T_CV_opt1, 5));
  REQUIRE(utils::RoundMatrix(T_CV, 5) == utils::RoundMatrix(T_CV_opt2, 5));
}
