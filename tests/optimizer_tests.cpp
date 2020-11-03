
#define CATCH_CONFIG_MAIN

#include <Eigen/Geometry>
#include <catch2/catch.hpp>
#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/solver.h>
#include <ceres/types.h>

#include "vicon_calibration/JsonTools.h"
#include "vicon_calibration/optimization/CeresCameraCostFunction.h"
#include "vicon_calibration/optimization/CeresLidarCostFunction.h"
#include "vicon_calibration/optimization/CeresOptimizer.h"
#include "vicon_calibration/utils.h"
#include <beam_calibration/CameraModel.h>

using namespace vicon_calibration;

// Global Transforms
Eigen::Matrix4d T_VC;
Eigen::Matrix4d T_CT;
Eigen::Matrix4d T_CV;
Eigen::Matrix4d T_VL;
Eigen::Matrix4d T_LT;
Eigen::Matrix4d T_LV;
Eigen::Matrix4d T_VT;
std::vector<Eigen::Matrix4d, AlignMat4d> T_VTs;
Eigen::Matrix4d T_CV_pert;
Eigen::Matrix4d T_VC_pert;
Eigen::Matrix4d T_LV_pert;
Eigen::Matrix4d T_VL_pert;

// Ceres global variables
ceres::Solver::Options ceres_solver_options_;
std::unique_ptr<ceres::LossFunction> loss_function_;
std::unique_ptr<ceres::LocalParameterization> se3_parameterization_;

std::string GetFileLocationData(const std::string& name) {
  std::string full_path = __FILE__;
  std::string this_filename = "optimizer_tests.cpp";
  full_path.erase(full_path.end() - this_filename.length(), full_path.end());
  full_path += "data/";
  full_path += name;
  return full_path;
}

void CreateTransforms() {
  // Create Camera Transforms
  T_VC = utils::BuildTransformEulerDegM(90, 10, -5, 0.1, -0.4, 0.2);
  T_CT = utils::BuildTransformEulerDegM(4, -2, 5, 0, 0, 1.5);
  T_CV = utils::InvertTransform(T_VC);

  // Create target transforms
  T_VT = T_VC * T_CT;
  T_VTs = std::vector<Eigen::Matrix4d, AlignMat4d>{T_VT};

  // create Lidar Transforms
  T_VL = utils::BuildTransformEulerDegM(20, 90, 5, 0.3, 0.1, -0.3);
  T_LV = utils::InvertTransform(T_VL);
  T_LT = T_LV * T_VT;

  // create perturbed initials
  Eigen::VectorXd perturbation_camera(6, 1);
  perturbation_camera << 0.3, -0.3, 0.3, 0.5, -0.5, 0.3;
  T_CV_pert = utils::PerturbTransformDegM(T_CV, perturbation_camera);
  T_VC_pert = utils::InvertTransform(T_CV_pert);
  Eigen::VectorXd perturbation_lidar(6, 1);
  perturbation_lidar << 0.2, -0.4, 0.1, 0.4, -0.2, -0.3;
  T_LV_pert = utils::PerturbTransformDegM(T_LV, perturbation_lidar);
  T_VL_pert = utils::InvertTransform(T_LV_pert);
}

TargetParamsVector GetTargetParams(const std::string& target_filename) {
  std::string config_file_location = GetFileLocationData(target_filename);
  JsonTools jtools;
  std::shared_ptr<TargetParams> param_tmp = std::make_shared<TargetParams>();
  param_tmp = jtools.LoadTargetParams(config_file_location);
  //   param_tmp->Print();
  TargetParamsVector params{param_tmp};
  return params;
}

CameraParamsVector GetCameraParams(const std::string& camera_model_filename) {
  std::string config_file_location = GetFileLocationData(camera_model_filename);
  std::shared_ptr<CameraParams> param_tmp =
      std::make_shared<CameraParams>(config_file_location);
  //   param_tmp->Print();
  CameraParamsVector params{param_tmp};
  return params;
}

CameraMeasurements CreateCameraMeasurements(
    const Eigen::Matrix4d& T_VC,
    const std::vector<Eigen::Matrix4d, AlignMat4d>& T_VTs,
    const TargetParams& target_params, const CameraParams& camera_params) {
  Eigen::Matrix4d T_CV = utils::InvertTransform(T_VC);
  std::vector<std::shared_ptr<CameraMeasurement>> measurements;
  for (int i = 0; i < T_VTs.size(); i++) {
    Eigen::Matrix4d _T_VT = T_VTs[i];
    CameraMeasurement measurement;
    measurement.T_VICONBASE_TARGET = _T_VT;
    measurement.camera_id = 0;
    measurement.target_id = 0;
    measurement.camera_frame = "CAMERA";
    measurement.target_frame = "TARGET";
    measurement.time_stamp = ros::Time(0, 0);
    pcl::PointCloud<pcl::PointXY>::Ptr pixels =
        boost::make_shared<pcl::PointCloud<pcl::PointXY>>();
    std::vector<Eigen::Vector3d, AlignVec3d> keypoint_in_tgt_frame =
        target_params.keypoints_camera;
    Eigen::Matrix4d T_CT = T_CV * _T_VT;
    for (Eigen::Vector3d P_TARGET : keypoint_in_tgt_frame) {
      Eigen::Vector3d P_CAMERA = (T_CT * P_TARGET.homogeneous()).hnormalized();
      opt<Eigen::Vector2d> point_projected =
          camera_params.camera_model->ProjectPointPrecise(P_CAMERA);
      if (point_projected.has_value()) {
        pixels->push_back(
            pcl::PointXY{.x = point_projected.value().cast<float>()[0],
                         .y = point_projected.value().cast<float>()[1]});
      }
    }
    measurement.keypoints = pixels;
    // measurement.Print();
    std::shared_ptr<CameraMeasurement> measurement_ptr =
        std::make_shared<CameraMeasurement>(measurement);
    measurements.push_back(measurement_ptr);
  }
  return CameraMeasurements{measurements};
}

LidarMeasurements CreateLidarMeasurements(
    const Eigen::Matrix4d& T_VL,
    const std::vector<Eigen::Matrix4d, AlignMat4d>& T_VTs,
    const TargetParams& target_params) {
  Eigen::Matrix4d T_LV = utils::InvertTransform(T_VL);
  std::vector<std::shared_ptr<LidarMeasurement>> measurements;
  for (int i = 0; i < T_VTs.size(); i++) {
    Eigen::Matrix4d _T_VT = T_VTs[i];
    LidarMeasurement measurement;
    measurement.T_VICONBASE_TARGET = _T_VT;
    measurement.lidar_id = 0;
    measurement.target_id = 0;
    measurement.lidar_frame = "LIDAR";
    measurement.target_frame = "TARGET";
    measurement.time_stamp = ros::Time(0, 0);

    pcl::PointCloud<pcl::PointXYZ>::Ptr measured_points =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    std::vector<Eigen::Vector3d, AlignVec3d> keypoint_in_tgt_frame =
        target_params.keypoints_lidar;
    Eigen::Matrix4d T_LT = T_LV * _T_VT;
    for (Eigen::Vector3d P_TARGET : keypoint_in_tgt_frame) {
      Eigen::Vector3f P_LIDAR =
          (T_LT * P_TARGET.homogeneous()).hnormalized().cast<float>();
      measured_points->push_back(
          pcl::PointXYZ{.x = P_LIDAR[0], .y = P_LIDAR[1], .z = P_LIDAR[2]});
    }
    measurement.keypoints = measured_points;
    // measurement.Print();
    std::shared_ptr<LidarMeasurement> measurement_ptr =
        std::make_shared<LidarMeasurement>(measurement);
    measurements.push_back(measurement_ptr);
  }
  return LidarMeasurements{measurements};
}

CalibrationResults CreateInitialCalibrations(const Eigen::Matrix4d& T_VC,
                                             const Eigen::Matrix4d& T_VL) {
  CalibrationResult camera_calib;
  camera_calib.transform = T_VC;
  camera_calib.type = SensorType::CAMERA;
  camera_calib.sensor_id = 0;
  camera_calib.to_frame = "VICONBASE";
  camera_calib.from_frame = "CAMERA";

  CalibrationResult lidar_calib;
  lidar_calib.transform = T_VL;
  lidar_calib.type = SensorType::LIDAR;
  lidar_calib.sensor_id = 0;
  lidar_calib.to_frame = "VICONBASE";
  lidar_calib.from_frame = "LIDAR";

  return CalibrationResults{camera_calib, lidar_calib};
}

std::shared_ptr<ceres::Problem> SetupCeresProblem() {
  // set ceres solver params
  ceres_solver_options_.minimizer_progress_to_stdout = false;
  ceres_solver_options_.max_num_iterations = 50;
  ceres_solver_options_.max_solver_time_in_seconds = 1e6;
  ceres_solver_options_.function_tolerance = 1e-6;
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

TEST_CASE("Test Ceres Optimizer With Perfect Initials") {
  CreateTransforms();

  // create measurements
  TargetParamsVector target_params =
  GetTargetParams("DiamondTargetSim.json"); CameraParamsVector camera_params
  =
      GetCameraParams("CamFactorIntrinsics.json");
  LidarMeasurements lidar_measurements =
      CreateLidarMeasurements(T_VL, T_VTs, *(target_params[0]));
  CameraMeasurements camera_measurements = CreateCameraMeasurements(
      T_VC, T_VTs, *(target_params[0]), *(camera_params[0]));
  LoopClosureMeasurements loop_closure_measurements;

  CalibrationResults calibrations_initial =
      CreateInitialCalibrations(T_VC, T_VL);

  // Build and solve problem
  OptimizerInputs optimizer_inputs{
      .target_params = target_params,
      .camera_params = camera_params,
      .lidar_measurements = lidar_measurements,
      .camera_measurements = camera_measurements,
      .loop_closure_measurements = loop_closure_measurements,
      .calibration_initials = calibrations_initial,
      .optimizer_config_path = GetFileLocationData("OptimizerConfig.json")};

  std::shared_ptr<CeresOptimizer> optimizer =
      std::make_shared<CeresOptimizer>(optimizer_inputs);
  optimizer->Solve();
  CalibrationResults calibrations_result = optimizer->GetResults();

  //   utils::OutputCalibrations(calibrations_initial,
  //                             "Initial Calibration Estimates:");
  //   utils::OutputCalibrations(calibrations_result, "Optimized
  //   Calibrations:");

  for (int i = 0; i < calibrations_result.size(); i++) {
    REQUIRE(calibrations_result[i].to_frame ==
            calibrations_initial[i].to_frame);
    REQUIRE(calibrations_result[i].from_frame ==
            calibrations_initial[i].from_frame);
    REQUIRE(vicon_calibration::utils::RoundMatrix(
                calibrations_result[i].transform, 4) ==
            vicon_calibration::utils::RoundMatrix(
                calibrations_initial[i].transform, 4));
  }
}

TEST_CASE("Test Ceres Optimizer With Perturbed Initials") {
  CreateTransforms();

  // create measurements
  TargetParamsVector target_params = GetTargetParams("DiamondTargetSim.json");
  CameraParamsVector camera_params =
      GetCameraParams("CamFactorIntrinsics.json");
  LidarMeasurements lidar_measurements =
      CreateLidarMeasurements(T_VL, T_VTs, *(target_params[0]));
  CameraMeasurements camera_measurements = CreateCameraMeasurements(
      T_VC, T_VTs, *(target_params[0]), *(camera_params[0]));
  LoopClosureMeasurements loop_closure_measurements;

  CalibrationResults calibrations_initial =
      CreateInitialCalibrations(T_VC, T_VL);
  CalibrationResults calibrations_perturbed =
      CreateInitialCalibrations(T_VC_pert, T_VL_pert);

  // Build and solve problem
  OptimizerInputs optimizer_inputs{
      .target_params = target_params,
      .camera_params = camera_params,
      .lidar_measurements = lidar_measurements,
      .camera_measurements = camera_measurements,
      .loop_closure_measurements = loop_closure_measurements,
      .calibration_initials = calibrations_perturbed,
      .optimizer_config_path = GetFileLocationData("OptimizerConfig.json")};

  std::shared_ptr<CeresOptimizer> optimizer =
      std::make_shared<CeresOptimizer>(optimizer_inputs);
  optimizer->Solve();
  CalibrationResults calibrations_result = optimizer->GetResults();

  // validate
//   utils::OutputCalibrations(calibrations_initial,
//                             "Initial Calibration Estimates:");
//   utils::OutputCalibrations(calibrations_perturbed, "Pertubed Calibrations:");
//   utils::OutputCalibrations(calibrations_result, "Optimized Calibrations:");

  for (int i = 0; i < calibrations_result.size(); i++) {
    REQUIRE(calibrations_result[i].to_frame ==
            calibrations_initial[i].to_frame);
    REQUIRE(calibrations_result[i].from_frame ==
            calibrations_initial[i].from_frame);
    REQUIRE(utils::RoundMatrix(calibrations_result[i].transform, 4) ==
            utils::RoundMatrix(calibrations_initial[i].transform, 4));
  }
}

TEST_CASE("Test with same data and not using Ceres Optimizer Class") {
  CreateTransforms();

  // create measurements
  TargetParamsVector target_params = GetTargetParams("DiamondTargetSim.json");
  CameraParamsVector camera_params =
      GetCameraParams("CamFactorIntrinsics.json");
  CameraMeasurements camera_measurements = CreateCameraMeasurements(
      T_VC, T_VTs, *(target_params[0]), *(camera_params[0]));
  LidarMeasurements lidar_measurements =
      CreateLidarMeasurements(T_VL, T_VTs, *(target_params[0]));

  // convert to ceres format
  Eigen::Matrix3d R = T_CV_pert.block(0, 0, 3, 3);
  Eigen::Quaternion<double> q = Eigen::Quaternion<double>(R);
  std::vector<std::vector<double>> results_perturbed_init;
  std::vector<double> tmp{q.w(),          q.x(),           q.y(),
                          q.z(),          T_CV_pert(0, 3), T_CV_pert(1, 3),
                          T_CV_pert(2, 3)};
  results_perturbed_init.push_back(tmp);

  R = T_LV_pert.block(0, 0, 3, 3);
  q = Eigen::Quaternion<double>(R);
  tmp = std::vector<double>{q.w(),          q.x(),           q.y(),
                            q.z(),          T_LV_pert(0, 3), T_LV_pert(1, 3),
                            T_LV_pert(2, 3)};
  results_perturbed_init.push_back(tmp);

  Eigen::Matrix4d T_ceres_initial_camera =
      utils::QuaternionAndTranslationToTransformMatrix(
          results_perturbed_init[0]);
  Eigen::Matrix4d T_ceres_initial_lidar =
      utils::QuaternionAndTranslationToTransformMatrix(
          results_perturbed_init[1]);

  // create problem and add parameters
  std::shared_ptr<ceres::Problem> problem = SetupCeresProblem();
  problem->AddParameterBlock(&(results_perturbed_init[0][0]), 7,
                             se3_parameterization_.get());
  problem->AddParameterBlock(&(results_perturbed_init[1][0]), 7,
                             se3_parameterization_.get());

  // get camera measurements and add cost functions
  std::shared_ptr<beam_calibration::CameraModel> camera_model =
      camera_params[0]->camera_model;
  for (int i = 0; i < target_params[0]->keypoints_camera.size(); i++) {
    Eigen::Vector3d P_TARGET = target_params[0]->keypoints_camera[i];
    Eigen::Vector3d P_VICONBASE = (T_VT * P_TARGET.homogeneous()).hnormalized();
    Eigen::Vector3d P_CAMERA_perf =
        (T_CV * T_VT * P_TARGET.homogeneous()).hnormalized();
    Eigen::Vector3d P_CAMERA_pert =
        (T_CV_pert * T_VT * P_TARGET.homogeneous()).hnormalized();
    opt<Eigen::Vector2d> pixels_true =
        camera_model->ProjectPointPrecise(P_CAMERA_perf);
    if (!pixels_true.has_value()) { continue; }
    std::unique_ptr<ceres::CostFunction> cost_function(
        CeresCameraCostFunction::Create(pixels_true.value(), P_VICONBASE,
                                        camera_model));
    problem->AddResidualBlock(cost_function.release(), loss_function_.get(),
                              &(results_perturbed_init[0][0]));
  }

  // get lidar measurements and add cost functions
  for (int i = 0; i < target_params[0]->keypoints_lidar.size(); i++) {
    Eigen::Vector3d P_TARGET = target_params[0]->keypoints_lidar[i];
    Eigen::Vector3d P_VICONBASE = (T_VT * P_TARGET.homogeneous()).hnormalized();
    Eigen::Vector3d P_LIDAR_perf =
        (T_LV * T_VT * P_TARGET.homogeneous()).hnormalized();
    Eigen::Vector3d P_LIDAR_pert =
        (T_LV_pert * T_VT * P_TARGET.homogeneous()).hnormalized();
    std::unique_ptr<ceres::CostFunction> cost_function(
        CeresLidarCostFunction::Create(P_LIDAR_perf, P_VICONBASE));
    problem->AddResidualBlock(cost_function.release(), loss_function_.get(),
                              &(results_perturbed_init[1][0]));
  }

  // solve
  ceres::Solver::Summary ceres_summary;
  ceres::Solve(ceres_solver_options_, problem.get(), &ceres_summary);

  // validate results
  Eigen::Matrix4d T_CV_opt = utils::QuaternionAndTranslationToTransformMatrix(
      results_perturbed_init[0]);
  Eigen::Matrix4d T_LV_opt = utils::QuaternionAndTranslationToTransformMatrix(
      results_perturbed_init[1]);
  REQUIRE(utils::RoundMatrix(T_CV_opt, 4) == utils::RoundMatrix(T_CV, 4));
  REQUIRE(utils::RoundMatrix(T_LV_opt, 4) == utils::RoundMatrix(T_LV, 4));
}