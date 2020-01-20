#pragma once

#include "vicon_calibration/TfTree.h"
#include "vicon_calibration/params.h"
#include <Eigen/Geometry>
#include <beam_calibration/CameraModel.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vicon_calibration {

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudColor;

#ifndef FILENAME
#define FILENAME                                                               \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#ifndef LOG_ERROR
#define LOG_ERROR(M, ...)                                                      \
  fprintf(stderr, "[ERROR] [%s:%d] " M "\n", FILENAME, __LINE__, ##__VA_ARGS__)
#endif

#ifndef LOG_INFO
#define LOG_INFO(M, ...) fprintf(stdout, "[INFO] " M "\n", ##__VA_ARGS__)
#endif

#ifndef LOG_WARN
#define LOG_WARN(M, ...) fprintf(stdout, "[WARNING] " M "\n", ##__VA_ARGS__)
#endif

#ifndef RAD_TO_DEG
#define RAD_TO_DEG 57.29577951
#endif

namespace utils {

double time_now(void);

/** Wraps input angle to the interval [-PI, PI).
 * @param[in] angle the original angle.
 * @return the wrapped angle.
 */
double WrapToPi(const double &angle);

/** Wraps input angle to the interval [0, 2*PI).
 * @param[in] angle the original angle.
 * @return the wrapped angle.
 */
double WrapToTwoPi(const double &angle);

Eigen::MatrixXd RoundMatrix(const Eigen::MatrixXd &M, const int &precision);

bool IsRotationMatrix(const Eigen::Matrix3d &R);

bool IsTransformationMatrix(const Eigen::Matrix4d &T);

Eigen::Matrix4d PerturbTransform(const Eigen::Matrix4d &T_in,
                                 const Eigen::VectorXd &perturbations);

Eigen::Vector3d InvSkewTransform(const Eigen::Matrix3d &M);

Eigen::Matrix3d SkewTransform(const Eigen::Vector3d &V);

Eigen::Vector3d RToLieAlgebra(const Eigen::Matrix3d &R);

Eigen::Matrix3d LieAlgebraToR(const Eigen::Vector3d &eps);

Eigen::Matrix4d InvertTransform(const Eigen::MatrixXd &T);

cv::Mat DrawCoordinateFrame(
    const cv::Mat &img_in, const Eigen::MatrixXd &T_cam_frame,
    const std::shared_ptr<beam_calibration::CameraModel> &camera_model,
    const double &scale, const bool &images_distorted);

cv::Mat ProjectPointsToImage(
    const cv::Mat &img,
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> &cloud,
    const Eigen::MatrixXd &T_IMAGE_CLOUD,
    std::shared_ptr<beam_calibration::CameraModel> &camera_model);

void OutputTransformInformation(const Eigen::Affine3d &T,
                                const std::string &transform_name);

void OutputTransformInformation(const Eigen::Matrix4d &T,
                                const std::string &transform_name);

void OutputCalibrations(
    const std::vector<vicon_calibration::CalibrationResult> &calib,
    const std::string &output_string);

inline Eigen::Vector3d PCLPointToEigen(const pcl::PointXYZ &pt_in) {
  return Eigen::Vector3d(pt_in.x, pt_in.y, pt_in.z);
}

inline Eigen::Vector2d PCLPixelToEigen(const pcl::PointXY &pt_in) {
  return Eigen::Vector2d(pt_in.x, pt_in.y);
}

inline pcl::PointXYZ EigenPointToPCL(const Eigen::Vector3d &pt_in) {
  pcl::PointXYZ pt_out;
  pt_out.x = pt_in[0];
  pt_out.y = pt_in[1];
  pt_out.z = pt_in[2];
  return pt_out;
}

inline pcl::PointXY EigenPixelToPCL(const Eigen::Vector2d &pt_in) {
  pcl::PointXY pt_out;
  pt_out.x = pt_in[0];
  pt_out.y = pt_in[1];
  return pt_out;
}

inline gtsam::Point3 EigenPointToGTSAM(const Eigen::Vector3d &pt_in) {
  return gtsam::Point3(pt_in[0], pt_in[1], pt_in[2]);
}

inline gtsam::Point2 EigenPixelToGTSAM(const Eigen::Vector2d &pt_in) {
  return gtsam::Point2(pt_in[0], pt_in[1]);
}

inline gtsam::Point3 PCLPointToGTSAM(const pcl::PointXYZ &pt_in) {
  return gtsam::Point3(pt_in.x, pt_in.y, pt_in.z);
}

inline gtsam::Point2 PCLPixelToGTSAM(const pcl::PointXY &pt_in) {
  return gtsam::Point2(pt_in.x, pt_in.y);
}

inline Eigen::Vector3d HomoPointToPoint(const Eigen::Vector4d &pt_in) {
  return Eigen::Vector3d(pt_in[0], pt_in[1], pt_in[2]);
}

inline Eigen::Vector4d PointToHomoPoint(const Eigen::Vector3d &pt_in) {
  return Eigen::Vector4d(pt_in[0], pt_in[1], pt_in[2], 1);
}

std::string
ConvertTimeToDate(const std::chrono::system_clock::time_point &time_);

std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
GetTargetLocation(
    const std::vector<std::shared_ptr<vicon_calibration::TargetParams>>
        &target_params,
    const std::string &vicon_baselink_frame, const ros::Time &lookup_time,
    const std::shared_ptr<vicon_calibration::TfTree> &lookup_tree);

/**
 * @brief gets full name to file inside data subfolder
 * @param file_name name of file to find
 * @return full file path
 */
std::string GetFilePathData(const std::string &file_name);

/**
 * @brief gets full name to file inside config subfolder
 * @param file_name name of file to find
 * @return full file path
 */
std::string GetFilePathConfig(const std::string &file_name);

/**
 * @brief gets full name to file inside test data subfolder
 * @param file_name name of file to find
 * @return full file path
 */
std::string GetFilePathTestData(const std::string &file_name);

/**
 * @brief gets full name to file inside test clouds subfolder
 * @param file_name name of file to find
 * @return full file path
 */
std::string GetFilePathTestClouds(const std::string &file_name);

/**
 * @brief gets full name to file inside test bag subfolder
 * @param file_name name of file to find
 * @return full file path
 */
std::string GetFilePathTestBags(const std::string &file_name);

} // namespace utils

} // namespace vicon_calibration
