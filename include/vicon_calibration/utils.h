#pragma once

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

namespace utils {

Eigen::MatrixXd RoundMatrix(const Eigen::MatrixXd &M, int precision);

bool IsRotationMatrix(const Eigen::Matrix3d R);

bool IsTransformationMatrix(const Eigen::Matrix4d T);

Eigen::Matrix4d PerturbTransform(const Eigen::Matrix4d &T_in,
                                 const Eigen::VectorXd &perturbations);

Eigen::Vector3d InvSkewTransform(const Eigen::Matrix3d &M);

Eigen::Matrix3d SkewTransform(const Eigen::Vector3d &V);

Eigen::Vector3d RToLieAlgebra(const Eigen::Matrix3d &R);

Eigen::Matrix3d LieAlgebraToR(const Eigen::Vector3d &eps);

Eigen::Matrix4d InvertTransform(const Eigen::Matrix4d &T);

cv::Mat
DrawCoordinateFrame(cv::Mat &img_in, Eigen::MatrixXd &T_cam_frame,
                    std::shared_ptr<beam_calibration::CameraModel> camera_model,
                    double &scale, bool images_distorted);

void OutputTransformInformation(Eigen::Affine3d &T, std::string transform_name);

void OutputCalibrations(
    std::vector<vicon_calibration::CalibrationResult> &calib,
    std::string output_string);

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

} // namespace utils

} // namespace vicon_calibration
