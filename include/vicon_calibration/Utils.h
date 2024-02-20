#pragma once

#include <sys/time.h>
#include <chrono>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <sensor_msgs/Image.h>

#include <vicon_calibration/Log.h>
#include <vicon_calibration/Aliases.h>
#include <vicon_calibration/Params.h>
#include <vicon_calibration/TfTree.h>
#include <vicon_calibration/camera_models/CameraModel.h>

namespace vicon_calibration {

static std::string utils_string_tmp;

// Forward declarations
struct CalibrationResult;
struct TargetParams;
struct CameraParams;
struct LidarParams;
enum class SensorType;
typedef std::vector<CalibrationResult> CalibrationResults;

namespace utils {

/**
 * @brief Simple timer object
 */
struct HighResolutionTimer {
  HighResolutionTimer() : start_time(take_time_stamp()) {}

  /**
   * @brief Restart timer
   */
  void restart() { start_time = take_time_stamp(); }

  /**
   * @brief Return elapsed time in seconds.
   * @return
   */
  double elapsed() const {
    return double(take_time_stamp() - start_time) * 1e-9;
  }

  std::uint64_t elapsed_nanoseconds() const {
    return take_time_stamp() - start_time;
  }

 protected:
  static std::uint64_t take_time_stamp() {
    return std::uint64_t(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
  }

 private:
  std::uint64_t start_time;
};

double RandomNumber(const double& min, const double& max);

/**
 * @Brief Wraps input angle to the interval [-PI, PI).
 * @param angle the original angle.
 * @return the wrapped angle.
 */
double WrapToPi(double angle);

/**
 * @Brief Wraps input angle to the interval [0, 2*PI).
 * @param angle the original angle.
 * @return the wrapped angle.
 */
double WrapToTwoPi(double angle);

/**
 * @Brief Return the smallest difference between two angles. This takes into
 * account the case where one or both angles are outside (0, 2PI). By smallest
 * error, we mean for example: GetSmallestAngleErrorDeg(0.1PI, 1.9PI) = 0.2PI,
 * not 1.8PI
 * @param angle 1 in radians
 * @param angle 2 in radians
 * @return error in radians
 */
double GetSmallestAngleErrorRad(double angle1, double angle2);

double CalculateTranslationErrorNorm(const Eigen::Vector3d& t1,
                                     const Eigen::Vector3d& t2);

double CalculateRotationError(const Eigen::Matrix3d& r1,
                              const Eigen::Matrix3d& r2);

double VectorAverage(const std::vector<double>& v);

double VectorStdev(const std::vector<double>& v);

std::string VectorToString(const std::vector<double>& v);

/** Converts degrees to radians. */
double DegToRad(double d);

/** Converts radians to degrees. */
double RadToDeg(double r);

Eigen::Matrix4d RoundMatrix(const Eigen::Matrix4d& M, const int& precision);

Eigen::Matrix3d RoundMatrix(const Eigen::Matrix3d& M, const int& precision);

bool IsRotationMatrix(const Eigen::Matrix3d& R);

bool IsTransformationMatrix(const Eigen::Matrix4d& T);

/** Peturbs a transformation
 * @param[in] T_in the original transformation matrix
 * @param[in] perturbations [rx(rad), ry(rad), rz(rad), tx(m), ty(m),
 * tx(m)]
 * @return perturbed transformation
 */
Eigen::Matrix4d PerturbTransformRadM(const Eigen::Matrix4d& T_in,
                                     const Eigen::VectorXd& perturbations);

/** Peturbs a transformation
 * @param[in] T_in the original transformation matrix
 * @param[in] perturbations [rx(deg), ry(deg), rz(deg), tx(m), ty(m),
 * tx(m)]
 * @return perturbed transformation
 */
Eigen::Matrix4d PerturbTransformDegM(const Eigen::Matrix4d& T_in,
                                     const Eigen::VectorXd& perturbations);

Eigen::Matrix4d BuildTransformEulerDegM(double rollInDeg, double pitchInDeg,
                                        double yawInDeg, double tx, double ty,
                                        double tz);

Eigen::Matrix3d SkewTransform(const Eigen::Vector3d& V);

Eigen::Matrix3d LieAlgebraToR(const Eigen::Vector3d& eps);

Eigen::Matrix4d InvertTransform(const Eigen::Matrix4d& T);

Eigen::Matrix4d QuaternionAndTranslationToTransformMatrix(
    const std::vector<double>& pose);

// [qw qx qy qz tx ty tx]
std::vector<double> TransformMatrixToQuaternionAndTranslation(
    const Eigen::Matrix4d& T);

cv::Mat DrawCoordinateFrame(const cv::Mat& img_in,
                            const Eigen::Matrix4d& T_Cam_Frame,
                            const std::shared_ptr<CameraModel>& camera_model,
                            const double& scale);

cv::Mat ProjectPointsToImage(
    const cv::Mat& img, std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const Eigen::Matrix4d& T_IMAGE_CLOUD,
    std::shared_ptr<CameraModel>& camera_model);

std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> ProjectPoints(
    std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    std::shared_ptr<CameraModel>& camera_model, const Eigen::Matrix4d& T);

PointCloudColor::Ptr ColorPointCloud(const PointCloud::Ptr& cloud, const int& r,
                                     const int& g, const int& b);

void OutputTransformInformation(const Eigen::Affine3d& T,
                                const std::string& transform_name);

void OutputTransformInformation(const Eigen::Matrix4d& T,
                                const std::string& transform_name);

void OutputCalibrations(
    const std::vector<vicon_calibration::CalibrationResult>& calib,
    const std::string& output_string);

std::string ConvertTimeToDate(
    const std::chrono::system_clock::time_point& time_);

std::vector<Eigen::Affine3d, AlignAff3d> GetTargetLocation(
    const std::vector<std::shared_ptr<vicon_calibration::TargetParams>>&
        target_params,
    const std::string& vicon_baselink_frame, const ros::Time& lookup_time,
    const std::shared_ptr<vicon_calibration::TfTree>& lookup_tree);

/**
 * @brief gets full name to file inside data subfolder
 * @param file_name name of file to find
 * @return full file path
 */
std::string GetFilePathData(const std::string& file_name);

/**
 * @brief gets full name to file inside config subfolder
 * @param file_name name of file to find
 * @return full file path
 */
std::string GetFilePathConfig(const std::string& file_name);

/**
 * @brief gets full name to file inside test data subfolder
 * @param file_name name of file to find
 * @return full file path
 */
std::string GetFilePathTestData(const std::string& file_name);

/**
 * @brief gets full name to file inside test clouds subfolder
 * @param file_name name of file to find
 * @return full file path
 */
std::string GetFilePathTestClouds(const std::string& file_name);

/**
 * @brief gets full name to file inside test bag subfolder
 * @param file_name name of file to find
 * @return full file path
 */
std::string GetFilePathTestBags(const std::string& file_name);

void GetScreenResolution(int& horizontal, int& vertical);

Eigen::Matrix4d GetT_Robot_Sensor(const CalibrationResults& calibs,
                                      SensorType type, uint8_t sensor_id,
                                      bool& success);

int DepthStrToInt(const std::string depth);

int GetCvType(const std::string& encoding);

/**
 * @brief Converts a ROS Image to a cv::Mat by sharing the data or changing
 * its endianness if needed
 * @param source ros image message
 * @return cv::Mat of image
 */
cv::Mat RosImgToMat(const sensor_msgs::Image& source);

enum PointCloudFileType { PCDBINARY, PCDASCII, PLYBINARY, PLYASCII };

/** Map for storing string input */
static std::map<std::string, PointCloudFileType> PointCloudFileTypeStringMap = {
    {"PCDBINARY", PointCloudFileType::PCDBINARY},
    {"PCDASCII", PointCloudFileType::PCDASCII},
    {"PLYBINARY", PointCloudFileType::PLYBINARY},
    {"PLYASCII", PointCloudFileType::PLYASCII}};

/** Map for storing file extension with each type of point cloud file */
static std::map<PointCloudFileType, std::string>
    PointCloudFileTypeExtensionMap = {{PointCloudFileType::PCDBINARY, ".pcd"},
                                      {PointCloudFileType::PCDASCII, ".pcd"},
                                      {PointCloudFileType::PLYBINARY, ".ply"},
                                      {PointCloudFileType::PLYASCII, ".ply"}};

/** function for listing types of PointCloud files */
inline std::string GetPointCloudFileTypes() {
  std::string types;
  for (auto it = PointCloudFileTypeStringMap.begin();
       it != PointCloudFileTypeStringMap.end(); it++) {
    types += it->first;
    types += ", ";
  }
  types.erase(types.end() - 2, types.end());
  return types;
}

bool HasExtension(const std::string& input, const std::string& extension);

/**
 * @brief function for saving PCD point clouds. Using the regular pcl i/o will
 * throw exceptions if the point clouds are empty, but we don't always want
 * this. This is a wrapper around the pcl save functions to avoid that and save
 * different types of file
 * @param filename full path to file
 * @param input_cloud
 * @param file_type enum class for point cloud file type
 * @param error_type string with the resulting error if save was unsuccessful
 * @return true if successful
 */
template <class PointT>
inline bool
    SavePointCloud(const std::string& filename,
                   const pcl::PointCloud<PointT>& cloud,
                   PointCloudFileType file_type = PointCloudFileType::PCDBINARY,
                   std::string& error_type = utils_string_tmp) {
  // check extension
  std::string extension_should_be = PointCloudFileTypeExtensionMap[file_type];
  if (!HasExtension(filename, extension_should_be)) {
    error_type = "Invalid file extension. Input file: " + filename +
                 " . Extension should be: " + extension_should_be;
    return false;
  }

  // check path exists
  boost::filesystem::path path(filename);
  if (!boost::filesystem::exists(path.parent_path())) {
    error_type =
        "File path parent directory does not exist. Input file: " + filename;
    return false;
  }

  // check pointcloud isn't empty
  if (cloud.size() == 0) {
    error_type = "Empty point cloud.";
    return false;
  }

  // try to save cloud
  pcl::PLYWriter writer;
  try {
    switch (file_type) {
      case PointCloudFileType::PCDASCII:
        pcl::io::savePCDFileASCII(filename, cloud);
        break;
      case PointCloudFileType::PCDBINARY:
        pcl::io::savePCDFileBinary(filename, cloud);
        break;
      case PointCloudFileType::PLYASCII:
        writer.write<PointT>(filename, cloud, false);
        break;
      case PointCloudFileType::PLYBINARY:
        writer.write<PointT>(filename, cloud, true);
        break;
    }
  } catch (pcl::PCLException& e) {
    error_type = "Exception throw by pcl: " + std::string(e.detailedMessage());
    return false;
  }

  return true;
}

/**
 * @brief enum class for storing error types for the CheckJson function.
 *
 *  NONE: no error
 *  MISSING: file does not exist
 *  FILETYPE: file extension is not .json
 *  EMPTY: json file is empty
 *
 */
enum JsonReadErrorType { NONE, FILETYPE, MISSING, EMPTY };

static JsonReadErrorType tmp_json_read_error_type_ = JsonReadErrorType::NONE;

/**
 * @brief reads a json and does some checks:
 *
 *  1. Check that file exists
 *  2. Check that file has the .json extension
 *  3. Check the json is not null
 *
 * @param filename full path to json
 * @param J reference to json to fill
 * @param error_type optional reference to enum class of error type
 * @param output_error optional bool to set if you want this function to output
 * the error type
 * @return true if passed
 */
bool ReadJson(const std::string& filename, nlohmann::json& J,
              JsonReadErrorType& error_type = tmp_json_read_error_type_,
              bool output_error = true);

}  // namespace utils

}  // namespace vicon_calibration
