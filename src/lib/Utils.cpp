#include <vicon_calibration/Utils.h>

#include <regex>

#include <X11/Xlib.h>
#include <boost/endian/conversion.hpp>
#include <boost/optional/optional_io.hpp>
#include <sensor_msgs/image_encodings.h>
#include <unsupported/Eigen/MatrixFunctions>

namespace enc = sensor_msgs::image_encodings;

namespace vicon_calibration { namespace utils {

double RandomNumber(const double& min, const double& max) {
  double random =
      static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
  return random * (max - min) + min;
}

double WrapToPi(double angle) {
  double wrapped_angle = WrapToTwoPi(angle + M_PI) - M_PI;
  return wrapped_angle;
}

double WrapToTwoPi(double angle) {
  double wrapped_angle = fmod(angle, 2 * M_PI);
  if (wrapped_angle < 0) { wrapped_angle += 2 * M_PI; }
  return wrapped_angle;
}

double DegToRad(double d) {
  return d * (M_PI / 180);
}

double RadToDeg(double r) {
  return r * (180 / M_PI);
}

double GetSmallestAngleErrorRad(double angle1, double angle2) {
  double angle1_wrapped = WrapToTwoPi(angle1);
  double angle2_wrapped = WrapToTwoPi(angle2);
  if (std::min(angle1_wrapped, angle2_wrapped) < M_PI / 2 &&
      std::max(angle1_wrapped, angle2_wrapped) > 3 * M_PI / 2) {
    return 2 * M_PI - (std::max(angle1_wrapped, angle2_wrapped) -
                       std::min(angle1_wrapped, angle2_wrapped));
  } else {
    return std::max(angle1_wrapped, angle2_wrapped) -
           std::min(angle1_wrapped, angle2_wrapped);
  }
}

double VectorStdev(const std::vector<double>& v) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / v.size();

  std::vector<double> diff(v.size());
  std::transform(v.begin(), v.end(), diff.begin(),
                 std::bind2nd(std::minus<double>(), mean));
  double sq_sum =
      std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / v.size());
  return stdev;
}

double VectorAverage(const std::vector<double>& v) {
  double sum = 0;
  for (int i = 0; i < v.size(); i++) {
    double value = v[i];
    sum += value;
  }
  double avg = sum / v.size();
  return avg;
}

std::string VectorToString(const std::vector<double>& v) {
  std::string output = "[";
  for (int i = 0; i < v.size(); i++) {
    double value = v[i];
    output += std::to_string(value);
    output += ", ";
  }
  output.erase(output.end() - 2, output.end());
  return output += " ]";
}

double CalculateTranslationErrorNorm(const Eigen::Vector3d& t1,
                                     const Eigen::Vector3d& t2) {
  Eigen::Vector3d error = t1 - t2;
  return error.norm();
}

double CalculateRotationError(const Eigen::Matrix3d& r1,
                              const Eigen::Matrix3d& r2) {
  double error_r1 = Eigen::AngleAxis<double>(r1).angle();
  double error_r2 = Eigen::AngleAxis<double>(r2).angle();
  return std::abs(error_r1 - error_r2);
}

Eigen::Matrix3d RoundMatrix(const Eigen::Matrix3d& M, int precision) {
  Eigen::Matrix3d Mround;
  for (int i = 0; i < M.rows(); i++) {
    for (int j = 0; j < M.cols(); j++) {
      Mround(i, j) = std::round(M(i, j) * std::pow(10, precision)) /
                     std::pow(10, precision);
    }
  }
  return Mround;
}

Eigen::Matrix4d RoundMatrix(const Eigen::Matrix4d& M, int precision) {
  Eigen::Matrix4d Mround;
  for (int i = 0; i < M.rows(); i++) {
    for (int j = 0; j < M.cols(); j++) {
      Mround(i, j) = std::round(M(i, j) * std::pow(10, precision)) /
                     std::pow(10, precision);
    }
  }
  return Mround;
}

bool IsRotationMatrix(const Eigen::Matrix3d& R) {
  int precision = 3;
  Eigen::Matrix3d shouldBeIdentity = R * R.transpose();
  shouldBeIdentity = RoundMatrix(shouldBeIdentity, precision);
  double detR = R.determinant();
  double detRRound = std::round(detR * precision) / precision;
  if (shouldBeIdentity.isIdentity() && detRRound == 1) {
    return 1;
  } else {
    return 0;
    if (!shouldBeIdentity.isIdentity()) {
      LOG_ERROR("Rotation matrix invalid. R x R^T != I");
    } else {
      LOG_ERROR(
          "Rotation matrix invalid. Determinant not equal to 1. det = %.5f",
          detR);
    }
  }
}

bool IsTransformationMatrix(const Eigen::Matrix4d& T) {
  Eigen::Matrix3d R = T.block(0, 0, 3, 3);
  bool homoFormValid;
  bool tValid;

  // check translation for infinity or nan
  if (std::isinf(T(0, 3)) || std::isinf(T(1, 3)) || std::isinf(T(2, 3)) ||
      std::isnan(T(0, 3)) || std::isnan(T(1, 3)) || std::isnan(T(2, 3))) {
    tValid = 0;
    LOG_ERROR("Translation invalid.t = [%.5f, %.5f, %.5f]", T(0, 3), T(1, 3),
              T(2, 3));
  } else {
    tValid = 1;
  }

  // check that bottom row is [0 0 0 1]
  if (T(3, 0) == 0 && T(3, 1) == 0 && T(3, 2) == 0 && T(3, 3) == 1) {
    homoFormValid = 1;
  } else {
    homoFormValid = 0;
    LOG_ERROR(
        "Transform not homographic form. Last row: [%.5f, %.5f, %.5f, %.5f]",
        T(3, 0), T(3, 1), T(3, 2), T(3, 3));
  }

  if (homoFormValid && tValid && IsRotationMatrix(R)) {
    return 1;
  } else {
    return 0;
  }
}

Eigen::Matrix4d PerturbTransformRadM(const Eigen::Matrix4d& T_in,
                                     const Eigen::VectorXd& perturbations) {
  Eigen::Vector3d r_perturb = perturbations.block(0, 0, 3, 1);
  Eigen::Vector3d t_perturb = perturbations.block(3, 0, 3, 1);
  Eigen::Matrix3d R_in = T_in.block(0, 0, 3, 3);
  Eigen::Matrix3d R_out = LieAlgebraToR(r_perturb) * R_in;
  Eigen::Matrix4d T_out;
  T_out.setIdentity();
  T_out.block(0, 3, 3, 1) = T_in.block(0, 3, 3, 1) + t_perturb;
  T_out.block(0, 0, 3, 3) = R_out;
  return T_out;
}

Eigen::Matrix4d PerturbTransformDegM(const Eigen::Matrix4d& T_in,
                                     const Eigen::VectorXd& perturbations) {
  Eigen::VectorXd perturbations_rad(perturbations);
  perturbations_rad[0] = utils::DegToRad(perturbations_rad[0]);
  perturbations_rad[1] = utils::DegToRad(perturbations_rad[1]);
  perturbations_rad[2] = utils::DegToRad(perturbations_rad[2]);
  return PerturbTransformRadM(T_in, perturbations_rad);
}

Eigen::Matrix4d BuildTransformEulerDegM(double rollInDeg, double pitchInDeg,
                                        double yawInDeg, double tx, double ty,
                                        double tz) {
  Eigen::Vector3d t(tx, ty, tz);
  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(DegToRad(rollInDeg), Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(DegToRad(pitchInDeg), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(DegToRad(yawInDeg), Eigen::Vector3d::UnitZ());
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block(0, 0, 3, 3) = R;
  T.block(0, 3, 3, 1) = t;
  return T;
}

Eigen::Matrix3d SkewTransform(const Eigen::Vector3d& V) {
  Eigen::Matrix3d M;
  M(0, 0) = 0;
  M(0, 1) = -V(2, 0);
  M(0, 2) = V(1, 0);
  M(1, 0) = V(2, 0);
  M(1, 1) = 0;
  M(1, 2) = -V(0, 0);
  M(2, 0) = -V(1, 0);
  M(2, 1) = V(0, 0);
  M(2, 2) = 0;
  return M;
}

Eigen::Matrix3d LieAlgebraToR(const Eigen::Vector3d& eps) {
  return SkewTransform(eps).exp();
}

Eigen::Matrix4d InvertTransform(const Eigen::Matrix4d& T) {
  Eigen::Matrix4d T_inv;
  T_inv.setIdentity();
  T_inv.block(0, 0, 3, 3) = T.block(0, 0, 3, 3).transpose();
  T_inv.block(0, 3, 3, 1) =
      -T.block(0, 0, 3, 3).transpose() * T.block(0, 3, 3, 1);
  return T_inv;
}

Eigen::Matrix4d
    QuaternionAndTranslationToTransformMatrix(const std::vector<double>& pose) {
  Eigen::Quaternion<double> quaternion{pose[0], pose[1], pose[2], pose[3]};
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block(0, 0, 3, 3) = quaternion.toRotationMatrix();
  T(0, 3) = pose[4];
  T(1, 3) = pose[5];
  T(2, 3) = pose[6];
  return T;
}

cv::Mat DrawCoordinateFrame(
    const cv::Mat& img_in, const Eigen::Matrix4d& T_Cam_Frame,
    const std::shared_ptr<vicon_calibration::CameraModel>& camera_model,
    const double& scale) {
  cv::Mat img_out;
  img_out = img_in.clone();
  Eigen::Vector4d origin(0, 0, 0, 1), x_end(scale, 0, 0, 1),
      y_end(0, scale, 0, 1), z_end(0, 0, scale, 1);
  Eigen::Vector4d o_trans = T_Cam_Frame * origin, x_trans = T_Cam_Frame * x_end,
                  y_trans = T_Cam_Frame * y_end, z_trans = T_Cam_Frame * z_end;

  Eigen::Vector3d o(o_trans(0), o_trans(1), o_trans(2)),
      x(x_trans(0), x_trans(1), x_trans(2)),
      y(y_trans(0), y_trans(1), y_trans(2)),
      z(z_trans(0), z_trans(1), z_trans(2));

  Eigen::Vector2d start_pixel;
  bool start_pixel_valid;
  Eigen::Vector2d end_pixel_x;
  bool end_pixel_x_valid;
  Eigen::Vector2d end_pixel_y;
  bool end_pixel_y_valid;
  Eigen::Vector2d end_pixel_z;
  bool end_pixel_z_valid;

  camera_model->ProjectPoint(o, start_pixel, start_pixel_valid);
  camera_model->ProjectPoint(x, end_pixel_x, end_pixel_x_valid);
  camera_model->ProjectPoint(y, end_pixel_y, end_pixel_y_valid);
  camera_model->ProjectPoint(z, end_pixel_z, end_pixel_z_valid);

  if (!start_pixel_valid || !end_pixel_x_valid || !end_pixel_y_valid ||
      !end_pixel_z_valid) {
    LOG_WARN("Unable to draw coordinate frame. Frame exceeds image dimensions");
    return img_out;
  }

  cv::Point start, end_x, end_y, end_z;
  start.x = start_pixel(0);
  start.y = start_pixel(1);
  end_x.x = end_pixel_x(0);
  end_x.y = end_pixel_x(1);
  end_y.x = end_pixel_y(0);
  end_y.y = end_pixel_y(1);
  end_z.x = end_pixel_z(0);
  end_z.y = end_pixel_z(1);

  cv::Scalar colourX(0, 0, 255); // BGR
  cv::Scalar colourY(0, 255, 0);
  cv::Scalar colourZ(255, 0, 0);
  int thickness = 3;

  cv::line(img_out, start, end_x, colourX, thickness);
  cv::line(img_out, start, end_y, colourY, thickness);
  cv::line(img_out, start, end_z, colourZ, thickness);

  return img_out;
}

cv::Mat ProjectPointsToImage(const cv::Mat& img,
                             std::shared_ptr<PointCloud>& cloud,
                             const Eigen::Matrix4d& T_IMAGE_CLOUD,
                             std::shared_ptr<CameraModel>& camera_model) {
  cv::Mat img_out;
  img_out = img.clone();
  for (int i = 0; i < cloud->size(); i++) {
    Eigen::Vector4d point(cloud->at(i).x, cloud->at(i).y, cloud->at(i).z, 1);
    Eigen::Vector4d point_transformed = T_IMAGE_CLOUD * point;
    bool pixel_valid;
    Eigen::Vector2d pixel;
    camera_model->ProjectPoint(point_transformed.hnormalized(), pixel,
                               pixel_valid);
    if (!pixel_valid) { continue; }
    cv::circle(img_out, cv::Point(pixel[0], pixel[1]), 2,
               cv::Scalar(0, 255, 0));
  }
  return img_out;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
    ProjectPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                  std::shared_ptr<CameraModel>& camera_model,
                  const Eigen::Matrix4d& T) {
  auto projected_points = std::make_shared<PointCloud>();
  for (int i = 0; i < cloud->size(); i++) {
    Eigen::Vector4d point(cloud->at(i).x, cloud->at(i).y, cloud->at(i).z, 1);
    Eigen::Vector4d point_transformed = T * point;
    bool pixel_valid;
    Eigen::Vector2d pixel;
    camera_model->ProjectPoint(point_transformed.hnormalized(), pixel,
                               pixel_valid);
    if (!pixel_valid) { continue; }
    pcl::PointXYZ point_projected(pixel[0], pixel[1], 0);
    projected_points->push_back(point_projected);
  }
  return projected_points;
}

PointCloudColor::Ptr ColorPointCloud(const PointCloud::Ptr& cloud, int r, int g,
                                     int b) {
  auto coloured_cloud = std::make_shared<PointCloudColor>();
  uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                  static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
  for (PointCloud::iterator it = cloud->begin(); it != cloud->end(); ++it) {
    pcl::PointXYZRGB point;
    point.x = it->x;
    point.y = it->y;
    point.z = it->z;
    point.rgb = *reinterpret_cast<float*>(&rgb);
    coloured_cloud->push_back(point);
  }
  return coloured_cloud;
}

void OutputTransformInformation(const Eigen::Affine3d& T,
                                const std::string& transform_name) {
  OutputTransformInformation(T.matrix(), transform_name);
}

void OutputTransformInformation(const Eigen::Matrix4d& T,
                                const std::string& transform_name) {
  Eigen::Matrix3d R = T.block(0, 0, 3, 3);
  Eigen::Vector3d rpy = R.eulerAngles(0, 1, 2);
  std::cout << transform_name << ":\n"
            << T << "\n"
            << "rpy (deg): [" << utils::RadToDeg(utils::WrapToPi(rpy[0]))
            << ", " << utils::RadToDeg(utils::WrapToPi(rpy[1])) << ", "
            << utils::RadToDeg(utils::WrapToPi(rpy[2])) << "]\n";
}

void OutputCalibrations(
    const std::vector<vicon_calibration::CalibrationResult>& calib,
    const std::string& output_string) {
  std::cout << "----------------------\n" << output_string << "\n";
  for (uint16_t i = 0; i < calib.size(); i++) {
    Eigen::Matrix4d T = calib[i].transform;
    Eigen::Matrix3d R = T.block(0, 0, 3, 3);
    Eigen::Vector3d rpy = R.eulerAngles(0, 1, 2);
    std::cout << "T_" << calib[i].to_frame << "_" << calib[i].from_frame
              << ":\n"
              << T << "\n"
              << "rpy (deg): [" << utils::RadToDeg(utils::WrapToPi(rpy[0]))
              << ", " << utils::RadToDeg(utils::WrapToPi(rpy[1])) << ", "
              << utils::RadToDeg(utils::WrapToPi(rpy[2])) << "]\n";
  }
}

void OutputTargetCorrections(const std::vector<Eigen::Matrix4d>& Ts) {
  std::cout << "----------------------\n"
            << "Target Corrections"
            << "\n";
  for (const auto& T : Ts) {
    Eigen::Matrix3d R = T.block(0, 0, 3, 3);
    Eigen::Vector3d rpy = R.eulerAngles(0, 1, 2);
    std::cout << "T_TargetCorrected_Target:\n"
              << T << "\n"
              << "rpy (deg): [" << utils::RadToDeg(utils::WrapToPi(rpy[0]))
              << ", " << utils::RadToDeg(utils::WrapToPi(rpy[1])) << ", "
              << utils::RadToDeg(utils::WrapToPi(rpy[2])) << "]\n";
  }
}

std::string
    ConvertTimeToDate(const std::chrono::system_clock::time_point& time_) {
  using namespace std;
  using namespace std::chrono;
  system_clock::duration tp = time_.time_since_epoch();
  time_t tt = system_clock::to_time_t(time_);
  tm local_tm = *localtime(&tt);

  string outputTime =
      to_string(local_tm.tm_year + 1900) + "_" +
      to_string(local_tm.tm_mon + 1) + "_" + to_string(local_tm.tm_mday) + "_" +
      to_string(local_tm.tm_hour) + "_" + to_string(local_tm.tm_min) + "_" +
      to_string(local_tm.tm_sec);
  return outputTime;
}

std::vector<Eigen::Affine3d> GetTargetLocation(
    const std::vector<std::shared_ptr<vicon_calibration::TargetParams>>&
        target_params,
    const std::string& vicon_baselink_frame, const ros::Time& lookup_time,
    const std::shared_ptr<vicon_calibration::TfTree>& lookup_tree) {
  std::vector<Eigen::Affine3d> T_Robot_Targets;
  for (uint8_t n = 0; n < target_params.size(); n++) {
    Eigen::Affine3d T_Robot_Target;
    T_Robot_Target = lookup_tree->GetTransformEigen(
        vicon_baselink_frame, target_params[n]->frame_id, lookup_time);
    T_Robot_Targets.push_back(T_Robot_Target);
  }
  return T_Robot_Targets;
}

std::string GetFilePathData(const std::string& file_name) {
  std::string file_location = __FILE__;
  std::string current_file = "src/lib/utils.cpp";
  file_location.erase(file_location.end() - current_file.size(),
                      file_location.end());
  file_location += "data/";
  file_location += file_name;
  return file_location;
}

std::string GetFilePathConfig(const std::string& file_name) {
  std::string file_location = __FILE__;
  std::string current_file = "src/lib/utils.cpp";
  file_location.erase(file_location.end() - current_file.size(),
                      file_location.end());
  file_location += "config/";
  file_location += file_name;
  return file_location;
}

std::string GetFilePathTestData(const std::string& file_name) {
  std::string file_location = __FILE__;
  std::string current_file = "src/lib/utils.cpp";
  file_location.erase(file_location.end() - current_file.size(),
                      file_location.end());
  file_location += "tests/data/";
  file_location += file_name;
  return file_location;
}

std::string GetFilePathTestClouds(const std::string& file_name) {
  std::string file_location = __FILE__;
  std::string current_file = "src/lib/utils.cpp";
  file_location.erase(file_location.end() - current_file.size(),
                      file_location.end());
  file_location += "tests/template_clouds/";
  file_location += file_name;
  return file_location;
}

std::string GetFilePathTestBags(const std::string& file_name) {
  std::string file_location = __FILE__;
  std::string current_file = "src/lib/utils.cpp";
  file_location.erase(file_location.end() - current_file.size(),
                      file_location.end());
  file_location += "tests/test_bags/";
  file_location += file_name;
  return file_location;
}

void GetScreenResolution(int& horizontal, int& vertical) {
  Display* d = XOpenDisplay(NULL);
  Screen* s = DefaultScreenOfDisplay(d);
  horizontal = s->width;
  vertical = s->height;
}

Eigen::Matrix4d GetT_Robot_Sensor(const CalibrationResults& calibs,
                                  SensorType type, uint8_t sensor_id,
                                  bool& success) {
  success = true;
  for (CalibrationResult calib : calibs) {
    if (calib.type == type && calib.sensor_id == sensor_id) {
      return calib.transform;
    }
  }
  success = false;
  return Eigen::Matrix4d::Identity();
}

// ImgToMat helper
int DepthStrToInt(const std::string depth) {
  if (depth == "8U") {
    return 0;
  } else if (depth == "8S") {
    return 1;
  } else if (depth == "16U") {
    return 2;
  } else if (depth == "16S") {
    return 3;
  } else if (depth == "32S") {
    return 4;
  } else if (depth == "32F") {
    return 5;
  }
  return 6;
}

// ImgToMat helper
int GetCvType(const std::string& encoding) {
  // Check for the most common encodings first
  if (encoding == enc::BGR8) return CV_8UC3;
  if (encoding == enc::MONO8) return CV_8UC1;
  if (encoding == enc::RGB8) return CV_8UC3;
  if (encoding == enc::MONO16) return CV_16UC1;
  if (encoding == enc::BGR16) return CV_16UC3;
  if (encoding == enc::RGB16) return CV_16UC3;
  if (encoding == enc::BGRA8) return CV_8UC4;
  if (encoding == enc::RGBA8) return CV_8UC4;
  if (encoding == enc::BGRA16) return CV_16UC4;
  if (encoding == enc::RGBA16) return CV_16UC4;

  // For bayer, return one-channel
  if (encoding == enc::BAYER_RGGB8) return CV_8UC1;
  if (encoding == enc::BAYER_BGGR8) return CV_8UC1;
  if (encoding == enc::BAYER_GBRG8) return CV_8UC1;
  if (encoding == enc::BAYER_GRBG8) return CV_8UC1;
  if (encoding == enc::BAYER_RGGB16) return CV_16UC1;
  if (encoding == enc::BAYER_BGGR16) return CV_16UC1;
  if (encoding == enc::BAYER_GBRG16) return CV_16UC1;
  if (encoding == enc::BAYER_GRBG16) return CV_16UC1;

  // Miscellaneous
  if (encoding == enc::YUV422) return CV_8UC2;

  // Check all the generic content encodings
  std::cmatch m;

  if (std::regex_match(encoding.c_str(), m,
                       std::regex("(8U|8S|16U|16S|32S|32F|64F)C([0-9]+)"))) {
    return CV_MAKETYPE(DepthStrToInt(m[1].str()), atoi(m[2].str().c_str()));
  }

  if (std::regex_match(encoding.c_str(), m,
                       std::regex("(8U|8S|16U|16S|32S|32F|64F)"))) {
    return CV_MAKETYPE(DepthStrToInt(m[1].str()), 1);
  }

  throw std::runtime_error("Unrecognized image encoding [" + encoding + "]");
}

cv::Mat RosImgToMat(const sensor_msgs::Image& source) {
  int source_type = GetCvType(source.encoding);
  int byte_depth = enc::bitDepth(source.encoding) / 8;
  int num_channels = enc::numChannels(source.encoding);

  if (source.step < source.width * byte_depth * num_channels) {
    std::stringstream ss;
    ss << "Image is wrongly formed: step < width * byte_depth * num_channels  "
          "or  "
       << source.step << " != " << source.width << " * " << byte_depth << " * "
       << num_channels;
    throw std::runtime_error(ss.str());
  }

  if (source.height * source.step != source.data.size()) {
    std::stringstream ss;
    ss << "Image is wrongly formed: height * step != size  or  "
       << source.height << " * " << source.step << " != " << source.data.size();
    throw std::runtime_error(ss.str());
  }

  // If the endianness is the same as locally, share the data
  cv::Mat mat(source.height, source.width, source_type,
              const_cast<uchar*>(&source.data[0]), source.step);
  if ((boost::endian::order::native == boost::endian::order::big &&
       source.is_bigendian) ||
      (boost::endian::order::native == boost::endian::order::little &&
       !source.is_bigendian) ||
      byte_depth == 1)
    return mat;

  // Otherwise, reinterpret the data as bytes and switch the channels
  // accordingly
  mat = cv::Mat(source.height, source.width,
                CV_MAKETYPE(CV_8U, num_channels * byte_depth),
                const_cast<uchar*>(&source.data[0]), source.step);
  cv::Mat mat_swap(source.height, source.width, mat.type());

  std::vector<int> fromTo;
  fromTo.reserve(num_channels * byte_depth);
  for (int i = 0; i < num_channels; ++i)
    for (int j = 0; j < byte_depth; ++j) {
      fromTo.push_back(byte_depth * i + j);
      fromTo.push_back(byte_depth * i + byte_depth - 1 - j);
    }
  cv::mixChannels(std::vector<cv::Mat>(1, mat),
                  std::vector<cv::Mat>(1, mat_swap), fromTo);

  // Interpret mat_swap back as the proper type
  mat_swap.reshape(num_channels);

  return mat_swap;
}

bool HasExtension(const std::string& input, const std::string& extension) {
  // get extension
  std::filesystem::path p(input);
  std::string extension_found = p.extension().string();

  // convert both to lowercase
  std::string extension_search_lowercase = extension;
  std::for_each(extension_search_lowercase.begin(),
                extension_search_lowercase.end(),
                [](char& c) { c = ::tolower(c); });
  std::string extension_found_lowercase = extension_found;
  std::for_each(extension_found_lowercase.begin(),
                extension_found_lowercase.end(),
                [](char& c) { c = ::tolower(c); });

  return extension_found_lowercase == extension_search_lowercase;
}

bool ReadJson(const std::string& filename, nlohmann::json& J,
              JsonReadErrorType& error_type, bool output_error) {
  // check file exists
  if (!std::filesystem::exists(filename)) {
    if (output_error) {
      LOG_ERROR("CheckJson failed - Json file does not exist. Input: %s",
                filename.c_str());
    }
    error_type = JsonReadErrorType::MISSING;
    return false;
  }

  // check for correct file extension
  if (!HasExtension(filename, ".json")) {
    if (output_error) {
      LOG_ERROR("CheckJson failed - Invalid file extension. Input: %s",
                filename.c_str());
    }
    error_type = JsonReadErrorType::FILETYPE;
    return false;
  }

  // load json and check that it's not empty
  std::ifstream file(filename);
  file >> J;

  if (J.empty()) {
    if (output_error) {
      LOG_ERROR("CheckJson failed - Json file is empty. Input: %s",
                filename.c_str());
    }
    error_type = JsonReadErrorType::EMPTY;
    return false;
  }

  error_type = JsonReadErrorType::NONE;
  return true;
}

}} // namespace vicon_calibration::utils
