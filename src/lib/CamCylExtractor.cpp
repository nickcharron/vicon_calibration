#include "vicon_calibration/CamCylExtractor.h"
#include "vicon_calibration/utils.hpp"

#include <thread>
#include <cmath>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

cv::Mat CamCylExtractor::dst_;
cv::Mat CamCylExtractor::detected_edges_;
cv::Mat CamCylExtractor::srg_gray_;
cv::Mat CamCylExtractor::src_;
int CamCylExtractor::lowThreshold;

CamCylExtractor::CamCylExtractor() { PopulateCylinderPoints(); }

void CamCylExtractor::ConfigureCameraModel(std::string intrinsic_file) {
  camera_model_ = beam_calibration::CameraModel::LoadJSON(intrinsic_file);
}

void CamCylExtractor::SetCylinderDimension(double radius, double height) {
  radius_ = radius;
  height_ = height;

  PopulateCylinderPoints();
}

void CamCylExtractor::SetThreshold(double threshold) {
  threshold_ = threshold;
  PopulateCylinderPoints();
}

std::pair<Eigen::Vector3d, bool> CamCylExtractor::GetMeasurementInfo() {
  if (measurement_complete_) {
    return std::make_pair(measurement_, measurement_valid_);
  } else {
    throw std::runtime_error{"Measurement incomplete. Please run "
                              "ExtractCylinder() before getting the "
                             "measurement information."};
  }
}

void CamCylExtractor::PopulateCylinderPoints() {
  // Clear all points before populating again
  cylinder_points_.clear();

  Eigen::Vector4d point(0, 0, 0, 1);
  //pcl::PointXYZ cloud_point;
  double x_sign, y_sign;
  double r = radius_ + threshold_;

  for (double theta = 0; theta < 2*M_PI; theta += M_PI / 12) {
    point(2) =  r * std::cos(theta) ;

    point(1) =  r * std::sin(theta) ;

    point(0) = -threshold_;
    cylinder_points_.push_back(point);

    point(0) = height_ + threshold_;
    cylinder_points_.push_back(point);
  }
}

void CamCylExtractor::ExtractCylinder(Eigen::Affine3d T_CAMERA_TARGET_EST,
                                      cv::Mat image, int measurement_num) {
  auto min_max_vectors = GetBoundingBox(T_CAMERA_TARGET_EST);
  //image = ColorPixelsOnImage(image);

  auto cropped_image = CropImage(image, min_max_vectors[0], min_max_vectors[1]);

  cv::Mat gray_image;
  /*
  if(cropped_image.depth() == CV_8U) {
    std::cout << "unsigned char image" << std::endl;
  }*/
  cv::cvtColor(cropped_image, gray_image, CV_BGR2GRAY);

  cv::Mat thresholded_image;
  cv::threshold(gray_image, thresholded_image, 50, 255, CV_THRESH_BINARY );//| CV_THRESH_OTSU);

  cv::imshow("thresholded", thresholded_image);

  //cv::namedWindow("Original image " + std::to_string(measurement_num), cv::WINDOW_NORMAL);
  //cv::namedWindow("Cropped image " + std::to_string(measurement_num), cv::WINDOW_NORMAL);

  //cv::resizeWindow("Original image " + std::to_string(measurement_num), image.cols/2, image.rows/2);
  //cv::resizeWindow("Cropped image " + std::to_string(measurement_num), cropped_image.cols/2, cropped_image.rows/2);

  //cv::imshow("Original image " + std::to_string(measurement_num), image);
  //cv::imshow("Cropped image " + std::to_string(measurement_num), cropped_image);
  std::cout << "Close images? [yN]" << std::endl;
  //DetectEdges(cropped_image, measurement_num);

  auto key = cv::waitKey();
  while(key != 121) {
    key = cv::waitKey();
  }
  cv::destroyAllWindows();

  measurement_complete_ = true;
  measurement_valid_ = true;
}

std::vector<Eigen::Vector2d>
CamCylExtractor::GetBoundingBox(Eigen::Affine3d T_CAMERA_TARGET_EST) {
  Eigen::Vector4d transformed_point;
  std::vector<double> u, v;

  projected_pixels_.clear();

  pcl::PointXYZ transformed_pcl_point;
  //pcl::PointXYZ point;
  //PointCloud::Ptr transformed_cloud(new PointCloud);
  //PointCloud::Ptr cloud(new PointCloud);
  for (int i = 0; i < cylinder_points_.size(); i++) {
    transformed_point = T_CAMERA_TARGET_EST.matrix() * cylinder_points_[i];
    /*point.x = cylinder_points_[i](0);
    point.y = cylinder_points_[i](1);
    point.z = cylinder_points_[i](2);
    transformed_pcl_point.x = transformed_point(0);
    transformed_pcl_point.y = transformed_point(1);
    transformed_pcl_point.z = transformed_point(2);
    transformed_cloud->push_back(transformed_pcl_point);
    cloud->push_back(point);*/
    //std::cout << "***********************************************" << std::endl;
    //std::cout << "POINT" << std::endl << transformed_point << std::endl;
    auto pixel = camera_model_->ProjectPoint(transformed_point);
    //std::cout << "PIXEL" << std::endl << pixel << std::endl;

    if (camera_model_->PixelInImage(pixel)) {
      u.push_back(pixel(0));
      v.push_back(pixel(1));
      projected_pixels_.push_back(pixel);
    }
  }
/*
  pcl::visualization::PCLVisualizer pcl_viewer("Cloud viewer");
  pcl_viewer.setBackgroundColor(0, 0, 0);
  pcl_viewer.addCoordinateSystem(1.0);
  pcl_viewer.initCameraParameters();
  pcl_viewer.addPointCloud<pcl::PointXYZ>(cloud, "temp cloud");
  pcl_viewer.addPointCloud<pcl::PointXYZ>(transformed_cloud, "transformed_cloud cloud");
  pcl_viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "temp cloud");
  pcl_viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "transformed_cloud cloud");
  while (!pcl_viewer.wasStopped()) {
    pcl_viewer.spinOnce(10);
    std::this_thread::sleep_for(10ms);
  }
*/
  if(u.empty() || v.empty()) {
    throw std::runtime_error{"Can't find minimum and maximum pixels on the image"};
  }
  const auto min_max_u = std::minmax_element(u.begin(), u.end());
  const auto min_max_v = std::minmax_element(v.begin(), v.end());

  Eigen::Vector2d min_vec(*min_max_u.first, *min_max_v.first);
  Eigen::Vector2d max_vec(*min_max_u.second, *min_max_v.second);

  return std::vector<Eigen::Vector2d>{min_vec, max_vec};
}

cv::Mat CamCylExtractor::CropImage(cv::Mat image, Eigen::Vector2d min_vector,
                                   Eigen::Vector2d max_vector) {
  std::cout << "Min vec: " << std::endl
            << min_vector << std::endl
            << "Max vec: " << std::endl
            << max_vector << std::endl;
  double width = max_vector(0) - min_vector(0);
  double height = max_vector(1) - min_vector(1);

  if (!camera_model_->PixelInImage(min_vector) ||
      !camera_model_->PixelInImage(max_vector)) {
    return image;
  }

  cv::Rect region_of_interest(min_vector(0), min_vector(1), width, height);
  cv::Mat cropped_image = image(region_of_interest);

  return cropped_image;
}

cv::Mat CamCylExtractor::ColorPixelsOnImage(cv::Mat &img) {
  for(auto pixel : projected_pixels_) {
    if(camera_model_->PixelInImage(pixel)) {
      cv::circle(img, cv::Point(pixel(0), pixel(1)),
                 2, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);
    }
  }
  return img;
}

void CamCylExtractor::CannyThreshold(int, void*) {
  /// Reduce noise with a kernel 3x3
  cv::blur(srg_gray_, detected_edges_, cv::Size(3,3) );

  /// Canny detector
  cv::Canny(detected_edges_, detected_edges_, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst_ = cv::Scalar::all(0);

  src_.copyTo(dst_, detected_edges_);
  imshow( window_name, dst_);
}

void CamCylExtractor::DetectEdges(cv::Mat &img, int measurement_num) {
  src_ = img;
  /// Load an image
  if (!src_.data) {
    return;
  }

  /// Create a matrix of the same type and size as src (for dst)
  dst_.create(src_.size(), src_.type());

  /// Convert the image to grayscale
  cv::cvtColor(src_, srg_gray_, CV_BGR2GRAY);

  window_name = "Edge detected image " + std::to_string(measurement_num);

  /// Create a window
  cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  cv::resizeWindow(window_name, img.cols/2, img.rows/2);

  /// Create a Trackbar for user to enter threshold
  cv::createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold,
                 CamCylExtractor::CannyThreshold);

  /// Show the image
  CannyThreshold(0, 0);

  /// Wait until user exit program by pressing a key
  cv::waitKey(0);
}

} // end namespace vicon_calibration
