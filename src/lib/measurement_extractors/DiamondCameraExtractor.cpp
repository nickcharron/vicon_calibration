#include <vicon_calibration/measurement_extractors/DiamondCameraExtractor.h>

#include <vicon_calibration/Utils.h>

namespace vicon_calibration {

void DiamondCameraExtractor::GetKeypoints() {
  this->CropImage();

  // find dimensions of checkerboard with [n_dim x m_dim] interior corners
  int n_dim = 0, m_dim = 0;
  double xmax = 0, ymax = 0;
  for (int k = 0; k < target_params_->keypoints_camera.cols(); k++){
    Eigen::Vector3d keypoint = target_params_->keypoints_camera.col(k);
    if (keypoint[0] > xmax) {
      xmax = keypoint[0];
      m_dim++;
    }
    if (keypoint[1] > ymax) {
      ymax = keypoint[1];
      n_dim++;
    }
  }

  // Find checkerboard corners
  std::vector<cv::Point2f> corners;
  bool checkerboard_found = cv::findChessboardCorners(
      *image_cropped_, cv::Size(n_dim, m_dim), corners, cv::CALIB_CB_FAST_CHECK);

  // check if valid
  if (!checkerboard_found) {
    LOG_INFO("OpenCV could not detect a checkerboard of size: n = %d, m = %d",
             n_dim, m_dim);
    measurement_valid_ = false;
    return;
  }

  // convert to pcl point cloud
  keypoints_measured_->points.clear();
  for (uint32_t i = 0; i < corners.size(); i++) {
    pcl::PointXY pixel;
    pixel.x = corners[i].x;
    pixel.y = corners[i].y;
    keypoints_measured_->points.push_back(pixel);
  }

  if (show_measurements_) {
    *image_annotated_ = *image_cropped_;
    drawChessboardCorners(*image_annotated_, cv::Size(n_dim, m_dim), corners,
                          checkerboard_found);
  }
}

} // namespace vicon_calibration
