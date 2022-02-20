#include <vicon_calibration/measurement_extractors/CheckerboardCameraExtractor.h>

#include <libcbdetect/boards_from_corners.h>
#include <libcbdetect/find_corners.h>
#include <libcbdetect/plot_corners.h>

#include <vicon_calibration/Utils.h>

namespace vicon_calibration {

CheckerboardCameraExtractor::CheckerboardCameraExtractor(
    CornerDetectorType type)
    : CameraExtractor(), corner_detector_(type) {}

void CheckerboardCameraExtractor::GetKeypoints() {
  CropImage();
  if (corner_detector_ == CornerDetectorType::OPENCV) {
    DetectCornersOpenCV();
  } else {
    DetectCornersLibCBDetect();
  }
}

void CheckerboardCameraExtractor::DetectCornersOpenCV() {
  // find dimensions of checkerboard with [n_dim x m_dim] interior corners
  int n_dim = 0, m_dim = 0;
  double xmax = 0, ymax = 0;
  for (int k = 0; k < target_params_->keypoints_camera.cols(); k++) {
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
  bool checkerboard_found =
      cv::findChessboardCorners(*image_cropped_, cv::Size(n_dim, m_dim),
                                corners, cv::CALIB_CB_EXHAUSTIVE);

  // check if valid
  if (!checkerboard_found) {
    LOG_WARN("OpenCV could not detect a checkerboard of size: n = %d, m = %d",
             n_dim, m_dim);
    measurement_valid_ = false;
    return;
  }

  // convert to pcl point cloud
  keypoints_measured_->clear();
  for (size_t i = 0; i < corners.size(); i++) {
    pcl::PointXY pixel(corners[i].x, corners[i].y);
    keypoints_measured_->push_back(pixel);
  }

  if (show_measurements_) {
    *image_annotated_ = *image_cropped_;
    drawChessboardCorners(*image_annotated_, cv::Size(n_dim, m_dim), corners,
                          checkerboard_found);
  }
}

void CheckerboardCameraExtractor::DetectCornersLibCBDetect() {
  // detect corners
  cbdetect::Corner corners;
  std::vector<cbdetect::Board> boards;
  cbdetect::Params params;
  if (corner_detector_ == CornerDetectorType::SADDLEPOINT) {
    params.corner_type = cbdetect::SaddlePoint;
  } else {
    params.corner_type = cbdetect::MonkeySaddlePoint;
  }
  cbdetect::find_corners(*image_cropped_, corners, params);
  cbdetect::boards_from_corners(*image_cropped_, corners, boards, params);

  // check if valid
  if (boards.empty() || corners.p.empty()) {
    LOG_WARN("LibCBDetect could not detect a checkerboard");
    measurement_valid_ = false;
    return;
  }

  if (boards.size() > 1) {
    LOG_WARN("LibCBDetect detected multiple checkerboards");
  }

  // convert to pcl point cloud
  keypoints_measured_->points.clear();
  for (auto p : corners.p) {
    pcl::PointXY pixel(p.x, p.y);
    keypoints_measured_->push_back(pixel);
  }

  if (show_measurements_) { PlotLibCBDetectCorners(corners); }
}

void CheckerboardCameraExtractor::PlotLibCBDetectCorners(
    const cbdetect::Corner& corners) {
  if (image_annotated_->channels() != 3) {
#if CV_VERSION_MAJOR >= 4
    cv::cvtColor(*image_annotated_, *image_annotated_, cv::COLOR_GRAY2BGR);
#else
    cv::cvtColor(*image_annotated_, *image_annotated_, cv::CV_GRAY2BGR);
#endif
  }
  for (int i = 0; i < corners.p.size(); ++i) {
    cv::line(*image_annotated_, corners.p[i], corners.p[i] + 20 * corners.v1[i],
             cv::Scalar(255, 0, 0), 2);
    cv::line(*image_annotated_, corners.p[i], corners.p[i] + 20 * corners.v2[i],
             cv::Scalar(0, 255, 0), 2);
    if (!corners.v3.empty()) {
      cv::line(*image_annotated_, corners.p[i],
               corners.p[i] + 20 * corners.v3[i], cv::Scalar(0, 0, 255), 2);
    }
    cv::circle(*image_annotated_, corners.p[i], 3, cv::Scalar(0, 0, 255), -1);
    cv::putText(*image_annotated_, std::to_string(i),
                cv::Point2i(corners.p[i].x - 12, corners.p[i].y - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
  }
}

} // namespace vicon_calibration
