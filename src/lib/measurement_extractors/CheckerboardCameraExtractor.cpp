#include <vicon_calibration/measurement_extractors/CheckerboardCameraExtractor.h>

#include <libcbdetect/boards_from_corners.h>
#include <libcbdetect/find_corners.h>
#include <libcbdetect/plot_boards.h>
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
  params.show_processing = false;
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

  // check that one of the boards has the right number of points
  int num_points = target_params_->keypoints_camera.cols();
  int board_id = -1;
  for (int i = 0; i < boards.size(); i++) {
    if (boards[i].num == num_points) {
      board_id = i;
      break;
    }
  }
  if (board_id == -1) {
    LOG_WARN("LibCBDetect could not detect a checkerboard with the correct "
             "number of points (%d)",
             num_points);
    measurement_valid_ = false;
    return;
  }

  // convert all points in the selected board to pcl and add to results
  keypoints_measured_->points.clear();
  const auto& board = boards.at(board_id);
  for (int i = 1; i < board.idx.size() - 1; ++i) {
    for (int j = 1; j < board.idx[i].size() - 1; ++j) {
      if (board.idx[i][j] < 0) { continue; }
      const auto& p = corners.p[board.idx[i][j]];
      pcl::PointXY pixel(p.x, p.y);
      keypoints_measured_->push_back(pixel);
    }
  }

  if (keypoints_measured_->size() != num_points) {
    LOG_WARN("Invalid number of point detected on board.");
    measurement_valid_ = false;
    return;
  }

  if (show_measurements_) {
    std::vector<cbdetect::Board> boards2{boards.at(board_id)};
    cbdetect::plot_boards(*image_annotated_, corners, boards2, params);
  }
}

} // namespace vicon_calibration
