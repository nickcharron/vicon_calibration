#include <vicon_calibration/Visualizer.h>

#include <chrono>
#include <stdio.h>
#include <thread>

#include <vicon_calibration/Utils.h>

namespace vicon_calibration {

Visualizer::Visualizer(const std::string display_name)
    : display_name_(display_name) {
  point_cloud_display_ =
      boost::make_shared<pcl::visualization::PCLVisualizer>(display_name_);
  point_cloud_display_->setBackgroundColor(
      background_col_[0], background_col_[1], background_col_[2]);
  point_cloud_display_->initCameraParameters();
  point_cloud_display_->registerKeyboardCallback(
      &Visualizer::ConfirmMeasurementKeyboardCallback, *this);
}

Visualizer::~Visualizer() {}

void Visualizer::AddPointCloudToViewer(const PointCloudPtr& cloud,
                                       const std::string& cloud_name,
                                       const Eigen::Vector3i& rgb_color,
                                       int point_size,
                                       const Eigen::Matrix4f& T) {
  PointCloudColorPtr cloud_col =
      utils::ColorPointCloud(cloud, rgb_color[0], rgb_color[1], rgb_color[2]);
  clouds_.emplace_back(cloud_col, cloud_name, point_size, T);
}

void Visualizer::AddPointCloudToViewer(int cloud_iter,
                                       const PointCloudPtr& cloud,
                                       const std::string& cloud_name,
                                       const Eigen::Vector3i& rgb_color,
                                       int point_size,
                                       const Eigen::Matrix4f& T) {
  if (cloud_iter >= clouds_.size()) {
    throw std::invalid_argument{"cloud iter greater than number of clouds."};
  }

  PointCloudColorPtr cloud_col =
      utils::ColorPointCloud(cloud, rgb_color[0], rgb_color[1], rgb_color[2]);
  clouds_[cloud_iter] = CloudInfo(cloud_col, cloud_name, point_size, T);
}

void Visualizer::ClearPointClouds() {
  clouds_.clear();
}

bool Visualizer::DisplayClouds(bool& stop_visualizer_flag_set) {
  if (clouds_.size() == 0) {
    LOG_INFO("Cannot display clouds, no clouds added to visualizer.");
    measurement_valid_ = false;
    return measurement_valid_;
  }

  measurement_valid_ = true;

  // add point clouds
  for (uint8_t i = 0; i < clouds_.size(); i++) {
    col_handler rgb(clouds_[i].cloud);
    point_cloud_display_->addPointCloud(clouds_[i].cloud, rgb, clouds_[i].id);
    if (!clouds_[i].T.isIdentity()) {
      point_cloud_display_->addCoordinateSystem(1, clouds_[i].Affine(),
                                                clouds_[i].id + "_frame");
    }
    point_cloud_display_->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clouds_[i].point_size,
        clouds_[i].id);
    point_cloud_display_->resetCamera();
  }
  Spin();

  // clear viewer
  point_cloud_display_->removeAllPointClouds();
  point_cloud_display_->removeAllCoordinateSystems();
  point_cloud_display_->removeAllShapes();

  stop_visualizer_flag_set = stop_visualizer_flag_;
  return measurement_valid_;
}

void Visualizer::Spin() {
  close_viewer_ = false;
  while (!point_cloud_display_->wasStopped() && !close_viewer_) {
    point_cloud_display_->spinOnce(10);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void Visualizer::ConfirmMeasurementKeyboardCallback(
    const pcl::visualization::KeyboardEvent& event, void* viewer_void) {
  // check if key has been down for two consecutive spins
  if (viewer_key_down_ && event.keyDown()) {
    return;
  } else if (viewer_key_down_ && !event.keyDown()) {
    viewer_key_down_ = false;
    return;
  } else if (!viewer_key_down_ && event.keyDown()) {
    viewer_key_down_ = true;
  } else if (!viewer_key_down_ && !event.keyDown()) {
    return;
  }

  if (event.getKeySym() == "y") {
    measurement_valid_ = true;
    close_viewer_ = true;
  } else if (event.getKeySym() == "n") {
    measurement_valid_ = false;
    close_viewer_ = true;
  } else if (event.getKeySym() == "c") {
    close_viewer_ = true;
  } else if (event.getKeySym() == "s") {
    stop_visualizer_flag_ = true;
    close_viewer_ = true;
  } else if (event.getKeySym() == "KP_1") {
    if (clouds_[0].cloud_on) {
      clouds_[0].cloud_on = false;
      point_cloud_display_->removePointCloud(clouds_[0].id);
    } else {
      clouds_[0].cloud_on = true;
      col_handler rgb(clouds_[0].cloud);
      point_cloud_display_->addPointCloud(clouds_[0].cloud, rgb, clouds_[0].id);
    }
  } else if (event.getKeySym() == "KP_2") {
    if (clouds_[1].cloud_on) {
      clouds_[1].cloud_on = false;
      point_cloud_display_->removePointCloud(clouds_[1].id);
    } else {
      clouds_[1].cloud_on = true;
      col_handler rgb(clouds_[1].cloud);
      point_cloud_display_->addPointCloud(clouds_[1].cloud, rgb, clouds_[1].id);
    }
  } else if (event.getKeySym() == "KP_3") {
    if (clouds_[2].cloud_on) {
      clouds_[2].cloud_on = false;
      point_cloud_display_->removePointCloud(clouds_[2].id);
    } else {
      clouds_[2].cloud_on = true;
      col_handler rgb(clouds_[2].cloud);
      point_cloud_display_->addPointCloud(clouds_[2].cloud, rgb, clouds_[2].id);
    }
  }
}

} // namespace vicon_calibration