#pragma once

// #include <atomic>
// #include <chrono>
#include <cstdint>
#include <fstream>
#include <math.h>
// #include <mutex>
// #include <string>
// #include <thread>

#include <boost/make_shared.hpp>
#include <nlohmann/json.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <beam_utils/pointclouds.h>

static bool stop_visualizer_flag_set_default{false};

namespace vicon_calibration {

using col_handler =
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>;

/**
 * @brief Interactive visualizer class to display point clouds. It also allows
 * you to toggle on/off the first three clouds added to the visualizer. Note: to
 * use:
 * 1. create visualizer instance
 * 2. call AddPointCloudToViewer (either by adding new clouds or replacing existing)
 * 3. call DisplayClouds() to view them in viewer
 * 4. call ClearPointClouds (unless replacing them later)
 */
class Visualizer {
public:
  // struct to contain cloud information
  struct CloudInfo {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    std::string id;
    int point_size;
    bool cloud_on;
    Eigen::Matrix4f T;

    Eigen::Affine3f Affine() {
      Eigen::Affine3f TA;
      TA.matrix() = T.cast<float>();
      return TA;
    }

    CloudInfo(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _cloud,
              const std::string& _id, int _point_size,
              const Eigen::Matrix4f& _T = Eigen::Matrix4f::Identity(),
              bool _cloud_on = true)
        : cloud(_cloud),
          id(_id),
          point_size(_point_size),
          cloud_on(_cloud_on),
          T(_T) {}
  };

  /**
   * @brief Constructor
   * @param display_name display name
   */
  Visualizer(const std::string display_name);

  /**
   * @brief Empty destructor
   */
  ~Visualizer();

  void AddPointCloudToViewer(
      const PointCloudPtr& cloud, const std::string& cloud_name,
      const Eigen::Vector3i& rgb_color, int point_size,
      const Eigen::Matrix4f& T = Eigen::Matrix4f::Identity());

  void AddPointCloudToViewer(
      int cloud_iter, const PointCloudPtr& cloud, const std::string& cloud_name,
      const Eigen::Vector3i& rgb_color, int point_size,
      const Eigen::Matrix4f& T = Eigen::Matrix4f::Identity());

  void ClearPointClouds();

  /**
   * @brief Method to display n clouds, cloud and id vectors must be the same
   * length, ids must be unique
   * @param stop_visualizer_flag_set use this if the client program wants to
   * know if the user selected to stop showing visualization.
   * @return measurement_valid
   */
  bool DisplayClouds(
      bool& stop_visualizer_flag_set = stop_visualizer_flag_set_default);

private:
  // vis thread method in which the visualizer spins
  void Spin();

  // setup keypoint inputs
  void ConfirmMeasurementKeyboardCallback(
      const pcl::visualization::KeyboardEvent& event, void* viewer_void);

  pcl::visualization::PCLVisualizer::Ptr point_cloud_display_;
  std::vector<CloudInfo> clouds_;
  std::string display_name_;
  bool stop_visualizer_flag_{false};
  bool measurement_valid_{true};
  bool close_viewer_{false};
  bool viewer_key_down_{false};
  std::vector<int> background_col_{0,0,0};
};

} // namespace vicon_calibration
