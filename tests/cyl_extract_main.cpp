#include "vicon_calibration/LidarCylExtractor.h"

#include <beam_calibration/TfTree.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <tf2_msgs/TFMessage.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <thread>
using namespace std::literals::chrono_literals;

int main(int argc, char *arv[]) {

  // TfTree for storing all static / dynamic transforms
  beam_calibration::TfTree tf_tree;

  // Rosbag for accessing bag data
  std::string bag_name = "test_bags/map_world_01_2019-05-03-21-40-48.bag";
  rosbag::Bag bag;
  bag.open(bag_name, rosbag::bagmode::Read);
  rosbag::View tf_bag_view = {bag, rosbag::TopicQuery("/tf")};

  tf_tree.start_time_ = tf_bag_view.getBeginTime();

  // Iterate over all message instances in our tf bag view
  for (const auto &msg_instance : tf_bag_view) {
    auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
    if (tf_message != nullptr) {
      for (const geometry_msgs::TransformStamped &tf : tf_message->transforms) {
        std::cout << "Adding transform" << std::endl;
        tf_tree.AddTransform(tf);
      }
    }
  }

  vicon_calibration::LidarCylExtractor cyl_extractor;

  pcl::visualization::PCLVisualizer pcl_viewer;
  while (pcl_viewer.wasStopped()) {
    pcl_viewer.spinOnce(100);
    std::this_thread::sleep_for(100ms);
  }
}