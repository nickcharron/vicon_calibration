#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "vicon_calibration/measurement_extractors/DiamondLidarExtractor.h"
#include "vicon_calibration/TfTree.h"
#include "vicon_calibration/utils.h"
#include "vicon_calibration/JsonTools.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_msgs/TFMessage.h>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace vicon_calibration;

// Global variables for testing
std::string bag_path, template_cloud, template_cloud_rot, target_config;
Eigen::Affine3d T_SCAN_TARGET1_EST, T_SCAN_TARGET2_EST;
Eigen::Affine3d T_SCAN_TARGET1_TRUE, T_SCAN_TARGET2_TRUE;
ros::Time transform_lookup_time;
JsonTools json_loader;
std::shared_ptr<TargetParams> target_params;
std::shared_ptr<LidarParams> lidar_params;
PointCloud::Ptr temp_cloud;
boost::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer;
PointCloud::Ptr sim_cloud;
bool close_viewer{false};
bool show_measurements{true};

using namespace std::literals::chrono_literals;

void FileSetup() {
  bag_path = utils::GetFilePathTestBags("roben_simulation.bag");
  target_config = utils::GetFilePathTestData("DiamondTargetSim.json");
  template_cloud = utils::GetFilePathTestClouds("diamond_target_simulation.pcd");
  if(show_measurements){
      pcl_viewer = boost::make_shared<pcl::visualization::PCLVisualizer>();
  }
}

void LoadSimulatedCloud() {
  sim_cloud = boost::make_shared<PointCloud>();
  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);
  rosbag::TopicQuery query = rosbag::TopicQuery("/m3d/aggregator/cloud");
  rosbag::View cloud_bag_view = {bag, query};
  for (const auto &msg_instance : cloud_bag_view) {
    auto cloud_message = msg_instance.instantiate<sensor_msgs::PointCloud2>();
    if (cloud_message != nullptr) {
      pcl::fromROSMsg(*cloud_message, *sim_cloud);
      transform_lookup_time = cloud_message->header.stamp;
    }
  }
}

void LoadTransforms() {
  TfTree tf_tree;
  std::string calibration_json;
  calibration_json = utils::GetFilePathTestData("roben_extrinsics.json");
  tf_tree.LoadJSON(calibration_json);
  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);
  rosbag::View tf_bag_view = {bag, rosbag::TopicQuery("/tf")};

  tf_tree.start_time = tf_bag_view.getBeginTime();

  // Iterate over all message instances in our tf bag view
  std::cout << "Adding transforms" << std::endl;
  for (const auto &msg_instance : tf_bag_view) {
    auto tf_message = msg_instance.instantiate<tf2_msgs::TFMessage>();
    if (tf_message != nullptr) {
      for (geometry_msgs::TransformStamped tf : tf_message->transforms) {
        tf_tree.AddTransform(tf);
      }
    }
  }
  // Get transforms between targets and lidar and world and targets
  // NOTE: there was an error when making the test bag and the frames are
  // named incorrectly but it still works
  std::string m3d_link_frame = "m3d_link";
  std::string target1_frame = "vicon/cylinder_target3/cylinder_target3";
  std::string target2_frame = "vicon/cylinder_target4/cylinder_target4";

  T_SCAN_TARGET1_TRUE = tf_tree.GetTransformEigen(m3d_link_frame, target1_frame,
                                                  transform_lookup_time);
  T_SCAN_TARGET2_TRUE = tf_tree.GetTransformEigen(m3d_link_frame, target2_frame,
                                                  transform_lookup_time);

  // Apply translation to estimate
  Eigen::Vector3d t_perturb(0.1, 0.1, 0.2);
  T_SCAN_TARGET1_EST = T_SCAN_TARGET1_TRUE;
  T_SCAN_TARGET2_EST = T_SCAN_TARGET2_TRUE;
  T_SCAN_TARGET1_EST.translation() =
      T_SCAN_TARGET1_EST.translation() + t_perturb;
  T_SCAN_TARGET2_EST.translation() =
      T_SCAN_TARGET2_EST.translation() + t_perturb;
}

void LoadTargetParams() {
  target_params = json_loader.LoadTargetParams(target_config);

  // replace template with template from test data
  temp_cloud = boost::make_shared<PointCloud>();
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud, *temp_cloud) ==
      -1) {
    PCL_ERROR("Couldn't read file %s \n", template_cloud.c_str());
  }
  target_params->template_cloud = temp_cloud;
}

void LoadLidarParams() {
  lidar_params = std::make_shared<LidarParams>();
  lidar_params->topic = "null_topic";
  lidar_params->frame = "lidar";
  lidar_params->time_steps = 0;
}

void LoadTargetParamsRotated() {
  target_params = json_loader.LoadTargetParams(target_config);
  PointCloud::Ptr temp_cloud;
  temp_cloud = boost::make_shared<PointCloud>();
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(template_cloud_rot,
                                          *temp_cloud) == -1) {
    PCL_ERROR("Couldn't read file %s \n", template_cloud_rot.c_str());
  }
  target_params->template_cloud = temp_cloud;
}

void ConfirmMeasurementKeyboardCallback(
    const pcl::visualization::KeyboardEvent &event, void *viewer_void) {
  if (event.getKeySym() == "c" && event.keyDown()) {
    close_viewer = true;
  }
}

void AddPointCloudToViewer(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                           std::string cloud_name, Eigen::Affine3d &T) {
  pcl_viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_name);
  pcl_viewer->addCoordinateSystem(1, T.cast<float>(), cloud_name + "frame");
  pcl::PointXYZ point;
  point.x = T.translation()(0);
  point.y = T.translation()(1);
  point.z = T.translation()(2);
  pcl_viewer->addText3D(cloud_name + " ", point, 0.05, 0.05, 0.05);
  pcl_viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

TEST_CASE("Test diamond extractor with empty template cloud && empty scan") {
  FileSetup();
  LoadTransforms();
  LoadSimulatedCloud();
  LoadTargetParams();
  LoadLidarParams();
  std::shared_ptr<TargetParams> invalid_target_params =
      std::make_shared<TargetParams>();
  *invalid_target_params = *target_params;
  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> null_cloud;
  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> empty_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  invalid_target_params->template_cloud = null_cloud;
  std::shared_ptr<LidarExtractor> diamond_extractor;
  diamond_extractor = std::make_shared<DiamondLidarExtractor>();
  diamond_extractor->SetShowMeasurements(show_measurements);
  diamond_extractor->SetLidarParams(lidar_params);
  diamond_extractor->SetTargetParams(invalid_target_params);
  REQUIRE_THROWS(
      diamond_extractor->ProcessMeasurement(T_SCAN_TARGET1_EST.matrix(), sim_cloud));
  invalid_target_params->template_cloud = empty_cloud;
  diamond_extractor->SetTargetParams(invalid_target_params);
  REQUIRE_THROWS(
      diamond_extractor->ProcessMeasurement(T_SCAN_TARGET1_EST.matrix(), sim_cloud));
  diamond_extractor->SetTargetParams(target_params);
  REQUIRE_THROWS(
      diamond_extractor->ProcessMeasurement(T_SCAN_TARGET1_EST.matrix(), null_cloud));
  REQUIRE_THROWS(diamond_extractor->ProcessMeasurement(T_SCAN_TARGET1_EST.matrix(),
                                                 empty_cloud));
}

TEST_CASE("Test extracting diamond with invalid transformation matrix") {
  Eigen::Affine3d TA_INVALID1;
  Eigen::Matrix4d T_INVALID2, T_INVALID3;
  T_INVALID3.setIdentity();
  T_INVALID3(3, 0) = 1;
  std::shared_ptr<LidarExtractor> diamond_extractor;
  diamond_extractor = std::make_shared<DiamondLidarExtractor>();
  diamond_extractor->SetShowMeasurements(show_measurements);
  diamond_extractor->SetLidarParams(lidar_params);
  diamond_extractor->SetTargetParams(target_params);
  REQUIRE_THROWS(
      diamond_extractor->ProcessMeasurement(TA_INVALID1.matrix(), sim_cloud));
  REQUIRE_THROWS(diamond_extractor->ProcessMeasurement(T_INVALID2, sim_cloud));
  REQUIRE_THROWS(diamond_extractor->ProcessMeasurement(T_INVALID3, sim_cloud));
}

TEST_CASE("Test extracting diamond target with and without diverged ICP "
          "registration") {
  // FileSetup();
  // LoadTransforms();
  // LoadSimulatedCloud();
  // LoadTargetParams();
  // LoadLidarParams();
  std::shared_ptr<TargetParams> div_target_params =
      std::make_shared<TargetParams>();
  *div_target_params = *target_params;
  Eigen::Vector3d div_crop1(-0.05, -0.05, -0.05);
  Eigen::Vector3d div_crop2(1, 1, 1);
  Eigen::Vector3d good_crop(0.3, 0.3, 0.3);
  div_target_params->crop_scan = div_crop1;
  std::shared_ptr<LidarExtractor> diamond_extractor =
      std::make_shared<DiamondLidarExtractor>();
  diamond_extractor->SetShowMeasurements(show_measurements);
  diamond_extractor->SetLidarParams(lidar_params);
  diamond_extractor->SetTargetParams(div_target_params);
  REQUIRE_THROWS(diamond_extractor->GetMeasurementValid());
  diamond_extractor->ProcessMeasurement(T_SCAN_TARGET1_TRUE.matrix(), sim_cloud);
  REQUIRE(diamond_extractor->GetMeasurementValid() == false);
  div_target_params->crop_scan = div_crop2;
  diamond_extractor->SetTargetParams(div_target_params);
  diamond_extractor->ProcessMeasurement(T_SCAN_TARGET1_TRUE.matrix(), sim_cloud);
  REQUIRE(diamond_extractor->GetMeasurementValid() == false);
  div_target_params->crop_scan = good_crop;
  diamond_extractor->SetTargetParams(div_target_params);
  diamond_extractor->ProcessMeasurement(T_SCAN_TARGET1_TRUE.matrix(), sim_cloud);
  REQUIRE(diamond_extractor->GetMeasurementValid() == true);
}

/* need to view these results, can't automate the test
TEST_CASE("Test best correspondence estimation") {
  // FileSetup();
  // LoadTransforms();
  // LoadSimulatedCloud();
  // LoadTargetParams();
  // LoadLidarParams();
  Eigen::Affine3d T_identity;
  T_identity.setIdentity();
  std::shared_ptr<LidarExtractor> diamond_extractor;
  diamond_extractor = std::make_shared<DiamondLidarExtractor>();
  std::shared_ptr<TargetParams> target_params2 =
  std::make_shared<TargetParams>(); target_params2 =
  target_params; Eigen::Vector3d div_crop(0.5, 0.2, 0.2);
  target_params2->crop_scan = div_crop;
  diamond_extractor->SetShowMeasurements(true);
  diamond_extractor->SetLidarParams(lidar_params);
  diamond_extractor->SetTargetParams(target_params2);
  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> keypoints1, keypoints2;

  // view keypoints 1
  diamond_extractor->ProcessMeasurement(T_SCAN_TARGET1_TRUE.matrix(), sim_cloud);
  keypoints1 = diamond_extractor->GetMeasurement();
  //pcl_viewer = boost::make_shared<pcl::visualization::PCLVisualizer>();
  AddPointCloudToViewer(keypoints1, "keypoints1", T_identity);
  while (!pcl_viewer->wasStopped() && !close_viewer) {
    pcl_viewer->spinOnce(10);
    pcl_viewer->registerKeyboardCallback(&ConfirmMeasurementKeyboardCallback);
    std::this_thread::sleep_for(10ms);
  }
  pcl_viewer->removeAllPointClouds();
  pcl_viewer->close();
  pcl_viewer->resetStoppedFlag();

  // view keypoints 2
  // can't run these both at the same time
  // Eigen::Vector3d div_crop2(0.7, 0.5, 0.5);
  // target_params2->crop_scan = div_crop2;
  // diamond_extractor->SetTargetParams(target_params2);
  // diamond_extractor->ProcessMeasurement(T_SCAN_TARGET1_EST.matrix(), sim_cloud);
  // keypoints2 = diamond_extractor->GetMeasurement();
  // pcl_viewer = boost::make_shared<pcl::visualization::PCLVisualizer>();
  // AddPointCloudToViewer(keypoints2, "keypoints2", T_identity);
  // while (!pcl_viewer->wasStopped() && !close_viewer) {
  //   pcl_viewer->spinOnce(10);
  // pcl_viewer->registerKeyboardCallback(&ConfirmMeasurementKeyboardCallback);
  //   std::this_thread::sleep_for(10ms);
  // }
  // pcl_viewer->removeAllPointClouds();
  // pcl_viewer->close();
  // pcl_viewer->resetStoppedFlag();
}
*/
