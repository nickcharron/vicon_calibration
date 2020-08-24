#include "vicon_calibration/measurement_extractors/IsolateTargetPoints.h"

#include <math.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <beam_filtering/CropBox.h>

namespace vicon_calibration {

void IsolateTargetPoints::SetConfig(const std::string &config_file) {
  config_file_ = config_file;
}

// TODO: move this to json tools object
void IsolateTargetPoints::LoadConfig() {
  if (config_file_ == "") {
    config_file_ = utils::GetFilePathConfig("IsolateTargetPointsConfig.json");
  }

  nlohmann::json J;
  std::ifstream file(config_file_);
  file >> J;
  crop_scan_ = J.at("crop_scan");
  isolator_size_weight_ = J.at("isolator_size_weight");
  isolator_distance_weight_ = J.at("isolator_distance_weight");
  clustering_multiplier_ = J.at("clustering_multiplier");
  min_cluster_size_ = J.at("min_cluster_size");
  max_cluster_size_ = J.at("max_cluster_size");
  output_cluster_scores_ = J.at("output_cluster_scores");
}

void IsolateTargetPoints::SetTransformEstimate(
    const Eigen::MatrixXd &T_TARGET_LIDAR) {
  T_TARGET_LIDAR_ = T_TARGET_LIDAR;
  transform_estimate_set_ = true;
}

void IsolateTargetPoints::SetScan(const PointCloud::Ptr &scan_in) {
  scan_in_ = scan_in;
}

void IsolateTargetPoints::SetTargetParams(
    const std::shared_ptr<TargetParams> &target_params) {
  target_params_ = target_params;
}

void IsolateTargetPoints::SetLidarParams(
    const std::shared_ptr<LidarParams> &lidar_params) {
  lidar_params_ = lidar_params;
}

PointCloud::Ptr IsolateTargetPoints::GetPoints() {
  LoadConfig();
  if (!CheckInputs()) {
    throw std::runtime_error{"Invalid inputs to IsolateTargetPoints."};
  }
  CropScan();
  ClusterPoints();
  GetTargetCluster();
  return scan_isolated_;
}

std::vector<PointCloud::Ptr> IsolateTargetPoints::GetClusters() {
  std::vector<PointCloud::Ptr> clusters;
  if (cluster_indices_.size() == 0) {
    LOG_ERROR("No target clusters, make sure IsolateTargetPoints::GetPoints() "
              "has been called.");
    return clusters;
  }

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(scan_cropped_);
  for (pcl::PointIndices indices : cluster_indices_) {
    Eigen::Vector4d centroid;
    boost::shared_ptr<pcl::PointIndices> indices_ptr =
        boost::make_shared<pcl::PointIndices>(indices);
    PointCloud::Ptr cluster = boost::make_shared<PointCloud>();
    extract.setIndices(indices_ptr);
    extract.filter(*cluster);
    clusters.push_back(cluster);
  }
  return clusters;
}

void IsolateTargetPoints::ClusterPoints() {
  // get estimated distance to target
  Eigen::Vector3d translation = T_TARGET_LIDAR_.block(0, 3, 3, 1);
  double distance = translation.norm();
  double max_point_distance =
      distance * tan(lidar_params_->max_angular_resolution_deg * DEG_TO_RAD);

  // create search tree
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree =
      boost::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
  tree->setInputCloud(scan_cropped_);

  // perform clustering
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(clustering_multiplier_ * max_point_distance);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(max_cluster_size_);
  ec.setSearchMethod(tree);
  ec.setInputCloud(scan_cropped_);
  ec.extract(cluster_indices_);
}

void IsolateTargetPoints::GetTargetCluster() {
  if (cluster_indices_.size() == 0) {
    LOG_INFO("Euclidiean clustering failed, using cropped scan. Try relax "
             "thresholding parameters");
    scan_isolated_ = scan_cropped_;
  }

  // get centroids of each cluster
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      centroids;
  std::vector<PointCloud::Ptr> clusters;
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(scan_cropped_);
  for (pcl::PointIndices indices : cluster_indices_) {
    Eigen::Vector4d centroid;
    boost::shared_ptr<pcl::PointIndices> indices_ptr =
        boost::make_shared<pcl::PointIndices>(indices);
    PointCloud::Ptr cluster = boost::make_shared<PointCloud>();
    extract.setIndices(indices_ptr);
    extract.filter(*cluster);
    pcl::compute3DCentroid(*cluster, centroid);
    clusters.push_back(cluster);
    centroids.push_back(centroid);
  }

  if (clusters.size() == 1) {
    scan_isolated_ = clusters[0];
  }

  // get centroid of target if not already calculated
  if (target_params_->template_centroid.isZero()) {
    Eigen::Vector4d template_centroid;
    pcl::compute3DCentroid(*target_params_->template_cloud, template_centroid);
    target_params_->template_centroid = template_centroid;
  }

  // transform target centroid to lidar frame
  Eigen::Vector4d centroid_estimated = utils::InvertTransform(T_TARGET_LIDAR_) *
                                       target_params_->template_centroid;

  // calculate template size
  if (target_params_->template_size == 0) {
    target_params_->template_size =
        CalculateMinimalSize(target_params_->template_cloud);
  }

  // iterate through clusters and calculate errors
  std::vector<double> distance_errors;
  std::vector<double> size_errors;
  std::vector<double> distances;
  std::vector<double> sizes;
  for (int i = 0; i < clusters.size(); i++) {
    double centroid_distance = centroids[i].norm();
    double size = CalculateMinimalSize(clusters[i]);
    double distance_error =
        std::abs(centroid_distance - centroid_estimated.hnormalized().norm());
    double size_error = std::abs(size - target_params_->template_size);
    distance_errors.push_back(distance_error);
    size_errors.push_back(size_error);
    distances.push_back(centroid_distance);
    sizes.push_back(size);
  }

  // normalize errors and calculate scores
  std::vector<double> scores;
  double best_score = std::numeric_limits<int>::max();
  int best_index = 0;
  double max_distance_error =
      *std::max_element(distance_errors.begin(), distance_errors.end());
  double max_size_error =
      *std::max_element(size_errors.begin(), size_errors.end());
  double min_distance_error =
      *std::min_element(distance_errors.begin(), distance_errors.end());
  double min_size_error =
      *std::min_element(size_errors.begin(), size_errors.end());
  for (int i = 0; i < clusters.size(); i++) {
    double distance_error_norm = (distance_errors[i] - min_distance_error) /
                                 (max_distance_error - min_distance_error);
    double size_error_norm =
        (size_errors[i] - min_size_error) / (max_size_error - min_size_error);
    double score = isolator_size_weight_ * size_error_norm +
                   isolator_distance_weight_ * distance_error_norm;
    scores.push_back(score);
    if (score < best_score) {
      best_score = score;
      best_index = i;
    }
  }

  if (output_cluster_scores_) {
    std::cout << "Cluster scores:\n"
              << "Template distance = "
              << centroid_estimated.hnormalized().norm() << "\n"
              << "Template size = " << target_params_->template_size << "\n"
              << "Index | score | size | distance \n";
    for (int i = 0; i < clusters.size(); i++) {
      std::cout << i << " | " << scores[i] << " | " << sizes[i] << " | "
                << distances[i] << "\n";
    }
    std::cout << "Top index: " << best_index << "\n";
  }

  scan_isolated_ = clusters.at(best_index);
  return;
}

PointCloud::Ptr IsolateTargetPoints::GetCroppedScan() { return scan_cropped_; }

bool IsolateTargetPoints::CheckInputs() {
  if (scan_in_ == nullptr) {
    LOG_ERROR("Input scan not set.");
    return false;
  } else if (target_params_ == nullptr) {
    LOG_ERROR("Target params not set.");
    return false;
  } else if (lidar_params_ == nullptr) {
    LOG_ERROR("Lidar params not set.");
    return false;
  } else if (!transform_estimate_set_) {
    LOG_ERROR("Transformation estimate not set.");
    return false;
  } else {
    return true;
  }
}

void IsolateTargetPoints::CropScan() {
  scan_cropped_ = boost::make_shared<PointCloud>();
  beam_filtering::CropBox cropper;
  Eigen::Vector3f min_vector, max_vector;
  max_vector = target_params_->crop_scan.cast<float>();
  min_vector = -max_vector;
  Eigen::Affine3f TA_TARGET_LIDAR;
  TA_TARGET_LIDAR.matrix() = T_TARGET_LIDAR_.cast<float>();
  cropper.SetMinVector(min_vector);
  cropper.SetMaxVector(max_vector);
  cropper.SetRemoveOutsidePoints(true);
  cropper.SetTransform(TA_TARGET_LIDAR);
  cropper.Filter(*scan_in_, *scan_cropped_);
}

double IsolateTargetPoints::CalculateMinimalSize(const PointCloud::Ptr &cloud) {
  pcl::PCA<pcl::PointXYZ> pca;
  PointCloud proj;
  pca.setInputCloud(cloud);
  pca.project(*cloud, proj);

  pcl::PointXYZ proj_min;
  pcl::PointXYZ proj_max;
  pcl::getMinMax3D(proj, proj_min, proj_max);

  double dx = proj_max.x - proj_min.x;
  double dy = proj_max.y - proj_min.y;
  double dz = proj_max.z - proj_min.z;

  if (target_params_->is_target_2d) {
    double minimum = std::min(std::min(dx, dy), dz);
    return dx * dy * dz / minimum;
  } else {
    return dx * dy * dz;
  }
}

} // namespace vicon_calibration
