#include "vicon_calibration/measurement_extractors/IsolateTargetPoints.h"

#include <math.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <beam_filtering/CropBox.h>

namespace vicon_calibration {

void IsolateTargetPoints::SetConfig(const std::string& config_file) {
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
    const Eigen::MatrixXd& T_TARGET_LIDAR) {
  T_TARGET_LIDAR_ = T_TARGET_LIDAR;
  transform_estimate_set_ = true;
}

void IsolateTargetPoints::SetScan(const PointCloud::Ptr& scan_in) {
  scan_in_ = scan_in;
}

void IsolateTargetPoints::SetTargetParams(
    const std::shared_ptr<TargetParams>& target_params) {
  target_params_ = target_params;
}

void IsolateTargetPoints::SetLidarParams(
    const std::shared_ptr<LidarParams>& lidar_params) {
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
      distance *
      tan(utils::DegToRad(lidar_params_->max_angular_resolution_deg));

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
    return;
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

  if (clusters.size() == 1) { scan_isolated_ = clusters[0]; }

  // get centroid of target if not already calculated
  if (target_params_->template_centroid.isZero()) {
    Eigen::Vector4d template_centroid;
    pcl::compute3DCentroid(*target_params_->template_cloud, template_centroid);
    target_params_->template_centroid = template_centroid;
  }

  // transform target centroid to lidar frame
  Eigen::Vector3d centroid_estimated =
      (utils::InvertTransform(T_TARGET_LIDAR_) *
       target_params_->template_centroid)
          .hnormalized();

  // calculate template dimensions in order of largest to smallest
  if (target_params_->template_dimensions.isZero()) {
    target_params_->template_dimensions =
        CalculateMinimalDimensions(target_params_->template_cloud);
  }

  // iterate through clusters and calculate errors
  std::vector<double> distance_errors;
  std::vector<double> distances;
  std::vector<Eigen::Vector3d, AlignVec3d> dimension_errors;
  std::vector<Eigen::Vector3d, AlignVec3d> dimensions;
  for (int i = 0; i < clusters.size(); i++) {
    // calculate centroid distances and their errors
    double centroid_distance = centroids[i].norm();
    double distance_error =
        std::abs(centroid_distance - centroid_estimated.norm());
    distance_errors.push_back(distance_error);
    distances.push_back(centroid_distance);

    // calculate dimensions and their errors
    Eigen::Vector3d dims = CalculateMinimalDimensions(clusters[i]);
    Eigen::Vector3d dims_error = Eigen::Vector3d(
        (dims - target_params_->template_dimensions).array().abs());
    dimension_errors.push_back(dims_error);
    dimensions.push_back(dims);
  }

  // calculate scores (normalize by dividing by the true value)
  std::vector<double> scores;
  double best_score = std::numeric_limits<int>::max();
  int best_index = 0;

  for (int i = 0; i < clusters.size(); i++) {
    double distance_error_norm =
        std::abs(distance_errors[i] / centroid_estimated.norm());
    double dim1_norm = std::abs(dimension_errors[i][0] /
                                target_params_->template_dimensions[0]);
    double dim2_norm = std::abs(dimension_errors[i][1] /
                                target_params_->template_dimensions[1]);
    double dim3_norm = std::abs(dimension_errors[i][2] /
                                target_params_->template_dimensions[2]);

    double score;
    if (target_params_->is_target_2d) {
      score = isolator_size_weight_ / 2 * dim1_norm +
              isolator_size_weight_ / 2 * dim2_norm +
              isolator_distance_weight_ * distance_error_norm;
    } else {
      score = isolator_size_weight_ / 3 * dim1_norm +
              isolator_size_weight_ / 3 * dim2_norm +
              isolator_size_weight_ / 3 * dim3_norm +
              isolator_distance_weight_ * distance_error_norm;
    }
    scores.push_back(score);

    if (score < best_score) {
      best_score = score;
      best_index = i;
    }
  }

  if (output_cluster_scores_) {
    std::cout << "Cluster scores:\n"
              << "Template distance = " << std::setprecision(4)
              << centroid_estimated.norm() << "\n"
              << "Template dims = [" << std::setprecision(4)
              << target_params_->template_dimensions[0] << ", "
              << std::setprecision(4) << target_params_->template_dimensions[1]
              << ", " << std::setprecision(4)
              << target_params_->template_dimensions[2] << "]\n"
              << "Index | score | distance | dim1 | dim2 | dim3 |\n";
    for (int i = 0; i < clusters.size(); i++) {
      std::cout << i << " | " << std::setprecision(4) << scores[i] << " | "
                << std::setprecision(4) << distances[i] << " | "
                << std::setprecision(4) << dimensions[i][0] << " | "
                << std::setprecision(4) << dimensions[i][1] << " | "
                << std::setprecision(4) << dimensions[i][2] << "\n";
    }
    std::cout << "Top index: " << best_index << "\n";
  }

  scan_isolated_ = clusters.at(best_index);
  return;
}

PointCloud::Ptr IsolateTargetPoints::GetCroppedScan() {
  return scan_cropped_;
}

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
  Eigen::Vector3f min_vector;
  min_vector << target_params_->crop_scan[0], target_params_->crop_scan[2],
      target_params_->crop_scan[4];
  Eigen::Vector3f max_vector;
  max_vector << target_params_->crop_scan[1], target_params_->crop_scan[3],
      target_params_->crop_scan[5];
  Eigen::Affine3f TA_TARGET_LIDAR;
  TA_TARGET_LIDAR.matrix() = T_TARGET_LIDAR_.cast<float>();
  cropper.SetMinVector(min_vector);
  cropper.SetMaxVector(max_vector);
  cropper.SetRemoveOutsidePoints(true);
  cropper.SetTransform(TA_TARGET_LIDAR);
  cropper.Filter(*scan_in_, *scan_cropped_);
}

Eigen::Vector3d IsolateTargetPoints::CalculateMinimalDimensions(
    const PointCloud::Ptr& cloud) {
  pcl::PCA<pcl::PointXYZ> pca;
  PointCloud proj;
  pca.setInputCloud(cloud);
  pca.project(*cloud, proj);

  pcl::PointXYZ proj_min;
  pcl::PointXYZ proj_max;
  pcl::getMinMax3D(proj, proj_min, proj_max);

  std::vector<double> dimensions{proj_max.x - proj_min.x,
                                 proj_max.y - proj_min.y,
                                 proj_max.z - proj_min.z};
  std::sort(dimensions.begin(), dimensions.begin() + dimensions.size());
  std::reverse(dimensions.begin(), dimensions.begin() + dimensions.size());
  return Eigen::Vector3d(dimensions[0], dimensions[1], dimensions[2]);
}

} // namespace vicon_calibration
