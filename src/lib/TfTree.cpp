#include <vicon_calibration/TfTree.h>

#include <fstream>
#include <iostream>

#include <tf2_eigen/tf2_eigen.h>

#include <vicon_calibration/Utils.h>

namespace vicon_calibration {

namespace {
Eigen::Matrix4d VectorToEigenTransform(const std::vector<double>& v) {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T(0, 0) = v[0];
  T(0, 1) = v[1];
  T(0, 2) = v[2];
  T(0, 3) = v[3];
  T(1, 0) = v[4];
  T(1, 1) = v[5];
  T(1, 2) = v[6];
  T(1, 3) = v[7];
  T(2, 0) = v[8];
  T(2, 1) = v[9];
  T(2, 2) = v[10];
  T(2, 3) = v[11];
  return T;
}
} // namespace

ros::Time TfTree::GetStartTime() const {
  return start_time_;
}

ros::Time TfTree::GetEndTime() const {
  return end_time_;
}

void TfTree::LoadJSON(const std::string& file_location) {
  LOG_INFO("Loading file: %s", file_location.c_str());

  int calibration_counter = 0;
  std::string type;
  std::string date;
  std::string method;

  nlohmann::json J;
  if (!utils::ReadJson(file_location, J)) {
    LOG_ERROR("Cannot read json tf tree object");
    throw std::runtime_error{"Cannot read tftree json file."};
  }

  try {
    type = J["type"];
    date = J["date"];
    method = J["method"];
    for (const auto& calibration : J["calibrations"]) {
      calibration_counter++;

      std::string to_frame = calibration["to_frame"];
      std::string from_frame = calibration["from_frame"];
      std::vector<double> T_vec = calibration["transform"];
      Eigen::Matrix4d T = VectorToEigenTransform(T_vec);
      if (!utils::IsTransformationMatrix(T)) {
        LOG_ERROR("Invalid transformation matrix read, not adding transform.");
        continue;
      }
      AddTransform(Eigen::Affine3d(T), to_frame, from_frame);
    }
  } catch (const nlohmann::json::exception& e) {
    LOG_ERROR("Cannot read tftree json: one or more missing params. "
              "Reason: %s",
              e.what());
    throw std::runtime_error{"Invalid json"};
  }

  LOG_INFO("Saved %d transforms", calibration_counter);

  if (type != "extrinsic_calibration") {
    LOG_ERROR("Attempting to create TfTree with invalid json type. Type: %s",
              type.c_str());
    throw std::runtime_error{
        "Attempting to create TfTree with invalid json type"};
  }

  SetCalibrationDate(date);
}

void TfTree::AddTransform(const Eigen::Affine3d& T, const std::string& to_frame,
                          const std::string& from_frame) {
  geometry_msgs::TransformStamped T_ROS =
      EigenToROS(T, to_frame, from_frame, start_time_);
  SetTransform(T_ROS, to_frame, from_frame, start_time_, true);
}

void TfTree::AddTransform(const Eigen::Affine3d& T, const std::string& to_frame,
                          const std::string& from_frame,
                          const ros::Time& time_stamp) {
  geometry_msgs::TransformStamped T_ROS =
      EigenToROS(T, to_frame, from_frame, time_stamp);
  SetTransform(T_ROS, to_frame, from_frame, time_stamp, false);
}

void TfTree::AddTransform(const geometry_msgs::TransformStamped& T_ROS,
                          bool is_static) {
  std::string to_frame = T_ROS.header.frame_id;
  std::string from_frame = T_ROS.child_frame_id;
  ros::Time transform_time = T_ROS.header.stamp;
  SetTransform(T_ROS, to_frame, from_frame, transform_time, is_static);
}

Eigen::Affine3d TfTree::GetTransformEigen(const std::string& to_frame,
                                          const std::string& from_frame) const {
  geometry_msgs::TransformStamped T_ROS;
  T_ROS = LookupTransform(to_frame, from_frame, start_time_);
  Eigen::Affine3d T = ROSToEigen(T_ROS);
  return T;
}

Eigen::Affine3d TfTree::GetTransformEigen(const std::string& to_frame,
                                          const std::string& from_frame,
                                          const ros::Time& lookup_time) const {
  geometry_msgs::TransformStamped T_ROS;
  T_ROS = LookupTransform(to_frame, from_frame, lookup_time);
  return ROSToEigen(T_ROS);
}

geometry_msgs::TransformStamped
    TfTree::GetTransformROS(const std::string& to_frame,
                            const std::string& from_frame,
                            const ros::Time& lookup_time) const {
  return LookupTransform(to_frame, from_frame, lookup_time);
}

geometry_msgs::TransformStamped
    TfTree::GetTransformROS(const std::string& to_frame,
                            const std::string& from_frame) const {
  return LookupTransform(to_frame, from_frame, start_time_);
}

std::string TfTree::GetCalibrationDate() const {
  if (!is_calibration_date_set_) {
    throw std::runtime_error{"cannot retrieve calibration date, value not set"};
  }
  return calibration_date_;
}

void TfTree::SetCalibrationDate(const std::string& calibration_date) {
  calibration_date_ = calibration_date;
  is_calibration_date_set_ = true;
}

void TfTree::Clear() {
  Tree_.clear();
  frames_.clear();
  calibration_date_ = "";
  is_calibration_date_set_ = false;
}

bool TfTree::IsValidFrame(const std::string& frame_id) {
  if (frames_.find(frame_id) == frames_.end()) { return false; }
  return true;
}

geometry_msgs::TransformStamped
    TfTree::LookupTransform(const std::string& to_frame,
                            const std::string& from_frame,
                            const ros::Time& time_stamp) const {
  geometry_msgs::TransformStamped T_ROS;
  std::string transform_error;
  bool can_transform =
      Tree_.canTransform(to_frame, from_frame, time_stamp, &transform_error);

  if (can_transform) {
    T_ROS = Tree_.lookupTransform(to_frame, from_frame, time_stamp);
  } else {
    LOG_ERROR("Cannot look up transform from frame %s to %s. Transform Error "
              "Message: %s",
              from_frame.c_str(), to_frame.c_str(), transform_error.c_str());
    throw std::runtime_error{"Cannot look up transform."};
  }
  return T_ROS;
}

geometry_msgs::TransformStamped
    TfTree::EigenToROS(const Eigen::Affine3d& T, const std::string& to_frame,
                       const std::string& from_frame,
                       const ros::Time& time_stamp) const {
  if (!utils::IsTransformationMatrix(T.matrix())) {
    LOG_ERROR("Invalid transformation matrix input");
    throw std::runtime_error{"Invalid transformation matrix"};
  }
  geometry_msgs::TransformStamped T_ROS = tf2::eigenToTransform(T);
  T_ROS.header.seq = 1;
  T_ROS.header.frame_id = to_frame;
  T_ROS.child_frame_id = from_frame;
  T_ROS.header.stamp = time_stamp;
  return T_ROS;
}

Eigen::Affine3d
    TfTree::ROSToEigen(const geometry_msgs::TransformStamped& T_ROS) const {
  return tf2::transformToEigen(T_ROS);
}

void TfTree::SetTransform(const geometry_msgs::TransformStamped& T_ROS,
                          const std::string& to_frame,
                          const std::string& from_frame,
                          const ros::Time& time_stamp, bool is_static) {
  /* ---------------------------------------------------------------------------
  here's the logic:
  if static and exact transform from child to parent exists, output error
  if static and a child already has a parent, add the inverse
  if static and child doesn't already have a parent, add normally
  if dynamic and adding identical child-parent then add normally
  if dynamic and adding child which already has a parent, add inverse
  if both frames have a parent already then output error
  ----------------------------------------------------------------------------*/

  // first check time range
  if (start_time_ == ros::Time(0)) {
    start_time_ = time_stamp;
    end_time_ = time_stamp;
  } else if (time_stamp < start_time_) {
    start_time_ = time_stamp;
  } else if (time_stamp > end_time_) {
    end_time_ = time_stamp;
  }

  // Static case:
  std::string transform_error;
  if (is_static) {
    // Check case the exact transform exists
    bool transform_exists =
        Tree_.canTransform(to_frame, from_frame, time_stamp, &transform_error);
    if (transform_exists) {
      LOG_ERROR("Trying to add a static transform that already exists "
                "(to_frame: %s, from frame %s)",
                to_frame.c_str(), from_frame.c_str());
      throw std::runtime_error{
          "Cannot add transform. Transform already exists."};
    }

    // Check for case where a child frame already has a parent
    std::string parent;
    bool parent_exists = Tree_._getParent(from_frame, time_stamp, parent);
    if (parent_exists) {
      // Then add inverse
      LOG_INFO("Attemping to add transform from %s to %s, but frame %s already "
               "has a parent (%s). Adding inverse of inputted transform.",
               from_frame.c_str(), to_frame.c_str(), from_frame.c_str(),
               parent.c_str());
      tf2::Transform inverse_transform;
      tf2::fromMsg(T_ROS.transform, inverse_transform);
      inverse_transform = inverse_transform.inverse();
      geometry_msgs::TransformStamped T_ROS_inv = T_ROS;
      T_ROS_inv.transform = tf2::toMsg(inverse_transform);
      T_ROS_inv.header.frame_id = from_frame;
      T_ROS_inv.child_frame_id = to_frame;

      if (!Tree_.setTransform(T_ROS_inv, "TfTree", is_static)) {
        LOG_ERROR("Cannot add transform from %s to %s", from_frame.c_str(),
                  to_frame.c_str());
        throw std::runtime_error{"Cannot add transform."};
      }
      InsertFrame(from_frame, to_frame);
      return;
    } else {
      // Add transform normally
      if (!Tree_.setTransform(T_ROS, "TfTree", is_static)) {
        LOG_ERROR("Cannot add transform from %s to %s", from_frame.c_str(),
                  to_frame.c_str());
        throw std::runtime_error{"Cannot add transform."};
      }
      InsertFrame(to_frame, from_frame);
      return;
    }
  }

  // Dynamic transform case
  std::string parent;
  bool parent_exists = Tree_._getParent(from_frame, time_stamp, parent);
  if (parent_exists && parent == to_frame) {
    // add normally
    if (!Tree_.setTransform(T_ROS, "TfTree", is_static)) {
      LOG_ERROR("Cannot add transform from %s to %s", from_frame.c_str(),
                to_frame.c_str());
      throw std::runtime_error{"Cannot add transform."};
    }
    InsertFrame(to_frame, from_frame);
    return;
  } else if (parent_exists) {
    // add inverse only if "new" child doesn't already have another parent
    parent_exists = Tree_._getParent(to_frame, time_stamp, parent);
    if (parent_exists && parent != from_frame) {
      LOG_ERROR("Cannot add transform from %s to %s", from_frame.c_str(),
                to_frame.c_str());
      throw std::runtime_error{"Cannot add transform."};
    } else {
      tf2::Transform inverse_transform;
      tf2::fromMsg(T_ROS.transform, inverse_transform);
      inverse_transform = inverse_transform.inverse();
      geometry_msgs::TransformStamped T_ROS_inv = T_ROS;
      T_ROS_inv.transform = tf2::toMsg(inverse_transform);
      T_ROS_inv.header.frame_id = from_frame;
      T_ROS_inv.child_frame_id = to_frame;

      if (!Tree_.setTransform(T_ROS_inv, "TfTree", is_static)) {
        LOG_ERROR("Cannot add transform from %s to %s", from_frame.c_str(),
                  to_frame.c_str());
        throw std::runtime_error{"Cannot add transform."};
      }
      InsertFrame(from_frame, to_frame);
      return;
    }
  } else {
    // add normally
    if (!Tree_.setTransform(T_ROS, "TfTree", is_static)) {
      LOG_ERROR("Cannot add transform from %s to %s", from_frame.c_str(),
                to_frame.c_str());
      throw std::runtime_error{"Cannot add transform."};
    }
    InsertFrame(to_frame, from_frame);
    return;
  }
}

void TfTree::InsertFrame(const std::string& to_frame,
                         const std::string& from_frame) {
  auto it = frames_.find(from_frame);
  if (it == frames_.end()) {
    // from_frame not added yet
    frames_.emplace(from_frame, std::vector<std::string>{to_frame});
  } else {
    for (auto child_frame : it->second) {
      // transform already exists. Return.
      if (child_frame == to_frame) return;
    }
    // from_frame already exists in the map, insert to_frame at the back
    frames_[from_frame].push_back(to_frame);
  }
}

} // namespace vicon_calibration