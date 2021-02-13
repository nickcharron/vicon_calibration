#include <vicon_calibration/TfTree.h>

#include <fstream>
#include <iostream>

#include <tf2_eigen/tf2_eigen.h>

#include <vicon_calibration/Utils.h>

using json = nlohmann::json;

namespace vicon_calibration {

void TfTree::LoadJSON(const std::string &file_location) {
  LOG_INFO("Loading file: %s", file_location.c_str());

  json J;
  int calibration_counter = 0, value_counter = 0;
  std::string type, date, method, to_frame, from_frame;
  Eigen::Matrix4d T;
  Eigen::Affine3d TA;

  std::ifstream file(file_location);
  file >> J;

  type = J["type"];
  date = J["date"];
  method = J["method"];

  if (type != "extrinsic_calibration") {
    LOG_ERROR("Attempting to create TfTree with invalid json type. Type: %s",
              type.c_str());
    throw std::invalid_argument{
        "Attempting to create TfTree with invalid json type"};
    return;
  }

  SetCalibrationDate(date);

  LOG_INFO("Type: %s", type.c_str());
  LOG_INFO("Date: %s", date.c_str());
  LOG_INFO("Method: %s", method.c_str());

  for (const auto &calibration : J["calibrations"]) {
    calibration_counter++;
    value_counter = 0;
    int i = 0, j = 0;

    to_frame = calibration["to_frame"];
    from_frame = calibration["from_frame"];

    for (const auto &value : calibration["transform"]) {
      value_counter++;
      T(i, j) = value.get<double>();
      if (j == 3) {
        i++;
        j = 0;
      } else {
        j++;
      }
    }
    if (value_counter != 16) {
      LOG_ERROR("Invalid transform matrix in .json file.");
      throw std::invalid_argument{"Invalid transform matrix in .json file."};
      return;
    }
    TA.matrix() = T;
    this->AddTransform(TA, to_frame, from_frame);
  }
  LOG_INFO("Saved %d transforms", calibration_counter);
}

void TfTree::AddTransform(const Eigen::Affine3d &T, const std::string &to_frame,
                          const std::string &from_frame) {
  ros::Time time_stamp = this->start_time;
  geometry_msgs::TransformStamped T_ROS =
      EigenToROS(T, to_frame, from_frame, time_stamp);
  this->SetTransform(T_ROS, to_frame, from_frame, time_stamp, true);
}

void TfTree::AddTransform(const Eigen::Affine3d &T, const std::string &to_frame,
                          const std::string &from_frame,
                          const ros::Time &time_stamp) {
  geometry_msgs::TransformStamped T_ROS =
      EigenToROS(T, to_frame, from_frame, time_stamp);
  this->SetTransform(T_ROS, to_frame, from_frame, time_stamp, false);
}

void TfTree::AddTransform(const geometry_msgs::TransformStamped &T_ROS,
                          const bool &is_static) {
  std::string to_frame = T_ROS.header.frame_id;
  std::string from_frame = T_ROS.child_frame_id;
  ros::Time transform_time = T_ROS.header.stamp;
  this->SetTransform(T_ROS, to_frame, from_frame, transform_time, is_static);
}

Eigen::Affine3d TfTree::GetTransformEigen(const std::string &to_frame,
                                          const std::string &from_frame) {
  geometry_msgs::TransformStamped T_ROS;
  T_ROS = this->LookupTransform(to_frame, from_frame, this->start_time);
  Eigen::Affine3d T = this->ROSToEigen(T_ROS);
  return T;
}

Eigen::Affine3d TfTree::GetTransformEigen(const std::string &to_frame,
                                          const std::string &from_frame,
                                          const ros::Time &lookup_time) {
  geometry_msgs::TransformStamped T_ROS;
  T_ROS = this->LookupTransform(to_frame, from_frame, lookup_time);
  return this->ROSToEigen(T_ROS);
}

geometry_msgs::TransformStamped
TfTree::GetTransformROS(const std::string &to_frame,
                        const std::string &from_frame,
                        const ros::Time &lookup_time) {
  return this->LookupTransform(to_frame, from_frame, lookup_time);
}

geometry_msgs::TransformStamped
TfTree::GetTransformROS(const std::string &to_frame,
                        const std::string &from_frame) {
  return this->LookupTransform(to_frame, from_frame, this->start_time);
}

std::string TfTree::GetCalibrationDate() {
  if (!is_calibration_date_set_) {
    throw std::runtime_error{"cannot retrieve calibration date, value not set"};
    LOG_ERROR("cannot retrieve calibration date, value not set.");
  }
  return calibration_date_;
}

void TfTree::SetCalibrationDate(const std::string &calibration_date) {
  calibration_date_ = calibration_date;
  is_calibration_date_set_ = true;
}

void TfTree::Clear() {
  Tree_.clear();
  frames_.clear();
  calibration_date_ = "";
  is_calibration_date_set_ = false;
}

geometry_msgs::TransformStamped
TfTree::LookupTransform(const std::string &to_frame,
                        const std::string &from_frame,
                        const ros::Time &time_stamp) {
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
TfTree::EigenToROS(const Eigen::Affine3d &T, const std::string &to_frame,
                   const std::string &from_frame, const ros::Time &time_stamp) {
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
TfTree::ROSToEigen(const geometry_msgs::TransformStamped T_ROS) {
  return tf2::transformToEigen(T_ROS);
}

void TfTree::SetTransform(const geometry_msgs::TransformStamped &T_ROS,
                          const std::string &to_frame,
                          const std::string &from_frame,
                          const ros::Time &time_stamp, const bool &is_static) {
  /* ---------------------------------------------------------------------------
  here's the logic:
  if static and exact transform from child to parent exists, output error
  if static and a child already has a parent, add the inverse
  if static and child doesn't already have a parent, add normally
  if dynamic and adding identical child-parent then add normally
  if dynamic and adding child which already has a parent, add inverse
  if both frames have a parent already then output error
  ----------------------------------------------------------------------------*/
  geometry_msgs::TransformStamped T_ROS_ = T_ROS;
  // Static case:
  std::string transform_error;
  if (is_static) {
    // Check case the exact transform exists
    bool transform_exists =
        Tree_.canTransform(to_frame, from_frame, time_stamp, &transform_error);
    if (transform_exists) {
      // Suppress this warning
      // LOG_ERROR("Trying to add a static transform that already exists "
      //               "(to_frame: %s, from frame %s)",
      //               to_frame.c_str(), from_frame.c_str());
      throw std::runtime_error{
          "Cannot add transform. Transform already exists."};
    }

    // Check for case where a child frame already has a parent
    std::string parent;
    bool parent_exists = Tree_._getParent(from_frame, time_stamp, parent);
    if (parent_exists) {
      // Then add inverse
      // LOG_INFO(
      //     "Attemping to add transform from %s to %s, but frame %s already "
      //     "has a parent (%s). Adding inverse of inputted transform.",
      //     from_frame.c_str(), to_frame.c_str(), from_frame.c_str(),
      //     parent.c_str());
      tf2::Transform inverse_transform;
      tf2::fromMsg(T_ROS_.transform, inverse_transform);
      inverse_transform = inverse_transform.inverse();
      T_ROS_.transform = tf2::toMsg(inverse_transform);
      T_ROS_.header.frame_id = from_frame;
      T_ROS_.child_frame_id = to_frame;

      if (!Tree_.setTransform(T_ROS_, "TfTree", is_static)) {
        LOG_ERROR("Cannot add transform from %s to %s", from_frame.c_str(),
                  to_frame.c_str());
        throw std::runtime_error{"Cannot add transform."};
      }
      this->InsertFrame(from_frame, to_frame);
      return;
    } else {
      // Add transform normally
      if (!Tree_.setTransform(T_ROS_, "TfTree", is_static)) {
        LOG_ERROR("Cannot add transform from %s to %s", from_frame.c_str(),
                  to_frame.c_str());
        throw std::runtime_error{"Cannot add transform."};
      }
      this->InsertFrame(to_frame, from_frame);
      return;
    }
  }

  // Dynamic transform case
  std::string parent;
  bool parent_exists = Tree_._getParent(from_frame, time_stamp, parent);
  if (parent_exists && parent == to_frame) {
    // add normally
    if (!Tree_.setTransform(T_ROS_, "TfTree", is_static)) {
      LOG_ERROR("Cannot add transform from %s to %s", from_frame.c_str(),
                to_frame.c_str());
      throw std::runtime_error{"Cannot add transform."};
    }
    this->InsertFrame(to_frame, from_frame);
    return;
  } else if (parent_exists) {
    // add inverse only if "new" child doesn't already have another parent
    parent_exists = Tree_._getParent(to_frame, time_stamp, parent);
    if (parent_exists && parent != from_frame) {
      LOG_ERROR("Cannot add transform from %s to %s", from_frame.c_str(),
                to_frame.c_str());
      throw std::runtime_error{"Cannot add transform."};
    } else {
      // add inverse
      // LOG_INFO("Attemping to add transform from %s to %s, but frame %s
      // already "
      //   "has a parent (%s). Adding inverse of inputted transform.",
      //   from_frame.c_str(), to_frame.c_str(), to_frame.c_str(),
      //   parent.c_str());
      tf2::Transform inverse_transform;
      tf2::fromMsg(T_ROS_.transform, inverse_transform);
      inverse_transform = inverse_transform.inverse();
      T_ROS_.transform = tf2::toMsg(inverse_transform);
      T_ROS_.header.frame_id = from_frame;
      T_ROS_.child_frame_id = to_frame;

      if (!Tree_.setTransform(T_ROS_, "TfTree", is_static)) {
        LOG_ERROR("Cannot add transform from %s to %s", from_frame.c_str(),
                  to_frame.c_str());
        throw std::runtime_error{"Cannot add transform."};
      }
      this->InsertFrame(from_frame, to_frame);
      return;
    }
  } else {
    // add normally
    if (!Tree_.setTransform(T_ROS_, "TfTree", is_static)) {
      LOG_ERROR("Cannot add transform from %s to %s", from_frame.c_str(),
                to_frame.c_str());
      throw std::runtime_error{"Cannot add transform."};
    }
    this->InsertFrame(to_frame, from_frame);
    return;
  }
}

void TfTree::InsertFrame(const std::string &to_frame,
                         const std::string &from_frame) {
  auto it = frames_.find(from_frame);
  if (it == frames_.end()) {
    // from_frame not added yet
    frames_.emplace(from_frame, std::vector<std::string>{to_frame});
  } else {
    for (auto child_frame : it->second) {
      // transform already exists. Return.
      if (child_frame == to_frame)
        return;
    }
    // from_frame already exists in the map, insert to_frame at the back
    frames_[from_frame].push_back(to_frame);
  }
}

} // namespace vicon_calibration
