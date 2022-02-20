# vicon_calibration

Package to perform extrinsic calibration of any number of lidar/cameras (with or without overlapping FOV) using a motion capture system (e.g., Vicon System).

See paper: COMING SOON....

## Installing

### Install Dependencies:

The following is a dependencies are required to build this repo:

1. PCL: minimum version 1.11.1
2. gflags: 
3. ROS: tested on kinetic and melodic
4. Opencv: default version installed with ROS should be fine
5. nlohmann_json: minimum version 
6. Ceres: minimum version 1.14.0

For simplicity, we have created an install script for installing these dependencies. To install any missing dependencies listed above, source the dependencies_install.bash and then call upon install functions for each missing dependency. 

For testing, catch2 is used, but testing needs to be enabled using -DCATKIN_ENABLE_TESTING=1. A function has been created in the install script to install catch2.

### Build:

```
git clone https://github.com/nickcharron/vicon_calibration.git
cd vicon_calibration
mkdir build
cd build
cmake ..
make -j8
```

## Running on your own data

There's a few steps you will need to do before running on your own data:

1. Create template cloud: A .pcd pointcloud of the same dimensions of your target is needed. This should be as precise as possible, as your calibration accuracy will be depended on it. These are needed for three reasons: <br />
    (i) For cropping your data down before running the measurment extractors, the template is used for getting dimensions of the target.<br />
    (ii) For the case of non-unique keypoints, the iterative approach will find correspondences using your template cloud.<br />
    (iii) Some measurement extractors rely on the template cloud to perform ICP and get inlier keypoints.
We have created a tool (see src/tools/generate_checkerboard_template.cpp) to generate a template cloud for a checkerboard target.   
    
2. Create target config json file: You will need to create a target config file which has all the details of the target you are using. See CylinderTarget_Example.json and CheckerboardCornersTarget_EXAMPLE.json for examples. We also have tools in src/tools for automatically creating the keypoint details for a checkerboard target or checkerboard target. 

3. Create an intrinsics json file: See the example in the config folder. Make sure you have the proper number of intrinsic values based on the camera model (see camera model documentation)

4. Specify extrinsics: There are two options for specifying the intitial extrinsics estimates. First option is to publish them to /tf or /tf_static. If your bag contains these transforms, then nothing else is needed (just don't specify an extrinsics file in the command line inputs). Otherwise, see the example extrinsics file in the config folder for the proper format. **NOTE:**  Our TfTree class uses tf2::BufferCore under the hood, however, it provies some extra functionality that allows you to specify frames with multiple parents. If this is the case, our TfTree will simply switch the transform order so that BufferCore doesn't contain any child frames with multiple parents. Be careful not to create any loops in your transforms, that will break the tree.

5. Create main config json file: Most of the setting you will need to adjust for your specific dataset are in the main config file. See ViconCalibrationConfigCylinder_EXAMPLE.json and ViconCalibrationConfigCheckerboard_EXAMPLE.json in the config folder. See the details bellow for a description of each parameter.

6. [OPTIONAL] Tune other config files: there are other config files that allow you to customize the implementation. See the details bellow for a description of each parameter.

To run the main executable: 

```
cd /path_to/vicon_calibration
./build/vicon_calbration_main [args] 
```

For information on the required/optional arguments run:

```
cd /path_to/vicon_calibration
./build/vicon_calbration_main --help
```

**IMPORTANT NOTE:** The keypoints in the target config file must be expressed in the same frame as the template cloud points. This coordinate frame must also be consistent with the frame tracked by the motion capture system. See the attached paper which shows how the frame was setup for both our example targets

## Running Examples

We have two example datasets and config files that you can use to get used to the program. Both datasets have camera and lidar data, the first dataset uses the checkerboard target and the second example uses the cylinder target.

These are simulation datasets, so we have ground truth calibrations which are published to /tf in the rosbags. Therefore, you can run the executable with a perfect initial guess, or use the provided "perturbed" extrinsics.

[Download Datasets Here](https://drive.google.com/drive/folders/1YQMN1eqoqLlGv-Sx3SfFnKjootUpO-Mr?usp=sharing)

### Running Checkerboard Target Dataset:

Using ground truth calibrations:

```
cd /path_to/vicon_calibration/
./build/vicon_calibration_main -bag /path_to_dataset/CheckerboardTargetDataset_EXAMPLE.bag -calibration_config ./config/ViconCalibrationConfigCheckerboard_EXAMPLE.json -output_directory . -show_camera_measurements=true -show_lidar_measurements=true

```

Using perturbed calibrations from a json extrinsics file:

```
cd /path_to/vicon_calibration
./build/vicon_calibration_main -bag /path_to_dataset/CheckerboardTargetDataset_EXAMPLE.bag -calibration_config ./config/ViconCalibrationConfigCheckerboard_EXAMPLE.json -initial_calibration ./data/initial_calibration_EXAMPLE.json -output_directory . -show_camera_measurements=true -show_lidar_measurements=true

```

## Config Parameters

This section will briefly go over all parameters in the configuration files.

### ViconCalibrationConfig

todo

### OptimizerConfig

todo

### TargetConfig

todo

### IsolateTargetPointsConfig

todo

### CalibrationVerification

todo
