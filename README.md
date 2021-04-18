# vicon_calibration

Package to perform extrinsic calibration of any number of lidar/cameras (with or without overlapping FOV) using a motion capture system (e.g., Vicon System).

See paper: COMING SOON....

## Building:

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
    
2. Create target config json file: You will need to create a target config file which has all the details of the target you are using. See CylinderTarget_Example.json and DiamondCornersTarget_EXAMPLE.json for examples. We also have tools in src/tools for automatically creating the keypoint details for a checkerboard target or diamond target. 
2. Create an intrinsics json file: See the example in the config folder. Make sure you have the proper number of intrinsic values based on the camera model (see camera model documentation)
3. Specify extrinsics: There are two options for specifying the intitial extrinsics estimates. First option is to publish them to /tf or /tf_static. If your bag contains these transforms, then nothing else is needed (just don't specify an extrinsics file in the command line inputs). Otherwise, see the example extrinsics file in the config folder for the proper format. **NOTE:**  Our TfTree class uses tf2::BufferCore under the hood, however, it provies some extra functionality that allows you to specify frames with multiple parents. If this is the case, our TfTree will simply switch the transform order so that BufferCore doesn't contain any child frames with multiple parents. Be careful not to create any loops in your transforms, that will break the tree.
4. Create main config json file: Most of the setting you will need to adjust for your specific dataset are in the main config file. See ViconCalibrationConfigCylinder_EXAMPLE.json and ViconCalibrationConfigDiamond_EXAMPLE.json in the config folder. See the details bellow for a description of each parameter.
5. [OPTIONAL] Tune other config files: there are other config files that allow you to customize the implementation. See the details bellow for a description of each parameter.

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

We have two example datasets and config files that you can use to get used to the program. Both datasets have camera and lidar data, the first dataset uses the diamond target and the second example uses the cylinder target.

These are simulation datasets, so we have ground truth calibrations which are published to /tf in the rosbags. Therefore, you can run the executable with a perfect initial guess, or use the provided "perturbed" extrinsics.

### Running Diamond Target Dataset:

Using ground truth calibrations:

```
cd /path_to/vicon_calibration/

./build/vicon_calibration_main -bag ~/datasets/vicon_calibration/simulation/DiamondTargetDataset_EXAMPLE.bag -calibration_config ./config/ViconCalibrationConfigDiamond_EXAMPLE.json -output_directory ~/tmp/ -show_camera_measurements=true -show_lidar_measurements=true

```

Using perturbed calibrations from a json extrinsics file:

```
cd /path_to/vicon_calibration
./build/vicon_calibration_main -bag ~/datasets/vicon_calibration/simulation/DiamondTargetDataset_EXAMPLE.bag -calibration_config ./config/ViconCalibrationConfigDiamond_EXAMPLE.json -initial_calibration ./data/initial_calibration_EXAMPLE.json -output_directory ~/tmp/ -show_camera_measurements=true -show_lidar_measurements=true

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
