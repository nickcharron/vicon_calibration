# vicon_calibration
Package to perform multi-sensor calibration using a Vicon system.

### ICP Registration Parameters Used for Lidar Cylinder Extraction
max_corr: Maximum correspondence distance [default: 1]
max_iter: Maximum number of iterations for ICP registration [default: 100]
t_eps: Transformation epsilon (maximum allowable translation squared difference between two consecutive transformations) [default: 1e-8]
fit_eps: Euclidean fitness epsilon (maximum allowed Euclidean error between two consecutive steps in the ICP loop) [default: 1e-2]
