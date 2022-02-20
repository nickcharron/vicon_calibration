#!/usr/bin/env python
#
# READ ME
##############
# This script generates gazebo models for black and white checkerboards
# on a checkerboard background. This was designed for use with the vicon calibration
# code. The checkerboards are pure black and white with a white checkerboard
# background
#
# Usage
#######
# The following command will generate a new folder and populate it with a Gazebo
# model.
#
# generate_checkerboard_target.py rows columns [square_size_in_meters]
# [checkerboard_size_in_meters]
#
# This folder can be placed anywhere in your Gazebo models path, such as the
# default location: $HOME/.gazebo/models.
################################################################################

import sys
import os
import math

# basic argument checking
# this script is not complicated enough to merit using a system such as argparse
if len(sys.argv) < 3:
    print("usage is: generate_checkerboard_target.py rows columns "
          "[square_size_in_meters] [checkerboard_size_in_meters]")
    exit()

# in meters
DEFAULT_SQUARE_SIZE = 0.0254
DEFAULT_CHECKERBOARD_SIZE = 0.500

# set up shop
rows = int(sys.argv[1])
cols = int(sys.argv[2])
if len(sys.argv) > 3:
    sq_size = float(sys.argv[3])
else:
    sq_size = DEFAULT_SQUARE_SIZE
if len(sys.argv) > 4:
    checkerboard_size = float(sys.argv[4])
else:
    checkerboard_size = DEFAULT_CHECKERBOARD_SIZE
dims = str(cols) + "x" + str(rows)
name = "_".join(["CheckerboardTarget", dims,
                 str(sq_size).replace(".", "_"), str(checkerboard_size).replace(".", "_")])
pretty_name = "CheckerboardTarget {} x {}, sq. size {}m checkerboard size {}m".format(cols, rows, sq_size, checkerboard_size)


# create directory for the model if it doesn't exist.
# If it does, don't overwrite it!
directory = name
if not os.path.exists(directory):
    os.makedirs(directory)
else:
    print("directory {} already exists. Aborting.".format(name))


# parts of the SDF file that don't depend on the size
sdf_preamble = """<?xml version="1.0"?>
<sdf version="1.4">
  <model name="{name}">
  <static>true</static>
  <link name="{name}_body">\n""".format(name=name)

sdf_postamble = """\n    </link>
  </model>
</sdf>\n"""

# generate the SDF code for the checkerboard itself
visual_elements = []
for i in range(cols):
    for j in range(rows):
        # this creates either "1 1 1 1" or "0 0 0 1" depending on if the square
        # should be black or white. It may be possible to save on filespace by
        # using built-in Gazebo materials, but whether or not this is actually
        # an improvement is unclear. For one, I don't know the characteristics
        # of the built-in Gazebo materials - they may have specular reflection,
        # for example, which we don't want.
        rgba = " ".join([str((i+j) % 2)]*3) + " 1"
        y = j*sq_size + sq_size/2
        x = i*sq_size + sq_size/2
        element = """      <visual name="checker_{i}_{j}">
        <pose>{x} {y} 0 0 0 0</pose>
        <geometry>
          <box>
            <size>{size} {size} 0.0002</size>
          </box>
        </geometry>
        <material>
          <ambient>{rgba}</ambient>
          <diffuse>{rgba}</diffuse>
          <specular>{rgba}</specular>
          <emissive>{rgba}</emissive>
        </material>
      </visual>""".format(i=i, j=j, x=x, y=y, rgba=rgba, size=sq_size)
        visual_elements.append(element)

# Create the white backdrop, padding the checkerboard by one square on all sides
rotation = math.pi/4
half_width = (cols * sq_size) / 2
half_height = (rows * sq_size) / 2
x_trans = half_width;
y_trans = half_height;
element = """      <visual name="checker_backdrop_vis">
        <pose>{x_trans} {y_trans} 0 0 0 {rotation}</pose>
        <geometry>
          <box>
            <size>{checkerboard_size} {checkerboard_size} 0.0001</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>1 1 1 1</specular>
          <emissive>1 1 1 1</emissive>
        </material>
      </visual>
      <collision name="checker_backdrop_col">
              <pose>{x_trans} {y_trans} 0 0 0 {rotation}</pose>
              <geometry>
                <box>
                  <size>{checkerboard_size} {checkerboard_size} 0.0001</size>
                </box>
              </geometry>
              <max_contacts>10</max_contacts>
        </collision>""".format(x_trans=x_trans, y_trans=y_trans, checkerboard_size=checkerboard_size, rotation=rotation)
visual_elements.append(element)

# create the final SDF contents
sdf_visuals = "\n".join(visual_elements)
sdf_string = sdf_preamble + sdf_visuals + sdf_postamble

# save the SDF
sdf_path = os.path.join(name, "CheckerboardTarget.sdf")
with open(sdf_path, "w") as f:
    f.write(sdf_string)
print("wrote target to {}".format(sdf_path))


# generate the config file to go with the sdf
config_string = """<?xml version='1.0'?>
<model>
  <name>{pretty_name}</name>
  <version>1.0.0</version>
  <sdf version='1.4'>CheckerboardTarget.sdf</sdf>
  <author>
    <name>generate_checkerboard_target.py</name>
    <email>N/A</email>
  </author>
</model>
""".format(pretty_name=pretty_name)

# save the config file
model_filepath = os.path.join(name, "model.config")
with open(model_filepath, "w") as f:
    f.write(config_string)
print("wrote config file to {}".format(model_filepath))
