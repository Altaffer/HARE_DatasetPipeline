#!/bin/bash

# roslaunch nv12_to_rgb8_converter cam_check.launch

# python3 dumb_odom.py
roslaunch nv12_to_rgb8_converter cam_check.launch & python3 dumb_odom.py && fg