#!/usr/bin/env python3

import rosbag
import argparse
import cv2
import pathlib 


"""
ISAM database: list of json files
"""


def aggregate():
    """
    Bring high-frequency measurements down to timescale of important 
    low-frequency measurements via aggregation (Total Variation?)
    """
    
    return None


def parse_bag_to_dict(bagname, topics):
    data = {}
    bag = rosbag.Bag(bagname)

    for t in topics:
        data[t] = []

    for topic, msg, t in bag.read_messages(topics):
        data[topic].append(msg)
        

    bag.close()
    
    return data


def main():


    return None


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Sanity chack on GPS and IMU sensors.')
    parser.add_argument('-b', '--bagname', type=str)
    parser.add_argument('-t', '--topics', nargs='+', type=str)
    args = parser.parse_args()

    main(args.bagname)
