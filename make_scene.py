#!/usr/bin/env python3

import sys
import math
import numpy as np
from random import random

def make_triangle(target_area, is_foreground):
    min_angle = math.radians(30)
    max_angle = math.radians(150)
    angle = min_angle + ((max_angle - min_angle) * random())
    min_leg_ratio = 0.5
    max_leg_ratio = 2.0
    leg_ratio = min_leg_ratio + ((max_leg_ratio - min_leg_ratio) * random())
    area_ratio = 0.5 * leg_ratio * math.sin(angle) # What would the area be if the length of the base was 1?
    base_length = math.sqrt(target_area / area_ratio)

    # Get coordinates for the triangle on the XY plane
    coordinates = np.array([
        [0.0,                                       0.0,                                    0.0],
        [base_length,                               0.0,                                    0.0],
        [base_length*leg_ratio*math.cos(angle),     base_length*leg_ratio*math.sin(angle),  0.0]
    ])
    assert(math.isclose(np.linalg.norm(np.cross(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0]))/2, target_area))
    
    # Translate the triangle so it is centered at (0, 0, 0)
    centroid = (1/3) * coordinates.sum(axis=0)
    coordinates -= centroid
    assert(math.isclose(np.linalg.norm(np.cross(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0]))/2, target_area))

    # Randomly yaw, pitch, and roll the triange
    yaw = random() * math.tau
    pitch = random() * math.tau
    roll = random() * math.tau
    if is_foreground:
        pitch = 0.0
        roll = 0.0
    rotation_matrix = np.array([
        [
            math.cos(yaw)*math.cos(pitch),
            math.cos(yaw)*math.sin(pitch)*math.sin(roll)-math.sin(yaw)*math.cos(roll),
            math.cos(yaw)*math.sin(pitch)*math.cos(roll)+math.sin(yaw)*math.sin(roll)
        ],
        [
            math.sin(yaw)*math.cos(pitch),
            math.sin(yaw)*math.sin(pitch)*math.sin(roll)+math.cos(yaw)*math.cos(roll),
            math.sin(yaw)*math.sin(pitch)*math.cos(roll)-math.cos(yaw)*math.sin(roll)
        ],
        [
            -math.sin(pitch),
            math.cos(pitch)*math.sin(roll),
            math.cos(pitch)*math.cos(roll)
        ]
    ])
    coordinates = np.asarray([np.dot(rotation_matrix, c) for c in coordinates])
    assert(math.isclose(np.linalg.norm(np.cross(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0]))/2, target_area))

    # Move the triangle to a random position in the scene
    min_translate = 0.1
    max_translate = 0.9
    x_translate = min_translate + ((max_translate - min_translate) * random())
    y_translate = min_translate + ((max_translate - min_translate) * random())
    if not is_foreground:
        z_translate = min_translate + ((max_translate - min_translate) * random())
    else:
        z_translate = -100
    coordinates += np.array([x_translate, y_translate, z_translate])
    assert(math.isclose(np.linalg.norm(np.cross(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0]))/2, target_area))

    return coordinates

num_triangles = int(sys.argv[1])
out_fname = sys.argv[2]

triangles = [make_triangle(0.05, False).flatten().tolist() for i in range(num_triangles)]
triangles.append(make_triangle(0.5, True).flatten().tolist()) # Occluding body
out_str = "\n".join(",".join(str(v) for v in t) for t in triangles)
with open(out_fname, "w") as f:
    f.write(out_str)
