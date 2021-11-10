#!/usr/bin/env python3

import sys
import math
import argparse
import numpy as np
from random import random, seed
#from numba import jit, prange # unfortunately, Numba doesn't seem to play nicely with RNG seeds

#@jit(nopython=True)
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
    #assert(math.isclose(np.linalg.norm(np.cross(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0]))/2, target_area))
    
    # Translate the triangle so it is centered at (0, 0, 0)
    centroid = (1/3) * coordinates.sum(axis=0)
    coordinates -= centroid
    #assert(math.isclose(np.linalg.norm(np.cross(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0]))/2, target_area))

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
    coordinates = np.dot(rotation_matrix, coordinates.T).T
    #np.einsum("mn,bn->bm", rotation_matrix, coordinates)
    #assert(math.isclose(np.linalg.norm(np.cross(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0]))/2, target_area))

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
    #assert(math.isclose(np.linalg.norm(np.cross(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0]))/2, target_area))

    # Ensure the coordinates are in "clockwise" order to fix the orientation of the normal
    normal = np.cross(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0])
    if normal[2] > 0:
        coordinates = np.dot(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), coordinates)
        #coordinates[[0, 2, 1]] #np.asarray([coordinates[0], coordinates[2], coordinates[1]])
    #assert(math.isclose(np.linalg.norm(np.cross(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0]))/2, target_area))
    #assert(np.cross(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0])[2] <= 0)

    return coordinates

#@jit(nopython=True)
def make_triangles(num_triangles, average_size, size_range):
    min_size = average_size - (size_range / 2)
    triangles = np.empty((num_triangles, 3, 3), dtype=np.float64)
    for i in range(num_triangles):
        triangles[i] = make_triangle(size_range*random()+min_size, False)
    return triangles

def make_scene(background_triangles, out_fname, rand_seed=None, average_size=0.05, size_range=0.06):
    if rand_seed is not None:
        seed(rand_seed)

    triangles = make_triangles(background_triangles, average_size, size_range)
    triangle_list = [t.flatten().tolist() for t in triangles.reshape(-1, 9)]
    #triangle_list.append([0.15, 0.15, -100, 0.5, 0.85, -100, 0.85, 0.15, -100]) # Occluding body -- no longer used since HOM picks these out automatically
    out_str = "\n".join(",".join(str(v) for v in t) for t in triangle_list)
    with open(out_fname, "w") as f:
        f.write(out_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("background_triangles", type=int, help="Number of triangles to generate in the image background")
    parser.add_argument("--out_fname", "-o", required=True, help="Output filename (standard extension is .tri)")
    parser.add_argument("--seed", type=int, help="Specify random seed")
    parser.add_argument("--average_size", type=float, default=0.05, help="Average size of triangles")
    parser.add_argument("--size_range", type=float, default=0.06, help="Difference in size between smallest and largest random triangles; uniform distribution")
    args = parser.parse_args()

    make_scene(args.background_triangles, args.out_fname, args.seed, args.average_size, args.size_range)

