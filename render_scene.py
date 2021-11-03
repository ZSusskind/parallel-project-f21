#!/usr/bin/env python3

import sys
import numpy as np
from numba import jit, prange
from matplotlib import pyplot as plt

resolution = 1000

# Juan Pineda - "A Parallel Algorithm for Polygon Rasterization" - 1988
@jit(nopython=True)
def edgeFunction(a, b, c):
    return ((c[0]-a[0])*(b[1]-a[1]) - (c[1]-a[1])*(b[0]-a[0]))

@jit(nopython=True)
def rasterize(tri, zbuf):
    # Precompute some constants for depth testing
    z0 = tri[0,2]
    z0_z1_delta = tri[1,2] - z0
    z0_z2_delta = tri[2,2] - z0

    # Scale the triangle to raster space
    raster_tri = (tri * resolution)[:,0:2]
    
    # Get the area of the parallelogram defined by edges (v0, v1) and (v0, v2)
    #  This is twice the area of the raster triangle
    twice_area = edgeFunction(raster_tri[0], raster_tri[1], raster_tri[2])

    # Get a 2D bounding box for the raster triangle to reduce the set of pixels to test
    raster_bbox = np.array([
        [raster_tri[:,0].min(), raster_tri[:,1].min()],
        [raster_tri[:,0].max(), raster_tri[:,1].max()]
    ])
    raster_bbox = np.maximum(np.minimum(raster_bbox, resolution-1), 0)
    raster_bbox = raster_bbox.astype(np.uint32)

    # Test membership for each pixel in the image
    for x in prange(raster_bbox[0,0], raster_bbox[1,0]+1):
        for y in prange(raster_bbox[0,1], raster_bbox[1,1]+1):
            point = np.array([x,y])
            w0 = edgeFunction(raster_tri[1], raster_tri[2], point)
            w1 = edgeFunction(raster_tri[2], raster_tri[0], point)
            w2 = edgeFunction(raster_tri[0], raster_tri[1], point)
            if (w0 >= 0) and (w1 >= 0) and (w2 >= 0): # Point is interior to the triangle
                # The normalized values of w0, w1, and w2 sum to 1,
                #  so we only have to explicitly compute two of them
                norm_w1 = w1 / twice_area
                norm_w2 = w2 / twice_area
                depth = z0 + norm_w1*z0_z1_delta + norm_w2*z0_z2_delta
                if depth < zbuf[y,x]:
                    zbuf[y,x] = depth

in_fname = sys.argv[1]
out_fname = sys.argv[2]

triangles = []
with open(in_fname, "r") as f:
    for line in f:
        triangles.append(np.asarray([float(x) for x in line.split(",")]).reshape(3, 3))

zbuf = np.full((resolution, resolution), float("inf"), dtype=np.float32)

print("Rasterizing image")
for tri in triangles[:-1]:
    rasterize(tri, zbuf)
rasterize(triangles[-1], zbuf)

print("Shading image")
min_depth = zbuf[zbuf>-100].min()
max_depth = zbuf[zbuf<float("inf")].max()
scaled = (zbuf-min_depth) / (max_depth-min_depth)
shaded = np.empty((resolution, resolution, 3), dtype=np.uint8)
shaded = np.stack([(255*(1-scaled)).astype(np.uint8), np.zeros(scaled.shape, dtype=np.uint8), (255*scaled).astype(np.uint8)], axis=-1)
shaded[zbuf == float("inf")] = (255,255,255)
shaded[zbuf <= -100] = (127, 127, 127)
plt.imshow(shaded, interpolation='nearest')
plt.savefig(out_fname)

