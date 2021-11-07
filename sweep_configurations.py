#!/usr/bin/env python3

import os
import re
import sys
from itertools import product
import pandas as pd
import subprocess

from make_scene import make_scene

triangle_counts = [1<<8, 1<<11, 1<<14, 1<<17, 1<<20]
triangle_sizes = [1e-6, 1e-4, 1e-2]
output_resolutions = [256, 512, 1024, 2048]
cores = [1, 2, 4, 8, 16, 32, 64]
raster_modes = [0, 1, 2]

print("Making input configurations")
os.makedirs("inputs", exist_ok=True)
for triangle_count, triangle_size in product(triangle_counts, triangle_sizes):
    fname = os.path.join("inputs", f"input_{triangle_count}_tris_{triangle_size}_avgsize.tri")
    if os.path.exists(fname):
        print(f"Input {fname} already exists, skipping; delete file to force")
        continue
    print(f"Generating input {fname}")
    seed = hash(fname)
    make_scene(triangle_count, fname, seed, triangle_size, triangle_size)

results = []
configurations = list(product(triangle_counts, triangle_sizes, output_resolutions, cores, raster_modes))
runtime_re = re.compile(r"Elapsed time: ([0-9\.]+)ms")
for idx, config in enumerate(configurations):
    triangle_count, triangle_size, output_resolution, core_count, raster_mode = config
    input_fname = os.path.join("inputs", f"input_{triangle_count}_tris_{triangle_size}_avgsize.tri")
    output_fname = "/dev/null"
    cmd = f"./render_zcull {input_fname} {output_fname} {output_resolution} {output_resolution} {core_count} {raster_mode}"
    print(f"Run {idx+1}/{len(configurations)}: {cmd}")
    timed_out = False
    try:
        proc = subprocess.run(
            ["./render_zcull", input_fname, output_fname, str(output_resolution), str(output_resolution), str(core_count), str(raster_mode)],
            timeout=300, check=True, stdout=subprocess.PIPE
        )
    except subprocess.TimeoutExpired:
        timed_out = True
    if timed_out:
        print("Timed out")
        runtime = "DNF"
    else:
        stdout = proc.stdout.decode("ascii")
        match = runtime_re.match(stdout)
        if match is None:
            sys.exit(f"Unexpected subprocess output: \"{stdout}\"")
        runtime = float(match.group(1))
        print(f"Runtime of {runtime} ms")
    results.append(list(config) + [runtime])
df = pd.DataFrame(results, columns=["# Triangles", "Avg. Triangle Size", "Image Resolution", "CPUs", "Raster Mode", "Runtime (ms)"])

import pdb; pdb.set_trace()

