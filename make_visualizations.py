#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sweep_configurations import triangle_counts, triangle_sizes, output_resolutions, cores, raster_modes

raster_mode_names = ["Naive Raster", "Bounding Box Raster", "Interpolated Raster"]

in_csv = sys.argv[1]

os.makedirs("visualizations", exist_ok=True)

df = pd.read_csv(in_csv)

# X axis is # cores, Y axis is runtime, three subplots for raster modes, 12 lines for triangle size + resolution
x_data = np.asarray(cores)
for n in triangle_counts:
    series_data = {}
    for c in cores:
        for r in raster_modes:
            for s in triangle_sizes:
                for o in output_resolutions:
                    row = df[\
                        (df["# Triangles"] == n) &\
                        (df["CPUs"] == c) &\
                        (df["Raster Mode"] == r) &\
                        (df["Avg. Triangle Size"] == s) &\
                        (df["Image Resolution"] == o)\
                    ]
                    row_data = row["Runtime (ms)"].item()
                    if r not in series_data:
                        series_data[r] = {}
                    if s not in series_data[r]:
                        series_data[r][s] = {}
                    if o not in series_data[r][s]:
                        series_data[r][s][o] = []
                    if row_data != "DNF":
                        series_data[r][s][o].append(float(row_data))
                    else:
                        series_data[r][s][o].append(float("nan"))
    plt.clf()
    fig, ax = plt.subplots(1, len(raster_modes))
    for r in raster_modes:
        ax[r].set_title(raster_mode_names[r]) 
        ax[r].set_xlabel("# of CPUs") 
        ax[r].set_ylabel("Runtime (ms)") 
        for s in triangle_sizes:
            for o in output_resolutions:
                y_data = np.asarray(series_data[r][s][o])
                x_values = x_data[np.where(~np.isnan(y_data))]
                y_values = y_data[np.where(~np.isnan(y_data))]
                ax[r].plot(y_data, label=f"{o}x{o}, size={s}")
                ax[r].set_xticks(list(range(len(x_data))))
                ax[r].set_xticklabels(x_data.tolist())
                ax[r].set_yscale("log")
                if r == len(raster_modes)-1:
                    ax[r].legend(bbox_to_anchor=(1.1, 0.8))
    fig.set_size_inches(20, 5)
    plt.suptitle(f"Swept Runs with {n} Triangles")
    plt.savefig(os.path.join("visualizations", f"plot_{n}.png"), bbox_inches="tight")

    







