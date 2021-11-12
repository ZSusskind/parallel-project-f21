#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sweep_configurations import triangle_counts, triangle_sizes, output_resolutions, cores, raster_modes, hom_enabled

raster_mode_names = ["Naive Raster", "Bounding Box Raster", "Interpolated Raster"]
hom_names = ["No HOM", "HOM"]

in_csv = sys.argv[1]

os.makedirs("visualizations", exist_ok=True)

df = pd.read_csv(in_csv)

# X axis is # cores, Y axis is runtime, three subplots for raster modes, 12 lines for triangle size + resolution
x_data = np.asarray(cores)
for n in triangle_counts:
    series_data = {}
    for h in hom_enabled:
        for r in raster_modes:
            for s in triangle_sizes:
                for o in output_resolutions:
                    data = df[\
                        (df["# Triangles"] == n) &\
                        (df["HOM Enabled"] == h) &\
                        (df["Raster Mode"] == r) &\
                        (df["Avg. Triangle Size"] == s) &\
                        (df["Image Resolution"] == o)\
                    ]
                    data = data.sort_values(by=["CPUs"])
                    runtime_data = data["Runtime (ms)"].tolist()
                    if h not in series_data:
                        series_data[h] = {}
                    if r not in series_data[h]:
                        series_data[h][r] = {}
                    if s not in series_data[h][r]:
                        series_data[h][r][s] = {}
                    series_data[h][r][s][o] = [float(x) if x != "DNF" else float("nan") for x in runtime_data]
    plt.clf()
    fig, ax = plt.subplots(len(hom_enabled), len(raster_modes))
    markers=[".","^","s",]
    for h in hom_enabled:
        for r in raster_modes:
            ax[h][r].set_title(f"{raster_mode_names[r]}; {hom_names[h]}") 
            ax[h][r].set_xlabel("# of CPUs") 
            ax[h][r].set_ylabel("Runtime (ms)") 
            for s_idx, s in enumerate(triangle_sizes):
                for o_idx, o in enumerate(output_resolutions):
                    y_data = np.asarray(series_data[h][r][s][o])
                    x_values = x_data[np.where(~np.isnan(y_data))]
                    y_values = y_data[np.where(~np.isnan(y_data))]
                    color_r = s_idx/(len(triangle_sizes)-1)
                    color_g = 1.0 - (o_idx/(len(output_resolutions)-1))
                    color_b = o_idx/(len(output_resolutions)-1)
                    luminance = 0.2126*color_r + 0.7152*color_g + 0.0722*color_b
                    target_max_luminance = 0.5
                    if luminance > target_max_luminance:
                        color_r *= target_max_luminance/luminance
                        color_g *= target_max_luminance/luminance
                        color_b *= target_max_luminance/luminance
                    ax[h][r].plot(y_data, label=f"{o}x{o}, size={s}", color=(color_r, color_g, color_b), marker=markers[s_idx%len(markers)])
                    ax[h][r].set_xticks(list(range(len(x_data))))
                    ax[h][r].set_xticklabels(x_data.tolist())
                    ax[h][r].set_yscale("log")
                    if (r == len(raster_modes)-1) and (h == len(hom_enabled)-1):
                        ax[h][r].legend(bbox_to_anchor=(1.65, 1.5))
    fig.set_size_inches(24, 12)
    plt.suptitle(f"Swept Runs with {n} Triangles")
    plt.savefig(os.path.join("visualizations", f"plot_{n}.png"), bbox_inches="tight")

    







