# Parallel Rasterization

Code to accompany "Parallel Rasterization, Z-Buffering, and Occlusion Culling".\
Final project for Vijay Garg's Parallel Algorithms course, Fall 2021.

## Authors:
Umair Shahzad (UShahzad@utexas.edu)\
Zachary Susskind (ZSusskind@utexas.edu)

## Description
This repository contains the code needed to generate test images for our parallel rasterizer, as well as the rasterizer itself.

``make_scene.py`` is a script used for generating random input scenes in vector form; it takes a number of parameters, including the number of trianglges and their average size.\
``render_zcull.cpp`` is the main rasterization / HOM implementation. Run ``make`` to build.\
``render_scene.py`` is a simple test rasterizer which we used to gild the outputs of ``render_zcull``.\
``sweep_configurations.py`` launches a large number of runs to profile the performance of ``render_zcull``. This was used to generate the output file ``sweep_run.csv``.\
Lastly, ``make_visualizations.py`` was used to create the graphs in the paper.
