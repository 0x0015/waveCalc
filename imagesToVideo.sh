#!/bin/bash

ffmpeg -framerate 30 -i outputImages/image%d.png -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4
