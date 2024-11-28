#!/bin/bash

ffmpeg -framerate 30 -i outputImages/image%d.png -c:v libx264 -r 30 output.mp4
