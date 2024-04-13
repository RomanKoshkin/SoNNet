#!/bin/bash

echo "working directory: `pwd`"

cd ../modules/ && python video_utils.py && \
cd ../assets \
&& \
ffpb \
-pattern_type glob \
-y \
-r 60 \
-s 1920x1080 \
-i "*.png" \
-vcodec libx264 \
-crf 25 \
-pix_fmt yuv420p \
-hide_banner \
../videos/output_$1.mp4 # -loglevel error \
echo "video is ready in `pwd`" 

cd ../scripts && sh purge_pngs_and_data.sh
echo "PNGs and state_dicts have been purged"