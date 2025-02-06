#!/bin/bash

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is not installed. Please install FFmpeg and try again."
    exit 1
fi

# Check if the input directory exists
input_dir="/home/banafshe/Desktop/junk/png_to_mp4"
if [ ! -d "$input_dir" ]; then
    echo "Input directory does not exist."
    exit 1
fi

# Check if the output file already exists
output_file="/home/banafshe/Desktop/junk/png_to_mp4/output.mp4"
if [ -f "$output_file" ]; then
    echo "Output file already exists. Please choose a different name."
    exit 1
fi

# Generate the .mp4 file
ffmpeg -framerate 30 -pattern_type glob -i "$input_dir/*.png" -c:v libx264 -pix_fmt yuv420p "$output_file"

echo "Video file generated successfully."

