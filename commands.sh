#!/bin/bash
# This script containes useful commands


# Extract the images and depths tarballs
tar -xvf images.tar.gz
tar -xvf depths.tar.gz

# download ml-based answer normalizer checkpoint
gdown --folder https://drive.google.com/drive/folders/1UaNqp7T_gEHdxAAvJCPr9MUqE6QaiJlm -O checkpoints
