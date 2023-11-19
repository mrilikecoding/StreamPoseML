#!/bin/bash

# Source and destination directories
SRC_DIR="example_data/"
DEST_DIR="data/"

# Copy contents recursively without overwriting
cp -Rn $SRC_DIR. $DEST_DIR

echo "Copy completed."
