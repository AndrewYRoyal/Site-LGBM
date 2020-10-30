#!/bin/sh
echo "Directory:"
read directory

aws s3 sync s3://lightgbm-output/ ./${directory}/ --delete
echo "Sync Complete (press enter)"
read finish
