#!/bin/sh

aws s3 sync ./input/ s3://lightgbm-input/ --delete
echo "Sync Complete (press enter)"
read finish
