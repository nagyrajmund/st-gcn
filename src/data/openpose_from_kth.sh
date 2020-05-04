#!/bin/bash

openpose_dir=$1
dataset_dir=$2

for action_dir in "${dataset_dir}"/*; do
	action=$(basename $action_dir);
	echo "Running analysis for ${action}";
	action_json_dir="output/KTH_Action_Dataset/${action}"; 
	mkdir -m777 -p "$action_json_dir";
	for vidpath in "$action_dir"/*.avi; do
		cd /home/shared/openpose && sudo ./build/examples/openpose/openpose.bin --video $vidpath --write_json $action_json_dir --display 0 --render_pose 0;
	done;
done;