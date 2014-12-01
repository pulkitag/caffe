#!/usr/bin/env sh

TOOLS=../../../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe train \
	 --solver=sun_solver.prototxt \
	 --weights=/data1/pulkitag/snapshots/keypoints/keypoints_siamese_iter_70000.caffemodel \
	 2>&1 | tee log.txt 

