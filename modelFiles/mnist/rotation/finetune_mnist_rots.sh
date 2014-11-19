#!/usr/bin/env sh

TOOLS=../../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe train \
	--solver=mnist_finetune_solver.prototxt 2>&1 \
	--weights=/data1/pulkitag/snapshots/mnist_rotation/mnist_rotation_iter_40000.caffemodel \
	| tee log_finetune.txt 
