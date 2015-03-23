#!/usr/bin/env sh

TOOLS=../../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe test \
	--model=mnist_test.prototxt 2>&1 \
	--iterations=100 \
	--gpu=0 \
	--weights=/data1/pulkitag/snapshots/mnist/finetune_rot/mnist_finetune_rot_iter_50000.caffemodel \
	| tee log_test.txt 
