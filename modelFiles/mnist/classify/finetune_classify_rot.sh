#!/usr/bin/env sh

TOOLS=../../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe train \
	--solver=mnist_finetune_solver.prototxt \
	--weights=/data1/pulkitag/snapshots/mnist/classify/mnist_iter_40000.caffemodel \
	 2>&1 | tee log_finetune.txt 
