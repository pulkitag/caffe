#!/usr/bin/env sh

TOOLS=../../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe train \
	--solver=mnist_finetune-random_solver.prototxt 2>&1 \
	| tee log_mnist_random.txt 
