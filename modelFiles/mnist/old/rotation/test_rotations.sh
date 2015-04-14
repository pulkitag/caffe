#!/usr/bin/env sh

TOOLS=../../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe test \
	--model=mnist_siamese_train_test.prototxt \
	--iterations=100 \
	--gpu=0 \
	--weights=/data1/pulkitag/snapshots/mnist_rotation/mnist_rotation_iter_50000.caffemodel 2>&1 \
	| tee log_test.txt 
