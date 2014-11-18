#!/usr/bin/env sh 
 
TOOLS=../../../build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train 
	 --solver=mnist_finetune_solver.prototxt	 --weights=/data1/pulkitag/snapshots/mnist_rotation/exp_train_2_4_6_7_8_9_1000K_val_0_1_3_5_10K/mnist__iter_40000.caffemodel	 2>&1 | tee log_finetune.txt 
