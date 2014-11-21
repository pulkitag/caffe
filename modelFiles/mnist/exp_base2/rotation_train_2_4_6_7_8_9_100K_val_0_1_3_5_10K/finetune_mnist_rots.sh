#!/usr/bin/env sh 
 
TOOLS=/home/eecs/pulkitag/Research/codes/codes/projCaffe/caffe-v2-2/build/tools/ 
 
GLOG_logtostderr=1 $TOOLS/caffe train 	 --solver=mnist_finetune_solver.prototxt	 --weights=/data1/pulkitag/snapshots/mnist_rotation/base2/exp_train_2_4_6_7_8_9_100K_val_0_1_3_5_10K/mnist__iter_50000.caffemodel	 2>&1 | tee log_finetune.txt 
