#!/usr/bin/env sh 
 
TOOLS=../../../build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe test	 --model=../rotation_train_2_4_6_7_8_9_1000K_val_0_1_3_5_10K/mnist_siamese_train_test.prototxt	 --iterations=1000	 --gpu=0	 --weights=/data1/pulkitag/snapshots/mnist_rotation/exp_train_2_4_6_7_8_9_1000K_val_0_1_3_5_10K/mnist__iter_50000.caffemodel	 2>&1 | tee log_test_rot.txt 
