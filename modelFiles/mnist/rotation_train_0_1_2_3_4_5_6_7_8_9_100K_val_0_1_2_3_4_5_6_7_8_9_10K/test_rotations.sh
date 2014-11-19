#!/usr/bin/env sh 
 
TOOLS=../../../build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe test	 --model=../rotation_train_0_1_2_3_4_5_6_7_8_9_100K_val_0_1_2_3_4_5_6_7_8_9_10K/mnist_siamese_train_test.prototxt	 --iterations=1000	 --gpu=0	 --weights=/data1/pulkitag/snapshots/mnist_rotation/exp_train_0_1_2_3_4_5_6_7_8_9_100K_val_0_1_2_3_4_5_6_7_8_9_10K/mnist__iter_50000.caffemodel	 2>&1 | tee log_test_rot.txt 
