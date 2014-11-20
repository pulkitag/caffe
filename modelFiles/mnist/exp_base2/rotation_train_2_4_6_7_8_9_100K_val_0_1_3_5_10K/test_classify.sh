#!/usr/bin/env sh 
 
TOOLS=/home/eecs/pulkitag/Research/codes/codes/projCaffe/caffe-v2-2/build/tools/ 
 
GLOG_logtostderr=1 $TOOLS/caffe test	 --model=../exp_base2/rotation_train_2_4_6_7_8_9_100K_val_0_1_3_5_10K/mnist_train_test.prototxt	 --iterations=1000	 --gpu=0	 --weights=/data1/pulkitag/snapshots/mnist/finetune_all_rot/base2/exp_train_2_4_6_7_8_9_100K_val_0_1_3_5_10K/mnist_finetune_all_rot_iter_50000.caffemodel	 2>&1 | tee log_test.txt 
