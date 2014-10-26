#!/usr/bin/env sh 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2//build/tools 
LOG_FILE_NAME=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/cifar10/best_ball/normal_big/logs/round1_lr0.00100000.txt 
 
GLOG_logtostderr=1 $TOOLS/caffe train --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/cifar10/best_ball/normal_big/solver.prototxt  2>&1 | tee -a ${LOG_FILE_NAME}