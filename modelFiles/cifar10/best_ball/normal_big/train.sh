#!/usr/bin/env sh 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2//build/tools 
LOG_FILE_NAME=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/cifar10/best_ball/normal_big/logs/round14_lr0.00006250.txt 
 
GLOG_logtostderr=1 $TOOLS/caffe train --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/cifar10/best_ball/normal_big/solver.prototxt  --snapshot=/data1/pulkitag/snapshots/cifar10/best_ball/normal_big/snap_round13_lr0.00001563_iter_48000.solverstate 2>&1 | tee -a ${LOG_FILE_NAME}