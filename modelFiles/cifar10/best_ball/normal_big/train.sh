#!/usr/bin/env sh 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2//build/tools 
LOG_FILE_NAME=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/cifar10/best_ball/normal_big/logs/round45_lr0.00000049.txt 
 
GLOG_logtostderr=1 $TOOLS/caffe train --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/cifar10/best_ball/normal_big/solver.prototxt  --snapshot=/data1/pulkitag/snapshots/cifar10/best_ball/normal_big/snap_round44_lr0.00000195_iter_22000.solverstate 2>&1 | tee -a ${LOG_FILE_NAME}