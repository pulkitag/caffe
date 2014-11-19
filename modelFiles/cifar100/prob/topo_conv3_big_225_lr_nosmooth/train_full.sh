#!/usr/bin/env sh

TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools
LOG_FILE_NAME=log.txt

#GLOG_logtostderr=1 $TOOLS/caffe train \
#    --solver=cifar10_full_solver.prototxt \
#    --snapshot=/data1/pulkitag/snapshots/cifar10/topo_conv3_big_225_lr_nosmooth/cifar10_full_iter_70000.solverstate \
#	2>&1  | tee -a ${LOG_FILE_NAME}

#GLOG_logtostderr=1 $TOOLS/caffe train \
#    --solver=cifar10_full_solver.prototxt \
#    --snapshot=/data1/pulkitag/snapshots/cifar100/prob/topo_conv3_big_225_lr_nosmooth/cifar100_full_iter_80000.solverstate \
#	2>&1  | tee -a ${LOG_FILE_NAME}

GLOG_logtostderr=1 $TOOLS/caffe train \
    --solver=cifar10_full_solver.prototxt \
    --snapshot=/data1/pulkitag/snapshots/cifar100/prob/topo_conv3_big_225_lr_nosmooth/cifar100_full_iter_100000.solverstate \
	2>&1  | tee -a ${LOG_FILE_NAME}

