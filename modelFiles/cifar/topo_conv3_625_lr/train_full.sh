#!/usr/bin/env sh

TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools
LOG_FILE_NAME=log.txt

GLOG_logtostderr=1 $TOOLS/caffe train \
    --solver=cifar10_full_solver.prototxt 2>&1 | tee ${LOG_FILE_NAME}

# reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/caffe train \
    --solver=cifar10_full_solver_lr1.prototxt \
    --snapshot=/data1/pulkitag/snapshots/cifar10/topo_conv3_625_lr/cifar10_full_iter_60000.solverstate \
	2>&1  | tee -a ${LOG_FILE_NAME}

# reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/caffe train \
    --solver=cifar10_full_solver_lr2.prototxt \
    --snapshot=/data1/pulkitag/snapshots/cifar10/topo_conv3_625_lr/cifar10_full_iter_65000.solverstate 2>&1 | tee -a ${LOG_FILE_NAME}
