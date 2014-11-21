#!/usr/bin/env sh

TOOLS=../../..//build/tools
LOG_FILE_NAME=log_resume.txt

GLOG_logtostderr=1 $TOOLS/caffe train \
    --solver=solver.prototxt \
    --snapshot=/data1/pulkitag/snapshots/caffenet_conv7_fc2048_op2_coeff0.1/caffenet_conv7_fc2048_op2_coeff0.1_iter_150000.solverstate 2>&1 | tee ${LOG_FILE_NAME}

