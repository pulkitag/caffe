#!/usr/bin/env sh

TOOLS=../../../../build/tools
LOG_FILE_NAME=log_nodrop_fc6.txt

GLOG_logtostderr=1 $TOOLS/caffe train \
    --solver=solver_nodrop_fc6.prototxt 2>&1 | tee ${LOG_FILE_NAME}

