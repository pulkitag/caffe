#!/usr/bin/env sh

TOOLS=/home/eecs/pulkitag/Research/codes/codes/projCaffe/caffe-v2-2/build/tools
LOG_FILE_NAME=log.txt

GLOG_logtostderr=1 $TOOLS/caffe train \
    --solver=solver.prototxt 2>&1 | tee ${LOG_FILE_NAME}

