#!/usr/bin/env sh

TOOLS=../../../../build/tools
LOG_FILE_NAME=log_fc.txt

GLOG_logtostderr=1 $TOOLS/caffe train \
    --solver=solver_fc.prototxt --weights=/data1/pulkitag/caffe_models/caffe_imagenet_train_iter_310000 2>&1 | tee ${LOG_FILE_NAME}

