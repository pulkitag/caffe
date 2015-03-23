#!/usr/bin/env sh

TOOLS=../../../../build/tools
LOG_FILE_NAME=log.txt

GLOG_logtostderr=1 $TOOLS/caffe train \
    --solver=solver.prototxt --weights=/data1/pulkitag/caffe_models/caffe_imagenet_train_iter_310000 2>&1 | tee ${LOG_FILE_NAME}

