#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz256_ntr30_run0/solver_pre-alex_ft-last_mxl-6.prototxt	 --weights=/data1/pulkitag/caffe_models/caffe_imagenet_train_iter_310000	 -gpu 1	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz256_ntr30_run0/log_train_pre-alex_ft-last_mxl-6.txt 
