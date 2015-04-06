#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz128_ntr30_run0/solver_pre-rotObjs_kmedoids30_20_iter60K_ft-last_mxl-5.prototxt	 --weights=/data1/pulkitag/snapshots/keypoints/exprotObjs_lblkmedoids30_20_imSz128/keypoints_siamese_scratch_iter_60000.caffemodel	 -gpu 0	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz128_ntr30_run0/log_train_pre-rotObjs_kmedoids30_20_iter60K_ft-last_mxl-5.txt 
