#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe test	 --weights=/data1/pulkitag/caltech101/snapshots/imSz128_ntr30_nte20_run0/caffenet_pre-random_ft-all_mxl-1_inLr1e-04_iter_10000.caffemodel	 --model=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz128_ntr30_nte20_run0/caffenet_pre-random_ft-all_mxl-1_inLr1e-04.prototxt 	 --iterations=100	 -gpu 0	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz128_ntr30_nte20_run0/log_test_pre-random_ft-all_mxl-1_inLr1e-04.txt 
