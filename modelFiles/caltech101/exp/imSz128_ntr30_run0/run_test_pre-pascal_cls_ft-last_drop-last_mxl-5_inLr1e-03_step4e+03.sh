#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe test	 --weights=/data1/pulkitag/caltech101/snapshots/imSz128_ntr30_run0/caffenet_pre-pascal_cls_ft-last_drop-last_mxl-5_inLr1e-03_step4e+03_iter_5000.caffemodel	 --model=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz128_ntr30_run0/caffenet_pre-pascal_cls_ft-last_drop-last_mxl-5_inLr1e-03_step4e+03.prototxt 	 --iterations=100	 -gpu 2	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz128_ntr30_run0/log_test_pre-pascal_cls_ft-last_drop-last_mxl-5_inLr1e-03_step4e+03.txt 
