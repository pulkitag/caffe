#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --weights=/data1/pulkitag/pascal3d/snapshots/pascal3d_imSz128_lbl-uni-az30el10_crp-contPad16_ns4e+04_mb50/caffenet_scratch_unsup_fc6_drop_iter_60000.caffemodel	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz128_ntr30_run0/solver_pre-uniform_az30_el10_drop_60K_ft-last_drop-last_mxl-2_inLr1e-03_step4e+03.prototxt	 -gpu 1	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz128_ntr30_run0/log_train_pre-uniform_az30_el10_drop_60K_ft-last_drop-last_mxl-2_inLr1e-03_step4e+03.txt 
