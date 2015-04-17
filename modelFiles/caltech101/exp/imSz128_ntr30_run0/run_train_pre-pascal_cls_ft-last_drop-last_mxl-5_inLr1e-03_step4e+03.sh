#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --weights=/data0/pulkitag/pascal3d/snapshots/pascal3d_imSz128_lbl-uni-az30el10_crp-contPad16_ns4e+04_mb50/caffenet_scratch_sup_noRot_fc6_iter_60000.caffemodel	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz128_ntr30_run0/solver_pre-pascal_cls_ft-last_drop-last_mxl-5_inLr1e-03_step4e+03.prototxt	 -gpu 2	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz128_ntr30_run0/log_train_pre-pascal_cls_ft-last_drop-last_mxl-5_inLr1e-03_step4e+03.txt 
