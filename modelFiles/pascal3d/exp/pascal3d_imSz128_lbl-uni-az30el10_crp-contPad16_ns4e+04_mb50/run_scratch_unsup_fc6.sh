#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/pascal3d/exp/pascal3d_imSz128_lbl-uni-az30el10_crp-contPad16_ns4e+04_mb50/solver_scratch_unsup_fc6.prototxt	 -gpu 1	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/pascal3d/exp/pascal3d_imSz128_lbl-uni-az30el10_crp-contPad16_ns4e+04_mb50/log_scratch_unsup_fc6.txt 
