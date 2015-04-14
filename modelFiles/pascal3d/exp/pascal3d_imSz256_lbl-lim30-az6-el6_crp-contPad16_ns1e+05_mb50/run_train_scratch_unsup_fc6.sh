#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/pascal3d/exp/pascal3d_imSz256_lbl-lim30-az6-el6_crp-contPad16_ns1e+05_mb50/solver_scratch_unsup_fc6.prototxt	 -gpu 0	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/pascal3d/exp/pascal3d_imSz256_lbl-lim30-az6-el6_crp-contPad16_ns1e+05_mb50/log_train_scratch_unsup_fc6.txt 
