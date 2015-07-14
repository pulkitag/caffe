#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/exp/mxDiff-7_pose-slowness_nrmlz-none_randcrp_concat-fc6_nTr-1000000/solver_con-fc6_scratch_pad24_imS227_ct-margin1.prototxt	 -gpu 1	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/exp/mxDiff-7_pose-slowness_nrmlz-none_randcrp_concat-fc6_nTr-1000000/log_train_con-fc6_scratch_pad24_imS227_ct-margin1.txt 
