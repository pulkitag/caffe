#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/exp/los-cls-ind-bn22_mxDiff-7_pose-sigMotion_nrmlz-zScoreScaleSeperate_randcrp_concat-fc6_nTr-1000000/solver_con-conv5_scratch_pad24_imS227_con-conv.prototxt	 -gpu 1	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/exp/los-cls-ind-bn22_mxDiff-7_pose-sigMotion_nrmlz-zScoreScaleSeperate_randcrp_concat-fc6_nTr-1000000/log_train_con-conv5_scratch_pad24_imS227_con-conv.txt 
