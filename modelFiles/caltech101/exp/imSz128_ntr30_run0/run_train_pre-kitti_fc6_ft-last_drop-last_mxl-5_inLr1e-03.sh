#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz128_ntr30_run0/solver_pre-kitti_fc6_ft-last_drop-last_mxl-5_inLr1e-03.prototxt	 --weights=/data1/pulkitag/projRotate/snapshots/kitti/los-cls-ind-bn22_mxDiff-7_pose-sigMotion_nrmlz-zScoreScaleSeperate_randcrp_concat-fc6_nTr-1000000/caffenet_con-fc6_scratch_pad24_imS227_iter_150000.caffemodel	 -gpu 1	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/imSz128_ntr30_run0/log_train_pre-kitti_fc6_ft-last_drop-last_mxl-5_inLr1e-03.txt 
