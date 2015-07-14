#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --weights=/data1/pulkitag/projRotate/snapshots/kitti/mxDiff-7_pose-slowness_nrmlz-none_randcrp_concat-fc6_nTr-1000000/caffenet_con-conv5_scratch_pad24_imS227_con-conv_ct-margin1_iter_60000.caffemodel	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/exp/fine-FROM-mxDiff-7_pose-slowness_nrmlz-none_randcrp_concat-fc6_nTr-1000000/solver_con-conv5_scratch_pad24_imS227_con-conv_small-data_sun_mItr60K_lrAbv-class_fc_bLr1e-03_run5_datN5e+00_mxl-relu4_drop_muImgnt.prototxt	 -gpu 1	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/exp/fine-FROM-mxDiff-7_pose-slowness_nrmlz-none_randcrp_concat-fc6_nTr-1000000/log_train_con-conv5_scratch_pad24_imS227_con-conv_small-data_sun_mItr60K_lrAbv-class_fc_bLr1e-03_run5_datN5e+00_mxl-relu4_drop_muImgnt.txt 
