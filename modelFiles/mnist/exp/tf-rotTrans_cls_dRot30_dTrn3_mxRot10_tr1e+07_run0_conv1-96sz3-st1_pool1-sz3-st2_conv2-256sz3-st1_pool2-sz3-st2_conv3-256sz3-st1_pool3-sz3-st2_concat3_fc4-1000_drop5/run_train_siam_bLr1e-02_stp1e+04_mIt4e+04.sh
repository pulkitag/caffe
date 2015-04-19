#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/exp/tf-rotTrans_cls_dRot30_dTrn3_mxRot10_tr1e+07_run0_conv1-96sz3-st1_pool1-sz3-st2_conv2-256sz3-st1_pool2-sz3-st2_conv3-256sz3-st1_pool3-sz3-st2_concat3_fc4-1000_drop5/solver_siam_bLr1e-02_stp1e+04_mIt4e+04.prototxt	 -gpu 1	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/exp/tf-rotTrans_cls_dRot30_dTrn3_mxRot10_tr1e+07_run0_conv1-96sz3-st1_pool1-sz3-st2_conv2-256sz3-st1_pool2-sz3-st2_conv3-256sz3-st1_pool3-sz3-st2_concat3_fc4-1000_drop5/log_train_siam_bLr1e-02_stp1e+04_mIt4e+04.txt 
