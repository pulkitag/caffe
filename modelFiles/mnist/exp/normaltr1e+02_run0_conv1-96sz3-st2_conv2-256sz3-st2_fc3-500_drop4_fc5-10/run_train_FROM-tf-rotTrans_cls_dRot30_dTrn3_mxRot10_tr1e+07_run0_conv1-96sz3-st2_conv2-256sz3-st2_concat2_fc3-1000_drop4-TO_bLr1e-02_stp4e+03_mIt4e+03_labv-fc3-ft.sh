#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --weights=/data1/pulkitag/mnist/snapshots/tf-rotTrans_cls_dRot30_dTrn3_mxRot10_tr1e+07_run0_conv1-96sz3-st2_conv2-256sz3-st2_concat2_fc3-1000_drop4/caffenet_siam_bLr1e-02_stp1e+04_mIt4e+04_iter_40000.caffemodel	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/exp/normaltr1e+02_run0_conv1-96sz3-st2_conv2-256sz3-st2_fc3-500_drop4_fc5-10/solver_FROM-tf-rotTrans_cls_dRot30_dTrn3_mxRot10_tr1e+07_run0_conv1-96sz3-st2_conv2-256sz3-st2_concat2_fc3-1000_drop4-TO_bLr1e-02_stp4e+03_mIt4e+03_labv-fc3-ft.prototxt	 -gpu 1	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/exp/normaltr1e+02_run0_conv1-96sz3-st2_conv2-256sz3-st2_fc3-500_drop4_fc5-10/log_train_FROM-tf-rotTrans_cls_dRot30_dTrn3_mxRot10_tr1e+07_run0_conv1-96sz3-st2_conv2-256sz3-st2_concat2_fc3-1000_drop4-TO_bLr1e-02_stp4e+03_mIt4e+03_labv-fc3-ft.txt 
