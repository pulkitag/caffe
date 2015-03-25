#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe test	 --weights=/data1/pulkitag/mnist/snapshots/classify_rot/dRot10_dTrn3_mxRot10_nLbl1e+02_numEx1e+06/mnist__rep4_iter_50000.caffemodel	 --iterations=100	 -gpu 0	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/classify_rot/dRot10_dTrn3_mxRot10_nLbl1e+02_numEx1e+06_reps/log_test_rep4.txt 
