#!/usr/bin/env sh 
 
TOOLS=../../../../build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe test	 --model=mnist_test.prototxt --weights=/data1/pulkitag/mnist/snapshots/classify_rot/dRot10_dTrn3_mxRot10_nLbl1e+03_numEx2e+05/mnist__iter_30000.caffemodel -gpu 0 --iterations=100	 2>&1 | tee log_test.txt 
