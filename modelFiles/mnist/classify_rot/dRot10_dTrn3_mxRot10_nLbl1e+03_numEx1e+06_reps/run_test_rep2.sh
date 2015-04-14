#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe test	 --weights=/data1/pulkitag/mnist/snapshots/classify_rot/dRot10_dTrn3_mxRot10_nLbl1e+03_numEx1e+06/mnist__rep2_iter_50000.caffemodel	 --model=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/classify_rot/dRot10_dTrn3_mxRot10_nLbl1e+03_numEx1e+06_reps/mnist_siamese_train_test_rep2.prototxt 	 --iterations=100	 -gpu 0	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/classify_rot/dRot10_dTrn3_mxRot10_nLbl1e+03_numEx1e+06_reps/log_test_rep2.txt 