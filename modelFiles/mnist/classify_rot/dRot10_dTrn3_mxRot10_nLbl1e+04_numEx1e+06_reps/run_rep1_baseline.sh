#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/classify_rot/dRot10_dTrn3_mxRot10_nLbl1e+04_numEx1e+06_reps/mnist_siamese_solver_baseline_rep1_baseline.prototxt	 -gpu 0	 2>&1 | tee /work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/classify_rot/dRot10_dTrn3_mxRot10_nLbl1e+04_numEx1e+06_reps/log_rep1_baseline.txt 
