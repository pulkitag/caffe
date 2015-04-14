#!/usr/bin/env sh 
 
TOOLS=../../../../build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=mnist_siamese_solver_baseline.prototxt	 2>&1 | tee log_baseline.txt 
