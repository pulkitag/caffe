#!/usr/bin/env sh 
 
TOOLS=../../../build/tools 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=mnist_siamese_solver.prototxt	 2>&1 | tee log.txt 
