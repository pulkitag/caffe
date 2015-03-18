#!/usr/bin/env sh 
 
TOOLS=/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools/ 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=mnist_siamese_solver.prototxt	 2>&1 | tee log.txt 
