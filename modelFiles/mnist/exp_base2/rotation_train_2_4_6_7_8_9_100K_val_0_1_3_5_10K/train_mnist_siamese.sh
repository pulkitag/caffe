#!/usr/bin/env sh 
 
TOOLS=/home/eecs/pulkitag/Research/codes/codes/projCaffe/caffe-v2-2/build/tools/ 
 
GLOG_logtostderr=1 $TOOLS/caffe train	 --solver=mnist_siamese_solver.prototxt	 2>&1 | tee log.txt 
