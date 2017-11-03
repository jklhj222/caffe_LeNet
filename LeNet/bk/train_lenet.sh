#!/usr/bin/env sh
set -e

#./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
/home/s2c/pkg/local/caffe-master_cuDNN/build/tools/caffe train --solver=./lenet_solver.prototxt $@
