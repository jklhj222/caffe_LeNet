#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

caffe_root = '/home/s2c/pkg/local/caffe-master_cuDNN'

sys.path.insert(0, caffe_root + 'python') # Add the path of pycaffe into environment 
import caffe

# Assign the structure of network, differ with lenet_train_test.prototxt 
MODEL_FILE = 'lenet.prototxt' 
PRETRAINED = 'lenet_iter_10000.caffemodel'

# lenet.prototxt, picture size (28x28）, black and white
IMAGE_FILE = '4.png'
 
input_image = caffe.io.load_image(IMAGE_FILE, color=False)
net = caffe.Classifier(MODEL_FILE, PRETRAINED) 
prediction = net.predict([input_image], oversample = False)
caffe.set_mode_gpu()
print( 'predicted class:', prediction[0].argmax() )
print( 'predicted class2:', prediction[0] )
