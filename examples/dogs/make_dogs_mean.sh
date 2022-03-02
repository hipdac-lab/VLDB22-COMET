#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/opt/caffe/examples/dogs
DATA=/opt/caffe/data/dogs/Images_Caffe
TOOLS=/opt/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/dogs_train_lmdb \
  $DATA/dogs_mean.binaryproto

echo "Done."
