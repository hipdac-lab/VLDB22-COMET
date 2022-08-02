# COMET: A Novel Memory-Efficient Deep Learning Training Framework by Using Error-Bounded Lossy Compression

By [Sian Jin](sian.jin@wsu.edu), [Chengming Zhang](chengming.zhang@wsu.edu), [Xintong Jiang](xintong.jiang@mail.mcgill.ca), [Yunhe Feng](yunhe@uw.edu), [Hui Guan](huiguan@cs.umass.edu), [Guanpeng Li](guanpeng-li@uiowa.edu), [Shuaiwen Leon Song](shuaiwen.song@sydney.edu.au), and [Dingwen Tao](dingwen.tao@wsu.edu).

COMET is modified from [Caffe](https://github.com/BVLC/caffe) framework to enable memory-efficient deep learning training by using error-bounded lossy compressor [SZ](https://github.com/szcompressor/SZ). We focused on modifying the layer function of Caffe [1] to support SZ [2] for compressing activation data.

## Method 1: Use Docker Image (Recommended)

To ease the use of COMET, we provide a docker image with the essential environment.

### Step 1: Pull the docker image

Assuming [docker](https://docs.docker.com/get-docker/) has been installed, please run the following command to pull our prepared docker image (https://hub.docker.com/r/jinsian/caffe) from DockerHub:
```
docker pull jinsian/caffe
```
The image comes with the well-set COMET and the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) for evaluating the training process with AlexNet [3].

### Step 2: Trainining with COMET

First, launch the docker image:

```
docker run -ti jinsian/caffe:COMET /bin/bash
```

Then, use the following command to simply start the training with AlexNet on the Stanford Dogs dataset with COMET:

```
cd /opt/caffe
./build/tools/caffe train -solver ./models/bvlc_reference_caffenet/solver.prototxt
```

After the training process starts, the compression ratio on all convolutional layers for each iteration should be displayed as follows.

<img width="352" alt="example" src="https://user-images.githubusercontent.com/50967682/156245957-1e22380b-802c-48e8-9ead-c73c6fbe026b.png">

Note that like the original Caffe, **solver.prototxt** can be modified to adjust the training strategy (e.g., learning rate, stepsize, etc.). 

## Method 2: Build From Source

### Step 1: Install SZ

First, clone the repo:

```
git clone https://github.com/jinsian/VLDB22-COMET
```

In order to build COMET, please install SZ following the instructions shown in https://github.com/szcompressor/SZ.
Next, include the path of your SZ library before building Caffe.

```
export CAFFE_HOME=/opt/VLDB22-COMET  // modify the path based on your system
export SZ_HOME=/opt/SZ  // modify the path based on your system
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SZ_HOME/install/lib
```

### Step 2: Compile Caffe for COMET

First, modify the link_directories in **CMakeLists.txt** (line 73 & 74) to your SZ include ($SZ_HOME/install/include) and lib ($SZ_HOME/install/lib) location, respectively.

```
cd $CAFFE_HOME
vim ./CMakeLists.txt
```

Then, modify the location to your sz configuration file in **conv_layer.cpp** (line 61) to $SZ_HOME/example/sz.config.

```
cd $CAFFE_HOME
vim ./src/caffe/layers/conv_layer.cpp
```

Finally, follow the instructions in https://github.com/BVLC/caffe to install prerequisites, modify the configurations based on your system, and compile and install Caffe with CMake.

```
cd $CAFFE_HOME
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=[INSTALL_DIR]
make all
make install
make runtest
```

### Step 3: Download and prepare dataset

First, Download and prepare the ImageNet dataset following the instructions shown in https://github.com/BVLC/caffe; or follow the below steps to download the [Stanford Dog dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html) and create its lmdb data format. The Stanford Dogs dataset is smaller than ImageNet for easy test. 

To download and unzip the dataset:

```
cd $CAFFE_HOME/data/dogs
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
tar xvf ./images.tar
```

Next, create the train.txt and val.txt. train.txt/val.txt contain the paths to images and class numbers from the training/testing data, respectively. Please modify the path (line 104) in **tag.py** to $CAFFE_HOME/data/dogs/Images and the path (line 105) to $CAFFE_HOME/data/dogs/Images_Caffe.

```
mkdir ./Images_Caffe
python ./tag.py 
```

Furthermore, create the lmdb dataset. Please modify the paths (line 6-11) in **create_imagenet.sh** as follows:

```
EXAMPLE=$CAFFE_HOME/examples/dogs
DATA=$CAFFE_HOME/data/dogs/Images_Caffe
TOOLS=$CAFFE_HOME/build/tools

TRAIN_DATA_ROOT=$CAFFE_HOME/data/dogs/Images_Caffe/
VAL_DATA_ROOT=$CAFFE_HOME/data/dogs/Images_Caffe/
```

Run the script:

```
source ./create_dogs.sh
```

Finally, please modify the paths (line 5-7) in **make_imagenet_mean.sh** as follows:

```
EXAMPLE=$CAFFE_HOME/examples/dogs
DATA=$CAFFE_HOME/data/dogs/Images_Caffe
TOOLS=$CAFFE_HOME/build/tools
```

Run the script to create the compute the mean with the following command:

```
cd ../../examples/dogs
source ./make_dogs_mean.sh
```

### Step 4: Training with COMET

Now, run the following command to train AlexNet on the Stanford Dogs dataset with COMET:

```
cd $CAFFE_HOME
./build/tools/caffe train -solver ./models/bvlc_reference_caffenet/solver.prototxt
```

After the training process starts, the compression ratio on all convolutional layers for each iteration should be displayed as follows.

<img width="352" alt="example" src="https://user-images.githubusercontent.com/50967682/156245957-1e22380b-802c-48e8-9ead-c73c6fbe026b.png">

Note that like the original Caffe, **solver.prototxt** can be modified to adjust the training strategy (e.g., learning rate, stepsize, etc.). 

## References
[1] Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. "Caffe: Convolutional architecture for fast feature embedding." In *Proceedings of the 22nd ACM international conference on Multimedia*, pp. 675-678. 2014.

[2] Tian, Jiannan, Sheng Di, Kai Zhao, Cody Rivera, Megan Hickman Fulp, Robert Underwood, Sian Jin, Xin Liang, Jon Calhoun, Dingwen Tao, and Franck Cappelo. "cuSZ: An Efficient GPU-Based Error-Bounded Lossy Compression Framework for Scientific Data." In *Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques*, pp. 3-15. 2020.

[3] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." *Advances in neural information processing systems* 25 (2012).
