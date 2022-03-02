import os
import numpy as np
import time
import shutil
import cPickle
import random

from PIL import Image, ImageOps


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def move_files(input, output):
    '''
        Input: folder with dataset, where every class is in separate folder
        Output: all images, in format class_number.jpg; output path should be absolute
    '''
    index = -1
    for root, dirs, files in os.walk(input):
        path = root.split('\\')
        print 'Working with path ', path
        print 'Path index ', index
        filenum = 0
        folder = root.split('/')
        if (index >= 0):
            folder = folder[6]
        for file in files:
            fileName, fileExtension = os.path.splitext(file)
            if fileExtension == '.jpg' or fileExtension == '.JPG':
                full_path = path[0] + '/' + file
                print full_path
                if (os.path.isfile(full_path)):
                    file = str(index) + '_' + folder  + str(filenum) + fileExtension
                    print output + '/' + file
                    shutil.copy(full_path, output + '/' + file)
                filenum += 1
        index += 1


def create_text_file(input_path, outpath, percentage):
    '''
        Creating train.txt and val.txt for feeding Caffe
    '''

    images, labels = [], []
    os.chdir(input_path)

    for item in os.listdir('.'):
        if not os.path.isfile(os.path.join('.', item)):
            continue
        try:
            label = int(item.split('_')[0])
            images.append(item)
            labels.append(label)
        except:
            continue

    images = np.array(images)
    labels = np.array(labels)
    images, labels = shuffle_in_unison(images, labels)

    X_train = images[0:len(images) * percentage]
    y_train = labels[0:len(labels) * percentage]

    X_test = images[len(images) * percentage:]
    y_test = labels[len(labels) * percentage:]

    os.chdir(outpath)

    trainfile = open("train.txt", "w")
    for i, l in zip(X_train, y_train):
        trainfile.write(i + " " + str(l) + "\n")

    testfile = open("val.txt", "w")
    for i, l in zip(X_test, y_test):
        testfile.write(i + " " + str(l) + "\n")

    trainfile.close()
    testfile.close()

def main():
    caffe_path = '/opt/caffe/data/dog/Images'
    new_path = '/opt/caffe/data/dog/Images_Caffe'
    move_files(caffe_path, new_path)
    create_text_file(new_path, './', 0.85)

main()
