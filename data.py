import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image

def getData(file):
    data=pd.read_csv(file)
    return data.shape[0], data
def FERLoad():
    n,train=getData('train.csv')
    index=28709+1

    y_train=tf.keras.utils.to_categorical(train['emotion'].iloc[0:index].to_numpy())
    x_train=np.asarray([np.reshape(np.asarray(tmp.split(),dtype=float),(-1,48)) for tmp in train[' pixels'].iloc[0:index]]).reshape(index,48,48,1)
    y_test=tf.keras.utils.to_categorical(train['emotion'].iloc[index:n+1].to_numpy())
    x_test=np.asarray([np.reshape(np.asarray(tmp.split(),dtype=float),(-1,48)) for tmp in train[' pixels'].iloc[index:n+1]]).reshape(n-index,48,48,1)

    return x_train, y_train, x_test, y_test

def ExpwLoadFromNp(im_path):
    f1 = open(im_path+'/'+"x_train.npy",'r')
    f2 = open(im_path+'/'+"y_train.npy",'r')
    f3 = open(im_path+'/'+"x_test.npy",'r')
    f4 = open(im_path+'/'+"y_test.npy",'r')
    return np.load(f1), np.load(f2), np.load(f3), np.load(f4)

def ExpwLoad(im_path, portion):
    x_train, y_train, x_test, y_test = [], [], [], []
    f = open(im_path+'/'+"label.lst",'r')
    i = 0
    end = int(88600*portion)
    train = int(end*0.75)
    for x in f:
        data = x.split()
        i += 1
        im = Image.open(im_path+'/'+"image/"+data[1]+'_'+data[0])
        im = np.asarray(im)

        if i<=train:
            x_train.append(im)
            y_train.append(data[7])
        else:
            x_test.append(im)
            y_test.append(data[7])
        if i==end:
            break
    return np.asarray(x_train), tf.keras.utils.to_categorical(np.asarray(y_train)), np.asarray(x_test), tf.keras.utils.to_categorical(np.asarray(y_test))


def ExpwCut(im_path):
    f = open(im_path+'/'+"label.lst",'r')
    for x in f:
        data = x.split()
        im = Image.open(im_path+'/'+data[0])
        im = im.crop((float(data[3]),float(data[2]),float(data[4]), float(data[5])))
        im.save(im_path+'/'+"image/"+data[1]+'_'+data[0],format="JPEG")


def ExpwResize(im_path):
    f = open(im_path+'/'+"label.lst",'r')
    size = (112,112)
    for x in f:
        data = x.split()
        im = Image.open(im_path+'/'+"image/"+data[1]+'_'+data[0])
        im = im.resize(size)
        im.save(im_path+'/'+"image/"+data[1]+'_'+data[0],format="JPEG")
