import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
 
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.functions.loss.mean_squared_error import mean_squared_error
from chainer.datasets import tuple_dataset
import optuna

import matplotlib.pyplot as plt

import math
import functools
import pickle

class Model(Chain):
    def __init__(self, list, dr=0.2):
        lst=[]
        p=list[0]
        for e in list[1:]:
            lst.append(L.LSTM(p, e))
            p=e
        super(Model, self).__init__()
        with self.init_scope():
            self.dr=dr
            self.train=True
            self.s=L.Linear(1,list[0])
            self.l=lst
            self.t=L.Linear(p, 1)
         
    def reset_state(self):
        for e in self.l:
            e.reset_state()
         
    def __call__(self, x):
        h = F.dropout(F.relu(self.s(x)), ratio=self.dr)
        for e in self.l:
            h = F.dropout(F.relu(e(h)), ratio=self.dr)
        o = self.t(h)
        return o

class LossFuncL(Chain):
    def __init__(self, predictor):
        super(LossFuncL, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        #print(type(x))
        if isinstance(x, np.ndarray):
            x=chainer.Variable(x)
            t=chainer.Variable(t)
        #print(x.data)
        x.data = x.data.reshape((-1, 1)).astype(np.float32)
        t.data = t.data.reshape((-1, 1)).astype(np.float32)

        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        report({'loss':loss}, self)
        return loss

class LSTM_Iterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size = 10, seq_len = 5, repeat = True):
        self.seq_length = seq_len
        self.dataset = dataset
        self.nsamples =  len(dataset)

        self.batch_size = batch_size
        self.repeat = repeat

        self.epoch = 0
        self.iteration = 0
        self.offsets = np.random.randint(0, len(dataset),size=batch_size)

        self.is_new_epoch = False

    def __next__(self):
        if not self.repeat and self.iteration * self.batch_size >= self.nsamples:
            raise StopIteration

        x, t = self.get_data()
        self.iteration += 1

        epoch = self.iteration // self.batch_size
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch
            self.offsets = np.random.randint(0, self.nsamples,size=self.batch_size)

        #print(x)
        return list(zip(x, t))

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / len(self.dataset)

    def get_data(self):
        tmp0 = [self.dataset[(offset + self.iteration)%self.nsamples][0]
               for offset in self.offsets]
        tmp1 = [self.dataset[(offset + self.iteration + 1)%self.nsamples][0]
               for offset in self.offsets]
        return tmp0,tmp1

    def serialzie(self, serialzier):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch     = serializer('epoch', self.epoch)

class LSTM_updater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(LSTM_updater, self).__init__(train_iter, optimizer, device=device)
        self.seq_length = train_iter.seq_length

    def update_core(self):
        loss = 0

        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        for i in range(self.seq_length):
            batch = np.array(train_iter.__next__()).astype(np.float32)
            x, t  = batch[:,0].reshape((-1,1)), batch[:,1].reshape((-1,1))
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.zerograds()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

def getdata():
    #normalize
    data = np.loadtxt("./traffic_1000s.txt", comments='#').astype(np.float32)
    data = np.log(data + 1)
    data_max = np.max(data)
    data = data/data_max
    return data, data_max

def expand(data, data_max):
    return np.exp(np.multiply(data, data_max)) - 1
