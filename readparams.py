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
from lib import *


data,data_max = getdata()

with open("trial_params.dump", 'rb') as dmp:
    trial_params = pickle.load(dmp)

seq_len = trial_params['seq_len']
data_size=int(6.0/8*len(data))+int(len(data)/8)
train = tuple_dataset.TupleDataset(data[:data_size])
train_iter = LSTM_Iterator(train, batch_size=10, seq_len=seq_len)

data2_size=len(data)-data_size
data2 = data[data_size:data_size+data2_size]
test = tuple_dataset.TupleDataset(data2)
test_iter = LSTM_Iterator(test, batch_size=10, seq_len=seq_len, repeat=False)

#trial_params = {"n_layers":2,
#    "n_units_l0":4.089363630348063,
#    "n_units_l1":68.64689283514159,
#    "adam_alpha":0.07205877227474339,
#    "weight_decay":3.0796992465748975e-05}

n_layers = trial_params['n_layers']

layers = []

for i in range(n_layers):
    n_units = int(trial_params['n_units_l{}'.format(i)])
    layers.append(n_units)

model = LossFuncL(Model(layers))
#serializers.load_npz("model.npz", model)

adam_alpha = trial_params['adam_alpha']
optimizer = optimizers.Adam(alpha=adam_alpha)
weight_decay = trial_params['weight_decay']
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

updater = LSTM_updater(train_iter, optimizer, -1)
trainer = training.Trainer(updater, (30, 'epoch'), out='result')

trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
trainer.extend(extensions.ProgressBar())
trainer.run()

model.predictor.reset_state()

res=[]
for i in range(data_size):
    y=model.predictor(chainer.Variable(train[i][0].reshape(-1,1)))
    res.append(train[i][0])
for i in range(data2_size):
    y = model.predictor(chainer.Variable(y.data))
    res.append(y.data)

res = np.array(res).astype(np.float32)
train = expand(train,data_max)
test = expand(test,data_max)
res = expand(res,data_max)
plt.plot([i for i in range(data_size)], train)
plt.plot([data_size + i for i in range(data2_size)], test)
plt.plot([i for i in range(data_size+data2_size)], res)
plt.show() 

serializers.save_npz("model.npz", model)