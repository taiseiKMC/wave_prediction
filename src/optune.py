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

def create_model(trial):

    # 全結合層の数を保存
    n_layers = trial.suggest_int('n_layers', 1, 10)

    layers = []

    for i in range(n_layers):
        n_units = int(trial.suggest_loguniform('n_units_l{}'.format(i), 4, 128))
        layers.append(n_units)

    dropout = trial.suggest_uniform('dropout', 0, 0.5)
    return LossFuncL(Model(layers, dropout))

def create_optimizer(trial, model):
    # 最適化関数の選択
    adam_alpha = trial.suggest_loguniform('adam_alpha', 1e-5, 1e-1)
    optimizer = chainer.optimizers.Adam(alpha=adam_alpha)

    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    return optimizer

def objective(trial):
    seq_len = trial.suggest_int('seq_len', 10, 300)
    rate=6.0/8
    data_size=int(len(data)*rate)
    train = tuple_dataset.TupleDataset(data[:data_size])
    train_iter = LSTM_Iterator(train, batch_size=10, seq_len=seq_len)

    data2_size=int(len(data)/8)
    data2 = data[data_size:data_size+data2_size]
    test = tuple_dataset.TupleDataset(data2)
    test_iter = LSTM_Iterator(test, batch_size=10, seq_len=seq_len, repeat=False)
    #test_iter = iterators.SerialIterator(test, 10, repeat=False, shuffle=False)

    model = create_model(trial)
    optimizer = create_optimizer(trial, model)
    updater = LSTM_updater(train_iter, optimizer, -1)

    stop_trigger = training.triggers.EarlyStoppingTrigger(
        monitor='validation/main/loss', check_trigger=(5, 'epoch'),
        max_trigger=(100, 'epoch'))
    trainer = training.Trainer(updater, stop_trigger, out="result")
    #trainer = training.Trainer(updater, (100, 'epoch'), out='result')

    test_model = model.copy()
    test_rnn = test_model.predictor
    test_rnn.train = False
    trainer.extend(extensions.Evaluator(test_iter, test_model, device=-1))

    trainer.extend(
        optuna.integration.ChainerPruningExtension(
            trial, 'validation/main/loss', (5, 'epoch')))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar())
    log_report_extension = extensions.LogReport(log_name=None)
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
    trainer.extend(log_report_extension)
    trainer.run()

    
    # 学習結果の保存
    log_last = log_report_extension.log[-1]
    print(log_last)
    for key, value in log_last.items():
        trial.set_user_attr(key, value)

    # 最終的なバリデーションの値を返す
    val_err = log_report_extension.log[-1]['validation/main/loss']
    return val_err


study = optuna.create_study(storage=f"sqlite:///shortwave.db",
    load_if_exists=True, pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)
trial = study.best_trial
print('Params: ')
for key, value in trial.params.items():
    print('{}:{}'.format(key, value))

print('User attrs: ')
for key, value in trial.user_attrs.items():
    print('{}:{}'.format(key, value))

with open("trial_params.dump", 'wb') as dmp:
  pickle.dump(trial.params , dmp)