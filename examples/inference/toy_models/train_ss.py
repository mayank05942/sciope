from sciope.models.cnn_regressor import CNNModel

import numpy as np
from create_summary_statistics import summarys
import pickle
import time

from load_data import load_spec

clay=[32,48,64,96]
num_timestamps=401
endtime=200
modelname = "vilar_ACR_" + str(endtime) + "_" + str(num_timestamps) + '_all_species'
species = [6]

train_thetas, train_ts = load_spec(modelname=modelname, type = "train", species=species)
train_sum = pickle.load(open('datasets/' + modelname + '/train_sum.p', "rb" ) )
validation_sum = pickle.load(open('datasets/' + modelname + '/val_sum.p', "rb" ) )
validation_ts = pickle.load(open('datasets/' + modelname + '/validation_ts.p', "rb" ) )[:,:,species]
test_thetas = pickle.load(open('datasets/' + modelname + '/test_sum.p', "rb" ) )
test_ts = pickle.load(open('datasets/' + modelname + '/test_ts.p', "rb" ) )[:,:,species]


nnm = CNNModel(input_shape=(train_ts.shape[1],train_ts.shape[2]), output_shape=(2), con_len=3, con_layers=clay, dense_layers=[100,100,100])

history1 = nnm.train(inputs=train_ts, targets=train_sum,validation_inputs=validation_ts,validation_targets=validation_sum,
          batch_size=32, epochs=40*10, val_freq=1, early_stopping_patience=5, plot_training_progress=False, verbose=1)

