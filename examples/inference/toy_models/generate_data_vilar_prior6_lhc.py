"""
The vilar Model: Approximate Bayesian Computation Test Run
"""

# Imports
from sciope.utilities.priors import uniform_prior
from sciope.inference import abc_inference
from sciope.utilities.summarystats import burstiness as bs
import numpy as np
import vilar
from vilar import Vilar_model
import dask
import pickle
import os
import time
from sciope.designs import latin_hypercube_sampling as lhs

from sklearn.metrics import mean_absolute_error


class DataGenerator:

    def __init__(self, sim):
        self.sim = dask.delayed(sim)

    def get_dask_graph(self, thetas):
        """
        Constructs the dask computational graph invloving sampling, simulation, summary statistics
        and distances.

        Parameters
        ----------
        batch_size : int
            The number of points being sampled in each batch.

        Returns
        -------
        dict
            with keys 'parameters', 'trajectories'
        """

        # Rejection sampling with batch size = batch_size

        # Draw from the prior
        trial_param = thetas

        # Perform the trial
        sim_result = [self.sim(param) for param in trial_param]

        return {"parameters": trial_param, "trajectories": sim_result}

    def gen(self, thetas):
        graph_dict = self.get_dask_graph(thetas=thetas)
        res_param, res_sim = dask.compute(graph_dict["parameters"], graph_dict["trajectories"])
        res_param = np.squeeze(np.array(res_param))
        res_sim = np.squeeze(np.array(res_sim))
        return res_param, res_sim

    def sim_param(self, param):
        return self.sim(param)



num_timestamps=401
endtime=200

Vilar_ = Vilar_model(num_timestamps=num_timestamps, endtime=endtime)


simulate = Vilar_.simulate

modelname = "vilar_ACR_prior6_" + str(endtime) + "_" + str(num_timestamps)



if not os.path.exists('datasets/lhc'):
    os.mkdir('datasets/lhc')


if not os.path.exists('datasets/lhc/'+modelname):
    os.mkdir('datasets/lhc/'+modelname)

dmin =           [0,   100,    0,   20,   10,   1,    1,   0,   0,   0, 0.5,    0,   0,    0,   0]
dmax =         [  80,   600,    4,   60,   60,   7,   12,   2,   3, 0.7, 2.5,   4,   3,   70,   300]
true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]



dg = DataGenerator(sim=simulate)
print("generating some data")
nr=0

if not os.path.exists('datasets/lhc'):
    os.mkdir('datasets/lhc')

if not os.path.isfile('datasets/lhc/' + modelname + '/train_thetas_'+str(1)+'.p'):
    lhs_obj = lhs.LatinHypercube(dmin, dmax)
    lhs_delayed = lhs_obj.generate(100000)
    train_thetas, = dask.compute(lhs_delayed)

train_thetas = np.reshape(train_thetas,(3,-1))
print("train_thetas shape: ", train_thetas.shape)




nr=0
while os.path.isfile('datasets/lhc/' + modelname + '/train_ts_'+str(nr)+'.p'):
    nr += 1
delta_t = time.time()
for nr in range(nr,3):
    train_thetas = pickle.load(open())
    train_ts = np.zeros((0,num_timestamps,3))
    delta_datapoints = train_thetas.shape[0]

    for i in range(100):
        ts = dg.gen(thetas=t[i])
        train_ts = np.concatenate((train_thetas,param),axis=0)
        # print("train_ts shape: ", train_ts.shape, ", ts shape: ", ts.shape)
        train_ts = np.concatenate((train_ts,ts),axis=0)
        if i == 0:
            tmin = np.min(train_thetas,axis=0)
            tmax = np.max(train_thetas,axis=0)

            for j in range(15):
                print("index: ", j, ", tmin: ", "{0:.1f}".format(tmin[j]), ", tmax: ", "{0:.1f}".format(tmax[j]),
                      ", dmin: ", "{0:.1f}".format(dmin[j]), ", dmax: ", "{0:.1f}".format(dmax[j]))

        print("trainig data shape: train_ts: ", train_ts.shape, ", train_thetas: ", train_thetas.shape)
        delta_t = time.time() - delta_t
        delta_datapoints = train_thetas.shape[0] - delta_datapoints

        print("delta time: ", delta_t, "s, delta_datapoints/delta_t: ", "{0:.2f}".format( delta_datapoints/delta_t ))

    print("generating trainig data done, shape: train_ts: ", train_ts.shape, ", train_thetas: ", train_thetas.shape)

    pickle.dump( train_ts, open( 'datasets/lhc/' + modelname + '/train_ts_'+str(nr)+'.p', "wb" ) )

# validation_thetas = np.zeros((0,15))
# validation_ts = np.zeros((0,num_timestamps,3))
# for i in range(20):
#     param, ts = dg.gen(batch_size=1000)
#     validation_thetas = np.concatenate((validation_thetas,param),axis=0)
#     validation_ts = np.concatenate((validation_ts,ts),axis=0)
#     if i%10 == 0:
#         print("validation data shape: train_ts: ", validation_ts.shape, ", train_thetas: ", validation_thetas.shape)
#
# print("generating validation data done, shape: validation_ts: ", validation_ts.shape, ", validation_thetas: ", validation_thetas.shape)
#
#
# pickle.dump( validation_thetas, open( 'datasets/lhc/' + modelname + '/validation_thetas.p', "wb" ) )
# pickle.dump( validation_ts, open( 'datasets/lhc/' + modelname + '/validation_ts.p', "wb" ) )
#
#
# test_thetas = np.zeros((0,15))
# test_ts = np.zeros((0,num_timestamps,3))
# for i in range(100):
#     param, ts = dg.gen(batch_size=1000)
#     test_thetas = np.concatenate((test_thetas,param),axis=0)
#     test_ts = np.concatenate((test_ts,ts),axis=0)
#
#     print("test data shape: test_ts: ", test_ts.shape, ", test_thetas: ", test_thetas.shape)
#
# print("generating test data done, shape: test_ts: ", test_ts.shape, ", test_thetas: ", test_thetas.shape)
# pickle.dump( test_thetas, open( 'datasets/lhc/' + modelname + '/test_thetas.p', "wb" ) )
# pickle.dump( test_ts, open( 'datasets/lhc/' + modelname + '/test_ts.p', "wb" ) )
#
# abc_trial_thetas = np.zeros((0,15))
# abc_trial_ts = np.zeros((0,num_timestamps,3))
# for i in range(100):
#     param, ts = dg.gen(batch_size=1000)
#     abc_trial_thetas = np.concatenate((abc_trial_thetas,param),axis=0)
#     abc_trial_ts = np.concatenate((abc_trial_ts,ts),axis=0)
#     print("abc_trial data shape: abc_trial_ts: ", abc_trial_ts.shape, ", abc_trial_thetas: ", abc_trial_thetas.shape)
#
# pickle.dump( abc_trial_thetas, open( 'datasets/lhc/' + modelname + '/abc_trial_thetas.p', "wb" ) )
# pickle.dump( abc_trial_ts, open( 'datasets/lhc/' + modelname + '/abc_trial_ts.p', "wb" ) )
#
#
#
#
#
#
#
#
#
