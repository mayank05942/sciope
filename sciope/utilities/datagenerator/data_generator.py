

from sciope.utilities.priors import uniform_prior
from sciope.data.dataset import DataSet
import numpy as np
import dask
import pickle
import os
from vilar_class import Vilar


dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]



class DataGenerator:

    def __init__(self, prior_function, sim):
        self.prior_function = prior_function
        self.sim = dask.delayed(sim)
        
    def get_dask_graph(self, batch_size):
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
        trial_param = [self.prior_function.draw() for x in range(batch_size)]


        # Perform the trial
        sim_result = [self.sim(param) for param in trial_param]

        return {"parameters": trial_param, "trajectories": sim_result}
        
    def gen(self, batch_size):

        graph_dict = self.get_dask_graph(batch_size=batch_size)
        res_param, res_sim = dask.compute(graph_dict["parameters"], graph_dict["trajectories"])
        return res_param, res_sim

    def sim_param(self,param):
        return self.sim(param)
    

print("start")

# Defining prior function
prior_function = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))

# Defining Vilar model as stochastic model
stoch_model = Vilar(final_time=500, species='all')
sim = stoch_model.simulate

print("species list: ", stoch_model.model.listOfSpecies.keys())

# Defining DataGenerator
dg = DataGenerator(prior_function=prior_function, sim=sim)



dataset=DataSet(name='test dataset')

for filenr in range(5):
    for epoch in range(20):
        tp, sim_result = dg.gen(batch_size=1000)
        dataset.add_points(inputs=np.array(tp), targets=None, time_series=np.array(sim_result), summary_stats=None)
        print("dataset size: ", np.array(dataset.ts).shape)

    # Name the dataset
    dataset_name = 'ds_' + stoch_model.name


    if os.path.exists(dataset_name):
        nr = 0
        while os.path.exists(dataset_name+'/'+dataset_name + str(nr) + '.p'):
            nr += 1
        # Save dataset
        pickle.dump(dataset, open(dataset_name+'/'+dataset_name + str(nr) + '.p', "wb"))
    else:
        os.mkdir(dataset_name)
        # Save dataset
        pickle.dump(dataset, open(dataset_name + '/' + dataset_name + str(0) + '.p', "wb"))

    print("complete file ", str(filenr))







