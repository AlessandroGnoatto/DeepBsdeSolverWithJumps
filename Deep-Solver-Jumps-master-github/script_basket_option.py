import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PureJumpSolver import BSDESolver
import PureJumpEquation as eqn
import munch
from scipy.stats import norm
import pandas as pd

 

if __name__ == "__main__":
    dim = 5 #dimension of brownian motion
    P = 2048*8 #number of outer Monte Carlo Loops
    batch_size = 64
    total_time = 1.0
    num_time_interval = 40
    strike = 0.9
    lamb = 0.3
    r = 0.04
    sigma = 0.25
    aver_jump = 0.5
    var_jump = 0.25
    x_init = 1.0
    config = {
                "eqn_config": {
                    "_comment": "a basket call option",
                    "eqn_name": "BasketOption",
                    "total_time": total_time,
                    "dim": dim,
                    "num_time_interval": num_time_interval,
                    "strike":strike,
                    "r":r,
                    "sigma":sigma,
                    "lamb":lamb,
                    "aver_jump":aver_jump,
                    "var_jump":var_jump,
                    "x_init":x_init,

                },
                "net_config": {
                    "num_hiddens": [dim+20,dim+20],
                    "lr_values": [5e-2, 5e-3],
                    "lr_boundaries": [10000],
                    "num_iterations": 20000,
                    "batch_size": batch_size,
                    "valid_size": 256,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "verbose": True
                }
                }
    config = munch.munchify(config) 
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    
    #apply algorithm 1
    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()  
   
    #Simulate the BSDE after training - MtM scenarios
    samples = bsde.sample(P)
    #simulations = bsde_solver.model.simulate_path(samples)
    
    
    
    
    """
    Monte Carlo Price
    """
    stock = samples[0]
    mcprice = np.exp(-r* total_time)*np.average(np.maximum(np.sum(stock[:,:,-1],1) - dim * strike,0))
    np.disp(mcprice)  
    
    # np.save("training_history_dim_"+str(dim),training_history)
    # np.save("mcprice_dim_"+str(dim),mcprice)