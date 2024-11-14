import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PureJumpSolver import BSDESolver
import PureJumpEquation_CGMYprova as eqn
import munch
from scipy.stats import norm
import pandas as pd
from CGMY import CallPricingFFT 

plt.rcParams['figure.dpi'] = 300

 

if __name__ == "__main__":
    dim = 1 #dimension of brownian motion
    P = 2**16 #number of outer Monte Carlo Loops
    batch_size = 2**7
    total_time = 1.0
    num_time_interval = 100
    strike = 0.9
    lamb = 0.3
    r = 0.04
    # sigma = 0.25
    aver_jump = 0.5
    var_jump = 0.25**2
    x_init = 1.0
    C=0.15
    G=13.
    M=14.
    Y=0.6

    eps = 0.00001
    d = 0 # è il dividend yield
    # h = 0.01 è il delta_t
    # n = 100 è il num_time_interval
    # Npaths = int(1e3) è la P
    config = {
                "eqn_config": {
                    "_comment": "a call contract with CGMY",
                    "eqn_name": "CallOptionCGMY",
                    "total_time": total_time,
                    "dim": dim,
                    "num_time_interval": num_time_interval,
                    "strike":strike,
                    "r":r,
                    #"sigma":sigma,
                    "lamb":lamb,
                    "aver_jump":aver_jump,
                    "var_jump":var_jump,
                    "x_init":x_init,
                    "C":C,
                    "G":G,
                    "M":M,
                    "Y":Y,
                    "eps":eps,
                    "d":d,
                },
                "net_config": {
                    "num_hiddens": [20,20],
                    "lr_values": [1.0e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4],#"lr_values": [5e-2, 5e-3, 5e-4, 1e-4],
                    "lr_boundaries": [1000, 2000, 3000, 4000, 5000],  #"lr_values": [5e-2, 5e-3],
                    "num_iterations": 6000,
                    "batch_size": batch_size,
                    "valid_size": 256,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "y_init_range": [0., 0.3],
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
    simulations = bsde_solver.model.simulate_path(samples)
    
    fig = plt.figure()
    plt.plot(simulations[0:10,0,:].T)
    plt.xlabel('Time step')
    plt.ylabel('Y')
    plt.show()
    
    
    
    
    """
    Monte Carlo Price
    """
    stock = samples[0]
    mcprice = np.exp(-r* total_time)*np.average(np.maximum(stock[:,0,-1] - strike,0))
    np.disp(mcprice)
    
    
    
    
#%%
#Simulate the BSDE after training - MtM scenarios
#samples = bsde.sample(P)
samples = bsde.sample(2**6)

simulations = bsde_solver.model.simulate_path(samples)
  
fig = plt.figure()
plt.plot(simulations[0:10,0,:].T)
plt.xlabel('Time step')
plt.ylabel('Y')
plt.show()
  
  
  
  
"""
Monte Carlo Price
"""
stock = samples[0]
mcprice = np.exp(-r* total_time)*np.average(np.maximum(stock[:,0,-1] - strike,0))
np.disp(mcprice)
  
  
  
    
    
    
    
    

#%%
# Create the plot with a specific figure size
NN = np.int(6000/100)
plt.figure(figsize=(4, 6))  # Adjust the dimensions to match the aspect ratio of your image

plt.plot(training_history[:NN,0], training_history[:NN,2], label=f'$Y_0$ (Approx.)')
plt.plot(training_history[:NN,0], mcprice * np.ones(len(training_history[:NN,0])), '--', color='red', label=f'$Y_0$ (Ref.)', linewidth=2)
mcprice = 0.1375305233367931 # FFT price

# Add labels and title
plt.xlabel('Number of iterations')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(4, 6))  # Adjust the dimensions to match the aspect ratio of your image
plt.plot(training_history[:NN,0], training_history[:NN,1], label='Loss value')

# Add labels and title
plt.xlabel('Number of batch iterations')
plt.yscale('log')
plt.legend()
plt.grid()

# Show the plot
plt.figure()
plt.plot(training_history[:NN,0], training_history[:NN,2], label=f'$Y_0$ (Approx.)')
plt.plot(training_history[:NN,0], np.ones(len(training_history[:NN,0])), '--', color='red', label=f'$Y_0$ (Ref.)', linewidth=2)
# Add labels and title
plt.xlabel('Number of iterations')
plt.ylabel('Y')
plt.legend()
plt.grid()

# Show the plot
plt.figure()
plt.plot(training_history[:NN,0], training_history[:NN,2], label=f'$Y_0$ (Approx.)')
plt.plot(training_history[:NN,0], mcprice * np.ones(len(training_history[:NN,0])), '--', color='red', label=f'$Y_0$ (Ref.)', linewidth=2)

# Add labels and title
plt.xlabel('Number of iterations')
plt.ylabel('Y')
plt.legend()
plt.grid()

plt.figure()
plt.plot(training_history[:NN,0], training_history[:NN,1], label='Loss value')

# Add labels and title
plt.xlabel('Number of batch iterations')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()


#%%
import random
plt.rcParams['figure.dpi'] = 100
P = 2**3
delta_t = 1/100
# Simulate the BSDE after training - MtM scenarios
samples = bsde.sample(P)
pay_off = np.maximum(samples[0][:,0,-1] - strike ,0)
simulations = bsde_solver.model.simulate_path(samples)     # Y
t = np.linspace(0,total_time,num_time_interval+1)
err_T = simulations[:,0,-1] - pay_off
#history_pred = bsde_solver.model.predict_step(samples)

color = ['blue', 'red', 'green', 'purple', 'orange', 'black', 'grey', 'magenta', 'cyan']

f_compensator = np.zeros((P,num_time_interval))     # NN_n (S_n)
NN_jump = np.zeros((P,num_time_interval)) 
f_jump = np.zeros((P,num_time_interval))
Z = np.zeros((P,num_time_interval)) 
one_net = False
for time in range(0, num_time_interval):
    assetprice = samples[0][:, 0, time]
    jumped_assetprice = samples[0][:, 0, time] * np.exp(samples[2][:,0,time])

    #price_approx[omega, time] = bsde_solver.model.subnet[time].call(np.reshape(assetprice, [1, 1]), training=False)
    #price_exact[omega, time] = merton_jump_call(assetprice, strike, total_time - time * delta_t, r, sigma, aver_jump, std_jump, lamb)
    if one_net:
        time_vector = np.ones(P)*time
        input_comp = np.array([time_vector, assetprice], dtype = np.float64).T
        f_compensator[:, time] = bsde_solver.model.subnet[0](input_comp, training=False)[:,0]
            
        #input_jump = np.reshape(np.array([time, assetprice, jumped_assetprice], dtype=np.float64), (P, 3))
        input_jump = np.array([time_vector, assetprice], dtype = np.float64).T
        NN_jump[:, time] = bsde_solver.model.subnetCompensator[0](input_jump,training=False)[:,0]
        
        input_Z = np.array([time_vector, assetprice], dtype = np.float64).T
        Z[:, time] = bsde_solver.model.subnetControl[0](input_Z, training=False)[:,0]

    else:
        f_compensator[:, time] = bsde_solver.model.subnet[0](np.reshape([samples[0][:, 0, time]], [P, 1]), training=False)[:,0]
        
        NN_jump[:, time] = bsde_solver.model.subnetCompensator[time](np.reshape([samples[0][:, 0, time], samples[0][:, 0, time]*np.exp(samples[2][:,0,time])], [P, 2]),training=False)[:,0] 
        #NN_jump[:, time] = bsde_solver.model.subnetCompensator[0](np.reshape([samples[0][:, 0, time]], [P, 1]),training=False)[:,0] 
        Z[:, time] = bsde_solver.model.subnetControl[time](np.reshape([assetprice], [P, 1]), training=False)[:,0]
    f_jump[:, time] = NN_jump[:, time] * (jumped_assetprice - assetprice)

FFT_price = np.zeros([5,num_time_interval+1])
for n in range(0,5):
    for i in range(0,num_time_interval+1):
        FFT_price[n,i] = CallPricingFFT(16,samples[0][n,0,i],strike,total_time - i *delta_t,r,0,C,G,M,Y)
compensated_jumps = f_jump - f_compensator * delta_t

plt.plot(NN_jump[0,:])
plt.plot(f_jump[0,:])

plt.figure()
plt.plot(t,simulations[:5, 0, :].T,color = 'red')
plt.plot(t[-1]*np.ones(5), pay_off[:5],'x', color='black')
plt.plot(t, samples[0][:5,0,:].T, '--', color = 'black')
plt.legend(['Y (Approx.)','Y (Ref.)'])
plt.grid()
plt.show()

n = 0

if strike < 0.0001:
    for omega in range(n,n+5):    
        plt.plot(t, simulations[omega, 0, :], color='red')
        plt.plot(t, samples[0][omega,0,:], '--', color='black')
        plt.plot(t[-1], pay_off[omega], 'x', color = 'black')
else:
    for omega in range(n,n+5):    
        plt.plot(t, simulations[omega, 0, :], color='red')
        plt.plot(t, FFT_price[omega,:], '--', color='black')
        plt.plot(t[0], mcprice, 'd', color='black')
        plt.plot(t[-1], pay_off[omega], 'x', color = 'black')

plt.grid()
plt.legend(['Y (Approx.)','Y (Ref.)'])



    