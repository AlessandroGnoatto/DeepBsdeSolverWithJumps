import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PureJumpSolver import BSDESolver
import PureJumpEquation as eqn
import munch
from scipy.stats import norm
import pandas as pd

plt.rcParams['figure.dpi'] = 300

 

if __name__ == "__main__":
    dim = 5 #dimension of brownian motion
    P = 2**12         #number of outer Monte Carlo Loops
    batch_size = 2**10
    total_time = 1.0
    num_time_interval = 40
    strike = 0.9
    lamb = 0.3
    r = 0.05
    sigma = 0.25
    aver_jump = 0.5
    var_jump = 0.25**2
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
                    "num_hiddens": [15,15],
                    "lr_values": [1.0e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4],
                    "lr_boundaries": [1000,3000,5000, 7000, 10000],#d=5#"lr_boundaries": [1000,4000, 8000, 12000, 15000],#d=25#"lr_boundaries": [5000,10000,15000, 20000, 23000], #                   
                    "num_iterations": 10000,
                    "batch_size": batch_size,
                    "valid_size": 256,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "y_init_range": [0.24, 0.4],
                    "verbose": True
                }
                }
    config = munch.munchify(config) 
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    
    #Monte Carlo Price
    samples = bsde.sample(P)
    stock = samples[0]
    #mcprice = np.exp(-r* total_time)*np.average(np.maximum(np.sum(stock[:,:,-1],1) - dim * strike,0))
    #payoff =  np.maximum(np.sum(stock[:,:,-1],1) - dim * strike,0)
    mcprice = np.exp(-r* total_time)*np.average(np.maximum(1/dim * np.sum(stock[:,:,-1],1) - strike,0))
    payoff =  np.maximum(1/dim * np.sum(stock[:,:,-1],1) - strike,0)
    np.disp(mcprice)  
    
    #apply algorithm 1
    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()  
   
    #Simulate the BSDE after training - MtM scenarios
    simulations = bsde_solver.model.simulate_path(samples)
    
    fig = plt.figure()
    plt.plot(simulations[0:10,0,:].T)
    plt.xlabel('Time step')
    plt.ylabel('Y')
    plt.show()
    
    

 
    # mcprice = 14.426902897231209
    
    # np.save("training_history_dim_"+str(dim),training_history)
    # np.save("mcprice_dim_"+str(dim),mcprice)
    
    
    
    
#%%
#Simulate the BSDE after training - MtM scenarios
samples = bsde.sample(P*2**4)
#samples = bsde.sample(2**6)

simulations = bsde_solver.model.simulate_path(samples)
    
fig = plt.figure()
plt.plot(simulations[0:10,0,:].T)
plt.xlabel('Time step')
plt.ylabel('Y')
plt.show()
    
    

# Monte Carlo Price
stock = samples[0]
#mcprice = np.exp(-r * total_time)*np.average(np.maximum(np.sum(stock[:,:,-1],1) - dim * strike,0))
mcprice = np.exp(-r * total_time)*np.average(np.maximum(1 / dim * np.sum(stock[:,:,-1],1) -  strike,0))

#payoff =  np.maximum(np.sum(stock[:,:,-1],1) - dim * strike,0)
payoff =  np.maximum(1 / dim * np.sum(stock[:,:,-1],1) - strike,0)

np.disp(mcprice)  
    
    # np.save("training_history_dim_"+str(dim),training_history)
    # np.save("mcprice_dim_"+str(dim),mcprice)
    
#%%
# Create the plot with a specific figure size
NN = np.int(10000/100)
plt.figure(figsize=(4, 6))  # Adjust the dimensions to match the aspect ratio of your image

plt.plot(training_history[:NN,0], training_history[:NN,2], label=f'$Y_0$ (Approx.)')
plt.plot(training_history[:NN,0], mcprice * np.ones(len(training_history[:NN,0])), '--', color='red', label=f'$Y_0$ (Ref.)', linewidth=2)

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
from scipy.stats import multivariate_normal as normal
num_sample = 2**12
t = np.linspace(0, total_time, num_time_interval + 1)
delta_t = total_time/num_time_interval
# simulazione del Browniano
start = 0
Y_MC_5 = np.zeros([5,num_time_interval + 1])
for nn in range(start, start+5):
    X_path = stock[nn,:,:]
    Y_MC = np.zeros(num_time_interval + 1)
    Y_MC[0] =  np.exp(-r * total_time) * np.mean(np.maximum(np.sum(stock[:,:,-1],1) - dim * strike,0))
    for n in range(0,num_time_interval):    
        dw_sample = normal.rvs(size=[num_sample,dim,num_time_interval - n]) * np.sqrt(delta_t)
        if num_time_interval - n < 2:
            dw_sample = np.expand_dims(dw_sample,axis=2)               
        if dim==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
            dw_sample = np.swapaxes(dw_sample,0,1)
                
        # simulazione dei salti
            
        eta = normal.rvs(mean=0.0 ,cov=1.0, size = [num_sample, dim, num_time_interval - n])
        eta = np.reshape(eta,[num_sample, dim, num_time_interval - n])

        Poisson = np.random.poisson(lamb * delta_t, [num_sample, dim , num_time_interval - n])

        jumps = np.multiply(Poisson, aver_jump) + np.sqrt(var_jump)*np.multiply(np.sqrt(Poisson),eta)
                
                
        # traiettorie forward
        x_sample = np.ones([num_sample, dim, num_time_interval + 1 - n]) 
        x_sample[:, :, 0] = np.ones([num_sample, dim]) * X_path[:,n]  
        
        factor = np.exp((r-(sigma**2)/2)*delta_t - lamb*(np.exp(aver_jump + 0.5*var_jump)-1)*delta_t)
        
        for i in range(num_time_interval - n): 
            x_sample[:, :, i + 1] = (factor * np.exp(sigma * dw_sample[:, :, i]) * np.exp(jumps[:, :, i])) * x_sample[:, :, i]

        #Y_MC[n] = np.exp(-r * (total_time - n*delta_t) ) * np.mean(np.maximum(np.sum(x_sample[:,:,-1],1) - dim * strike,0))
        Y_MC[n] = np.exp(-r * (total_time - n*delta_t) ) * np.mean(np.maximum(1 / dim * np.sum(x_sample[:,:,-1],1) - strike,0))

        if n == 0:
            plt.plot(np.mean(np.mean(x_sample,0),0))
            plt.plot(np.exp(r*t))
    Y_MC[-1] = payoff[nn]
    Y_MC_5[nn-start,:] = Y_MC

plt.figure()
for nn in range(start, start+5):
    plt.plot(t, simulations[nn,0,:], color='red')
    plt.plot(t, Y_MC_5[nn-start,:], '--', color='black')
    plt.plot(total_time, payoff[nn], 'x', color = 'black')
plt.grid()
plt.legend(['Y (Approx.)','Y (Ref.)'])



