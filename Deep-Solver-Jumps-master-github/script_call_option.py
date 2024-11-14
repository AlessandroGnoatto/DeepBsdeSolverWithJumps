
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PureJumpSolver import BSDESolver
import PureJumpEquation as eqn
import munch
from scipy.stats import norm
import pandas as pd
import sys
import os
import itertools
from scipy.stats import multivariate_normal as normal
from scipy.optimize import minimize_scalar   
N = norm.cdf


# path:   C:\Users\mrcpa\Dropbox\Documenti\Marco\UNIVERSITA\RtdA_Verona\Ricerca\Deep Learning\20240627_Deep-Solver-master PureJump_ez_MP

plt.rcParams['figure.dpi'] = 300

 

if __name__ == "__main__":
    
    ## Specify if we want to train the model (and save it), or to load it   
    train_model = True # If this is False, the next is not checked 
    save_model = True
    
    dim = 1 #dimension of brownian motion
    P = 2**16 #number of outer Monte Carlo Loops
    batch_size = 2**5 #2048*8
    total_time = 1.
    num_time_interval = 40
    strike = 0.9
    lamb = 0.3
    r = 0.05
    sigma = 0.25
    aver_jump = 0.5
    var_jump = 0.25**2
    x_init = 1.0
    
    ## Specify the name of the directory where model and simulations lie
    path_dir = 'LocalRisk{}points'.format(num_time_interval)
    
    config = {  
                "eqn_config": {
                    "_comment": "a call contract",
                    "eqn_name": "CallOption",
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
                    "num_hiddens": [15, 15],
                    "lr_values": [7.5e-2, 5e-2, 5e-3, 5e-4, 1e-4],
                    "lr_boundaries": [1000,4000,6000, 15000], #                    "lr_boundaries": [1000,4000,10000, 15000],
                    "num_iterations": 10000,
                    "batch_size": batch_size,
                    "valid_size": 256,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "y_init_range": [0.24, 0.4],
                    "verbose": True
                }
                }
    

    
    '''
    "net_config": {
    "num_hiddens": [dim+30, dim+30],
    "lr_values": [5e-2, 5e-3, 5e-4, 1e-4],
    "lr_boundaries": [5000, 10000, 15000],
    "num_iterations": 20000,
    "batch_size": batch_size,
    "valid_size": 256,
    "logging_frequency": 100,
    "dtype": "float64",
    "y_init_range": [0.24, 0.4],
    "verbose": True
    '''
    config = munch.munchify(config) 
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    
    samples = bsde.sample(P)
    stock = samples[0]
    mcprice = np.exp(-r* total_time)*np.average(np.maximum(stock[:,0,-1] - strike,0))
    np.disp(mcprice)
    
    #apply algorithm 1
    bsde_solver = BSDESolver(config, bsde)

    training_history = bsde_solver.train()  
    simulations = bsde_solver.model.simulate_path(samples)
    
    
    # if train_model:

    #     ## Train the model
    #     training_history = bsde_solver.train()  
    #     simulations = bsde_solver.model.simulate_path(samples)
    #     # aba = bsde_solver.model.predict_step(samples)
    
    #     if save_model:
    
    #         # Create the directory
    #         try:
    #             os.mkdir(path_dir)
    #             print('Directory ', '\033[1m' + path_dir + '\033[0m', ' created')
    
    #         except OSError:
    #             print('Directory ', '\033[1m' + path_dir + '\033[0m', ' already existing: try another name')
    
    #         ## Save
    #         path = path_dir + '/Model1_MV{}points'.format(num_time_interval)
    #         bsde_solver.model.save_model(path)
    #         np.save(path_dir + '/training_history_{}.npy'.format(num_time_interval), training_history)
    # else:
    
    #     ## Load
    #     path = path_dir +  '/Model1_MV{}points'.format(num_time_interval)
    #     bsde_solver.model.load_model(path)
    #     training_history = np.load(path_dir + '/training_history_{}.npy'.format(num_time_interval))
    
    
    ####### SEMI-EXPLICIT SOLUTION
    
    
    def BS_CALL(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * N(d1) - K * np.exp(-r*T)* N(d2)
    
    def merton_jump_call(S, K, T, r, sigma, m , v, lam):
        p = 0
        for k in range(100):
            r_k = r - lam*(m-1) + (k*np.log(m) ) / T
            sigma_k = np.sqrt( sigma**2 + (k* v** 2) / T)
            k_fact = np.math.factorial(k)
            p += (np.exp(-m*lam*T) * (m*lam*T)**k / (k_fact))  * BS_CALL(S, K, T, r_k, sigma_k)
        
        return p 
    
   ###########
   
   
    # Simulate the BSDE after training - MtM scenarios
    samples = bsde.sample(P)
    simulations = bsde_solver.model.simulate_path(samples)     # Y
    
    
    #history_pred = bsde_solver.model.predict_step(samples)
    num_paths = 4
    price_exact = np.zeros((num_paths,num_time_interval+1)) 
    price_exact[:,num_time_interval] = np.maximum(samples[0][0:num_paths,0,num_time_interval]-strike,0)      # formula semi-analitica
    price_approx = np.zeros((num_paths,num_time_interval))     # NN_n (S_n)
    
    delta_t = total_time / num_time_interval 
    std_jump = np.sqrt(var_jump)
    
    '''
    fig = plt.figure()
    ax = plt.gca()
    
    from matplotlib.pyplot import cm
    color = cm.rainbow(np.linspace(0, 1, num_paths))
    
    for omega in range(0,num_paths):
        for time in range(0,num_time_interval):
          assetprice = samples[0][omega,0,time]
          #price_approx[omega,time] = bsde_solver.model.subnet[time].call(np.reshape(assetprice, [1, 1]),training=False)
          price_exact[omega,time] = merton_jump_call(assetprice, strike, total_time-time*delta_t, r, sigma, aver_jump , std_jump, lamb)
          #price_exact[omega,time] = merton_jump_call(assetprice, strike, total_time-time*delta_t, r, sigma, np.exp(aver_jump + std_jump**2/2) , std_jump, lamb)

        #color = next(ax._get_lines.prop_cycler)['color']
        cc = color[omega]
        plt.plot(price_exact[omega,:], '--',color = cc)
        plt.plot(simulations[omega,0,:], color = cc)
        #plt.plot(price_approx[omega,:], color = color)
    
    
    fig = plt.figure()
    for omega in range(0,num_paths):
        plt.plot(samples[0][omega,0,:])
 '''       

    
#%%
# Create the plot with a specific figure size
NN = np.int(15000/100)
plt.figure(figsize=(4, 6))  # Adjust the dimensions to match the aspect ratio of your image

plt.plot(training_history[:NN,0], training_history[:NN,2], label=f'$Y_0$ (Approx.)')
plt.plot(training_history[:NN,0], mcprice * np.ones(len(training_history[:NN,0])), '--', color='red', label=f'$Y_0$ (Analytic)', linewidth=2)

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
plt.plot(training_history[:NN,0], mcprice * np.ones(len(training_history[:NN,0])), '--', color='red', label=f'$Y_0$ (Analytic)', linewidth=2)

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
P = 2**16
# Simulate the BSDE after training - MtM scenarios
samples = bsde.sample(P)
pay_off = np.maximum(samples[0][:,0,-1] - strike ,0)
simulations = bsde_solver.model.simulate_path(samples)     # Y

#history_pred = bsde_solver.model.predict_step(samples)

color = ['blue', 'red', 'green', 'purple', 'orange', 'black', 'grey', 'magenta', 'cyan']

price_exact = np.zeros((P,num_time_interval+1))
price_exact[:,num_time_interval] = np.maximum(samples[0][:,0,num_time_interval]-strike,0)      # formula semi-analitica
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
    price_exact[:,time] = merton_jump_call(assetprice, strike, total_time-time*delta_t, r, sigma, np.exp(aver_jump + std_jump**2/2) , std_jump, lamb)
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
  
compensated_jumps = f_jump - f_compensator * delta_t
MSE_t = np.mean((price_exact-simulations[:,0,:])**2, 0)


#%%    
n = random.randint(0, P) - num_paths
#n = 0
num_paths = 5
t = np.linspace(0, 1, num_time_interval+1)
pay_off = np.maximum(samples[0][:, 0, -1] - strike, 0)
terminal_error = simulations[:, 0, -1] - pay_off
sorted_indices = np.argsort(np.abs(terminal_error))
top_five_indices = sorted_indices[-5:]
top_five_indices_desc = sorted_indices[-1:-6:-1]

#n = top_five_indices[0]
'''    
for omega in range(n,n+num_paths):    
    plt.plot(price_exact[omega, :], color=color[omega-n], label='Y (exact)' if omega == 0 else "")
    plt.plot(time+1, pay_off[omega], 'x', color = color[omega-n])
    plt.plot(simulations[omega, 0, :],'--', color=color[omega-n], label='Y (EM-approx)' if omega == 0 else "")

    # Ensure legend is added only once per figure
    if omega-n == 0:
        plt.legend()
    plt.grid()
'''

plt.plot(np.mean(samples[0][:,0,:],0))
plt.plot(np.mean(simulations[:,0,:],0))

plt.figure()
for omega in range(5):    
    plt.plot(t, simulations[top_five_indices[omega], 0, :], color='red')
    plt.plot(t, price_exact[top_five_indices[omega], :], '--', color='black')
    plt.plot(t[-1], pay_off[top_five_indices[omega]], 'x', color = 'black')

plt.grid()
plt.legend(['Y (Approx.)','Y (Analytic)'])
plt.figure()

for omega in range(n,n+num_paths):    
    plt.plot(t, simulations[omega, 0, :], color='red')
    plt.plot(t, price_exact[omega, :], '--', color='black')
    plt.plot(t[-1], pay_off[omega], 'x', color = 'black')

plt.grid()
plt.legend(['Y (Approx.)','Y (Analytic)'])


#%%    

plt.plot(f_compensator[n,:])
plt.plot(f_jump[n,:])
plt.plot(Z[n,:],':')
plt.plot(price_exact[n,:],'--')
plt.plot(samples[0][n,0,:], '.-')
plt.plot(samples[3][n,0,:], 'o-')
#plt.plot(simulations[n,0,:])

    

#%%
n = random.randint(0, P)-1
y = np.zeros((P,num_time_interval+1))
y[:,0] = simulations[n,0,0]
#k = np.argmax(np.abs(f_jump[n,:]))
for i in range(num_time_interval):
    y[:,i+1] = y[:,i] + delta_t*r*y[:,i] + compensated_jumps[:,i] + Z[:,i]*sigma*samples[0][:,0,i]*samples[3][:,0,i]
    

plt.plot(samples[0][n,0,:])
plt.figure()
#plt.plot(y[n,:], '.-')
plt.plot(price_exact[n,:], 'k')
#plt.plot(simulations[n,0,:k+2],'--')
plt.plot(simulations[n,0,:],'--')
plt.plot(f_jump[n,:],'.')



 


#%%
import random

# Identifying different jump indices
index_0_jumps = np.argwhere(np.sum(samples[1][:, 0, :], 1) == 0)[:, 0]
index_jumps = np.argwhere(np.sum(samples[1][:, 0, :], 1) > 0)[:, 0]
index_1_jump = np.argwhere(np.sum(samples[1][:, 0, :], 1) == 1)[:, 0]
index_2_jumps = np.argwhere(np.sum(samples[1][:, 0, :], 1) == 2)[:, 0]
index_3_jumps = np.argwhere(np.sum(samples[1][:, 0, :], 1) == 3)[:, 0]

# Calculate payoff and terminal error
pay_off = np.maximum(samples[0][:, 0, -1] - strike, 0)
terminal_error = simulations[:, 0, -1] - pay_off
terminal_error_y = y[:, -1] - pay_off


# Plot 1: Histogram of terminal errors for all samples
plt.hist(terminal_error, 50)
plt.title('Histogram of Terminal Errors (All Samples)')
plt.xlabel('Terminal Error')
plt.ylabel('Frequency')

# Plot 2: Histogram for cases with no jumps
plt.figure()
plt.hist(terminal_error[index_0_jumps], 50)
plt.title('Histogram of Terminal Errors (No Jumps)')
plt.xlabel('Terminal Error')
plt.ylabel('Frequency')

# Plot 3: Histogram for cases with 1 jump
plt.figure()
plt.hist(terminal_error[index_1_jump], 50)
plt.title('Histogram of Terminal Errors (1 Jump)')
plt.xlabel('Terminal Error')
plt.ylabel('Frequency')

# Plot 4: Histogram for cases with 2 jumps
plt.figure()
plt.hist(terminal_error[index_2_jumps], 50)
plt.title('Histogram of Terminal Errors (2 Jumps)')
plt.xlabel('Terminal Error')
plt.ylabel('Frequency')

# Plot 5: Histogram for cases with 3 jumps
plt.figure()
plt.hist(terminal_error[index_3_jumps], 50)
plt.title('Histogram of Terminal Errors (3 Jumps)')
plt.xlabel('Terminal Error')
plt.ylabel('Frequency')

plt.show()  # Ensure all plots are displayed
'''
# Plot 1: Histogram of terminal errors for all samples
plt.hist(terminal_error_y, 50)
plt.title('Histogram of Terminal Errors (All Samples)')
plt.xlabel('Terminal Error')
plt.ylabel('Frequency')

# Plot 2: Histogram for cases with no jumps
plt.figure()
plt.hist(terminal_error_y[index_0_jumps], 50)
plt.title('Histogram of Terminal Errors (No Jumps)')
plt.xlabel('Terminal Error')
plt.ylabel('Frequency')

# Plot 3: Histogram for cases with 1 jump
plt.figure()
plt.hist(terminal_error_y[index_1_jump], 50)
plt.title('Histogram of Terminal Errors (1 Jump)')
plt.xlabel('Terminal Error')
plt.ylabel('Frequency')

# Plot 4: Histogram for cases with 2 jumps
plt.figure()
plt.hist(terminal_error_y[index_2_jumps], 50)
plt.title('Histogram of Terminal Errors (2 Jumps)')
plt.xlabel('Terminal Error')
plt.ylabel('Frequency')

# Plot 5: Histogram for cases with 3 jumps
plt.figure()
plt.hist(terminal_error_y[index_3_jumps], 50)
plt.title('Histogram of Terminal Errors (3 Jumps)')
plt.xlabel('Terminal Error')
plt.ylabel('Frequency')

plt.show()  # Ensure all plots are displayed
'''
n = random.randint(0, P)

plt.plot(NN_jump[n:n+5,:].T)
plt.grid()

plt.figure()
plt.plot(f_compensator[n:n+5,:].T)
plt.grid()

plt.figure()
plt.plot(f_jump[n:n+5,:].T)
plt.grid()
print('E|Y_N-g(X_N)|^2: ', np.mean(terminal_error**2))
print('E[compensated_jumps]^2: ', np.sum(np.mean(compensated_jumps,0)**2))
print('loss: ', 10*np.mean(terminal_error**2) + np.sum(np.mean(compensated_jumps,0)**2))
print('||err_Y||^2: ', np.mean(MSE_t))





   