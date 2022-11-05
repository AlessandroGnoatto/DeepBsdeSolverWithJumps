import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
num_sample = 2048
dim = 10
total_time = 1
num_time_interval = 40
delta_t = total_time/num_time_interval
sqrt_delta_t = np.sqrt(delta_t)
lamb = 1.4
aver_jump = 0.5
var_jump = 0.25
x_init = 1.0
r = 0.01
sigma = 0.25


dw_sample = normal.rvs(size=[num_sample,     
                              dim,
                              num_time_interval]) * sqrt_delta_t
       
if dim==1:
    dw_sample = np.expand_dims(dw_sample,axis=0)
    dw_sample = np.swapaxes(dw_sample,0,1)

# simulazione dei salti

eta = normal.rvs(mean=0.0 ,cov=1.0, size = [num_sample, dim, num_time_interval])
eta = np.reshape(eta,[num_sample, dim, num_time_interval])
Poisson = np.random.poisson(lamb * delta_t, [num_sample, dim , num_time_interval])
jumps = np.multiply(Poisson, aver_jump) + np.sqrt(var_jump)*np.multiply(np.sqrt(Poisson),eta)


# traiettorie forward

x_sample = np.zeros([num_sample, dim, num_time_interval + 1]) 
x_sample[:, :, 0] = np.ones([num_sample, dim]) * x_init  



factor = np.exp((r-(sigma**2)/2)*delta_t - lamb*(np.exp(aver_jump + 0.5*var_jump)-1)*delta_t)
for i in range(num_time_interval): 
    x_sample[:, :, i + 1] = (factor * np.exp(sigma * dw_sample[:, :, i]) * np.exp(jumps[:, :, i])) * x_sample[:, :, i]


u = np.zeros([num_sample, dim, num_time_interval]) 
for t in range(0, num_time_interval-1):       
    u[: ,:, t] = x_sample[:, :, t + 1]* np.exp(jumps[:, :, t])
    
    
data = x_sample, Poisson, jumps, dw_sample 

temp = tf.reduce_sum(x_sample, 1,keepdims=True)





