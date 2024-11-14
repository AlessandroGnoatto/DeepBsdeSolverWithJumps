from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal


class Expectation(Equation):
    def __init__(self,eqn_config):
        super(Expectation, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init  # initial value of x, the underlying
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.useExplict = True #whether to use explict formula to evaluate dyanamics of x
        self.lamb = eqn_config.lamb
        self.aver_jump = eqn_config.aver_jump
        self.var_jump = eqn_config.var_jump
        self.u = eqn_config.u
        
        
    def sample(self, num_sample):  
        
        # simulazione del Browniano
        
        dw_sample = normal.rvs(size=[num_sample,     
                                      self.dim,
                                      self.num_time_interval]) * self.sqrt_delta_t
               
        if self.dim==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
            dw_sample = np.swapaxes(dw_sample,0,1)
        
        # simulazione dei salti
    
        eta = normal.rvs(mean=0.0 ,cov=1.0, size = [num_sample, self.dim, self.num_time_interval])
        eta = np.reshape(eta,[num_sample, self.dim, self.num_time_interval])
        Poisson = np.random.poisson(self.lamb * self.delta_t, [num_sample, self.dim , self.num_time_interval])
        jumps = np.multiply(Poisson, self.aver_jump) + np.sqrt(self.var_jump)*np.multiply(np.sqrt(Poisson),eta)
        
        
        # traiettorie forward

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict: 
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t - self.lamb*(np.exp(self.aver_jump + 0.5*self.var_jump)-1)*self.delta_t)
            for i in range(self.num_time_interval): 
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i]) * np.exp(jumps[:, :, i])) * x_sample[:, :, i]
        return x_sample, Poisson, jumps, dw_sample  
  
    def f_tf(self, t, x, y, z):
         return 0
   
    def g_tf(self, t, x):
        return tf.math.exp(self.u*tf.math.log(x))

    def getFsdeDiffusion(self, t, x):
        return 0


class PricingForward(Equation):
    def __init__(self,eqn_config):
        super(PricingForward, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init  # initial value of x, the underlying
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.useExplict = True #whether to use explict formula to evaluate dyanamics of x
        self.lamb = eqn_config.lamb
        self.aver_jump = eqn_config.aver_jump
        self.var_jump = eqn_config.var_jump
        self.u = eqn_config.u
        
        
    def sample(self, num_sample):  
        
        # simulazione del Browniano
        
        dw_sample = normal.rvs(size=[num_sample,     
                                      self.dim,
                                      self.num_time_interval]) * self.sqrt_delta_t
               
        if self.dim==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
            dw_sample = np.swapaxes(dw_sample,0,1)
        
        # simulazione dei salti
    
        eta = normal.rvs(mean=0.0 ,cov=1.0, size = [num_sample, self.dim, self.num_time_interval])
        eta = np.reshape(eta,[num_sample, self.dim, self.num_time_interval])
        Poisson = np.random.poisson(self.lamb * self.delta_t, [num_sample, self.dim , self.num_time_interval])
        jumps = np.multiply(Poisson, self.aver_jump) + np.sqrt(self.var_jump)*np.multiply(np.sqrt(Poisson),eta)
        
        
        # traiettorie forward

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict: 
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t - self.lamb*(np.exp(self.aver_jump + 0.5*self.var_jump)-1)*self.delta_t)
            for i in range(self.num_time_interval): 
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i]) * np.exp(jumps[:, :, i])) * x_sample[:, :, i]
        return x_sample, Poisson, jumps, dw_sample 
  
    def f_tf(self, t, x, y, z):
         return -self.r * y
   
    def g_tf(self, t, x):
        return x - self.strike

    def getFsdeDiffusion(self, t, x):
        return self.sigma * x

    
class CallOption(Equation):
    def __init__(self, eqn_config):
        super(CallOption, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.lamb = eqn_config.lamb
        self.aver_jump = eqn_config.aver_jump
        self.var_jump = eqn_config.var_jump     
        self.useExplict = True #whether to use explict formula to evaluate dyanamics of x

    def sample(self, num_sample):  
        
        # simulazione del Browniano
        
        dw_sample = normal.rvs(size=[num_sample,     
                                      self.dim,
                                      self.num_time_interval]) * self.sqrt_delta_t
               
        if self.dim==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
            dw_sample = np.swapaxes(dw_sample,0,1)
        
        # simulazione dei salti
    
        eta = normal.rvs(mean=0.0 ,cov=1.0, size = [num_sample, self.dim, self.num_time_interval])
        eta = np.reshape(eta,[num_sample, self.dim, self.num_time_interval])
        Poisson = np.random.poisson(self.lamb * self.delta_t, [num_sample, self.dim , self.num_time_interval])
        jumps = np.multiply(Poisson, self.aver_jump) + np.sqrt(self.var_jump)*np.multiply(np.sqrt(Poisson),eta)
        
        
        # traiettorie forward

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict: 
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t - self.lamb*(np.exp(self.aver_jump + 0.5*self.var_jump)-1)*self.delta_t)
            for i in range(self.num_time_interval): 
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i]) * np.exp(jumps[:, :, i])) * x_sample[:, :, i]
        return x_sample, Poisson, jumps, dw_sample  
    
    def f_tf(self, t, x, y, z):
        return -self.r * y
    
    def g_tf(self, t, x):
        return tf.maximum( x - self.strike, 0)
    
    def getFsdeDiffusion(self, t, x):
        return self.sigma * x
    
    
class BasketOption(Equation):
    def __init__(self, eqn_config):
        super(BasketOption, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.lamb = eqn_config.lamb
        self.aver_jump = eqn_config.aver_jump
        self.var_jump = eqn_config.var_jump     
        self.useExplict = True #whether to use explict formula to evaluate dyanamics of x

    def sample(self, num_sample):  
        
        # simulazione del Browniano
        
        dw_sample = normal.rvs(size=[num_sample,     
                                      self.dim,
                                      self.num_time_interval]) * self.sqrt_delta_t
               
        if self.dim==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
            dw_sample = np.swapaxes(dw_sample,0,1)
        
        # simulazione dei salti
    
        eta = normal.rvs(mean=0.0 ,cov=1.0, size = [num_sample, self.dim, self.num_time_interval])
        eta = np.reshape(eta,[num_sample, self.dim, self.num_time_interval])
        Poisson = np.random.poisson(self.lamb * self.delta_t, [num_sample, self.dim , self.num_time_interval])
        jumps = np.multiply(Poisson, self.aver_jump) + np.sqrt(self.var_jump)*np.multiply(np.sqrt(Poisson),eta)
        
        
        # traiettorie forward

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict: 
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t - self.lamb*(np.exp(self.aver_jump + 0.5*self.var_jump)-1)*self.delta_t)
            for i in range(self.num_time_interval): 
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i]) * np.exp(jumps[:, :, i])) * x_sample[:, :, i]
        return x_sample, Poisson, jumps, dw_sample  
    
    def f_tf(self, t, x, y, z):
        return -self.r * y
    
    def g_tf(self, t, x):
       temp = tf.reduce_sum(x, 1,keepdims=True)
       #return tf.maximum(temp - self.dim * self.strike, 0)
       return tf.maximum(1 / self.dim * temp - self.strike, 0)

    
    def getFsdeDiffusion(self, t, x):
        return self.sigma * x   
    

    
