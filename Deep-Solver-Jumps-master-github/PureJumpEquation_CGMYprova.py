from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
import scipy.special as sc
# from scipy.fft import fft, ifft


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
       return tf.maximum(temp - self.dim * self.strike, 0)
    
    def getFsdeDiffusion(self, t, x):
        return self.sigma * x   
    
    
    


class CallOptionCGMY(Equation):
    def __init__(self, eqn_config):
        super(CallOptionCGMY, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init
        # self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.lamb = eqn_config.lamb
        self.aver_jump = eqn_config.aver_jump
        self.var_jump = eqn_config.var_jump     
        self.useExplict = True #whether to use explict formula to evaluate dyanamics of x
        self.eps = eqn_config.eps
        self.C = eqn_config.C
        self.G = eqn_config.G
        self.M = eqn_config.M
        self.Y = eqn_config.Y
        self.d = eqn_config.d

    def sample(self, num_sample):  #inserire qui il nuovo codice
        h = self.delta_t
        n = self.num_time_interval 
        eps = self.eps
        C = self.C
        G = self.G 
        M = self.M
        Y = self.Y 
        Npaths = num_sample
        
        a = 1-Y
        sigmaEpsSq = C/M**(2-Y)*sc.gamma(a+1)*sc.gammainc(a+1,M*eps)+C/G**(2-Y)*sc.gamma(a+1)*sc.gammainc(a+1,G*eps)    
        # print(sigmaEpsSq)    
        dW = normal.rvs(size=[Npaths,self.dim,n])*np.sqrt(h)  
        if self.dim==1:
            dW = np.expand_dims(dW,axis=0)
            dW = np.swapaxes(dW,0,1)

        diffu = np.cumsum(np.sqrt(sigmaEpsSq)*dW,axis=2)
        
        def CGMYPositiveOneSided(Npaths,h,n,eps, C,M,Y):
            a=1-Y
            bep=C/(M**a)*sc.gamma(a)*(sc.gammainc(a,M*eps) -1)
            # print(bep)
            lambdaep=C*np.exp(-M*eps)*eps**(-Y)/Y +bep*M/Y
            # print(lambdaep)
            DXep = np.zeros([Npaths,self.dim,n])
            jumpSizes = np.zeros([Npaths,self.dim,n])
            
            for k in range(0,Npaths): 
                N=np.random.poisson(lambdaep)
                U=np.random.uniform(0,1,N)
            
                DJep = np.zeros(n)
                J = np.zeros(N)          
                
                def CGMYJumpSize(eps,C,M,Y):               
                    W=np.random.uniform(0,1,1)  
                    V=np.random.uniform(0,1,1)  
                    X=eps*W**(-1/Y)
                    T= np.exp(M*(X-eps))               
                    while V*T>1: 
                        W=np.random.uniform(0,1,1)    
                        V=np.random.uniform(0,1,1)  
                        X=eps*W**(-1/Y)
                        T= np.exp(M*(X-eps))
                    return X       
                       
                for j in range(0,N): 
                    J[j]= CGMYJumpSize(eps,C,M,Y) 

                for i in range(0,n): 
                    DJep[i]=np.sum(J*(U>=i/n)*(U<(i+1)/n))

                jumpSizes[k,:,:] = DJep
                DXep[k,:,:]=bep*h + DJep
            return DXep, jumpSizes

        [DXpep, jumpSizesUp] = CGMYPositiveOneSided(Npaths,h,n,eps, C,M,Y)
        [DXnep, jumpSizesDown] = CGMYPositiveOneSided(Npaths,h,n,eps, C,G,Y)
        DXep=DXpep-DXnep+diffu
        jumpSizes = jumpSizesUp-jumpSizesDown 
        jumpIndicator = np.zeros([Npaths,self.dim,n])
        jumpIndicator[jumpSizes!=0] = 1
        
        m  = -C*sc.gamma(-Y)*((M-1)**Y-M**Y+(G+1)**Y-G**Y)
        # Discretizzazione completa nel tempo
        x_sample = np.zeros([Npaths,self.dim,n+1])
        x_sample[:,:,0]=np.ones([num_sample, self.dim]) * self.x_init  
        for i in range(1,n+1):
            x_sample[:,:,i] = x_sample[:,:,i-1]*np.exp( (self.r-self.d+m)*h + DXep[:,:,i-1] )
            
        return x_sample, jumpIndicator, jumpSizes, dW

    
    def f_tf(self, t, x, y, z):
        return -self.r * y
    
    def g_tf(self, t, x):
        return tf.maximum( x - self.strike, 0)
    
    def getFsdeDiffusion(self, t, x):
        eps = self.eps
        C = self.C
        G = self.G 
        M = self.M
        Y = self.Y        
        a = 1-Y
        sigmaEpsSq = C/M**(2-Y)*sc.gamma(a+1)*sc.gammainc(a+1,M*eps)+C/G**(2-Y)*sc.gamma(a+1)*sc.gammainc(a+1,G*eps)    
        # print(sigmaEpsSq) 
        return sigmaEpsSq * x  #il sigma deve essere il sigma calcolato con il CGMY cioè quello che in matlab è sigmaEpsSq dentro pathsimulation
    
    

    
