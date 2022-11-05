import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PureJumpSolver import BSDESolver
import PureJumpEquation_CGMYprova as eqn
import munch
from scipy.stats import norm
import pandas as pd
import scipy.special as sc
from scipy.fft import fft, ifft


plt.rcParams['figure.dpi'] = 300

 

if __name__ == "__main__":
    dim = 1 #dimension of brownian motion
    P = 2048*8 #number of outer Monte Carlo Loops
    batch_size = 64
    total_time = 1.0
    num_time_interval = 100
    strike = 90
    lamb = 0.3
    r = 0.0
    # sigma = 0.25
    aver_jump = 0.5
    var_jump = 0.25
    x_init = 100
    C=0.1
    G=1.4
    M=1.3
    Y=0.5
    eps = 0.0001
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
                    # "sigma":sigma,
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
                    "num_hiddens": [dim+20,dim+20],
                    "lr_values": [5e-2, 5e-3],
                    "lr_boundaries": [4000],
                    "num_iterations": 8000,
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
    
    # #apply algorithm 1
    # bsde_solver = BSDESolver(config, bsde)
    # training_history = bsde_solver.train()  
   
    #Simulate the BSDE after training - MtM scenarios
    samples = bsde.sample(P)
    #simulations = bsde_solver.model.simulate_path(samples)

stock = samples[0]

m  = -C*sc.gamma(-Y)*((M-1)**Y-M**Y+(G+1)**Y-G**Y)

DXep = np.zeros([P,dim,num_time_interval])
for i in range(0,num_time_interval):
    DXep[:,:,i] = np.log(stock[:,:,i+1]/stock[:,:,i])-(r-d+m)*(total_time/num_time_interval)
  
Xep = np.cumsum(DXep,axis=2)    

def cf_cgmy(u,T,r,d,C,G,M,Y):
    m = 0
    tmp = C*T*sc.gamma(-Y)*((M-1j*u)**Y-M**Y+(G+1j*u)**Y-G**Y)   
    y = np.exp(1j*u*( (r-d+m)*T) + tmp)
    return y

def getProbabilityDensity(n,T,r,d,C,G,M,Y):
    N = 2**n;
    eta = 0.1;
    lambda_ = (2*np.pi)/(N*eta)
    u = np.arange(0,(N)*eta,eta)
    b=(N*lambda_)/2    
    psi = np.zeros([len(u)],dtype=np.complex_)
   
    for j in range(0,len(u)):
        psi[j] = cf_cgmy(u[j],T,r,d,C,G,M,Y)
    
    cf = psi*np.exp(1j*b*(np.transpose(u)))*eta

    jvec = np.transpose(range(1,N+1))
    cf = (cf/3)*(3+(-1)**jvec-((jvec-1)==0))
    ft = fft(cf,N)
    kv = np.arange(-b,(N)*lambda_-b,lambda_)
    kv = np.transpose(kv)
    
    density = np.real(ft/np.pi)
    return density, kv

# True density via FFT
Bound = 1.5
[density, kv] = getProbabilityDensity(12,total_time,r,d,C,G,M,Y)
I = np.where((kv >= -Bound) & (kv <= Bound))


fig = plt.figure()
plt.plot(kv[I], density[I])
myhist = plt.hist(Xep[:,:,-1]*(np.abs(Xep[:,:,-1])<Bound), bins = 100, density=True)
plt.show()


# MartingaleCheck
MartingaleCheck = np.mean(stock,0)

fig2 = plt.figure()
plt.plot(range(0,101), MartingaleCheck[0,:])
plt.show()

    
    
    
    
def cf_cgmy_noexp(u,lnS,T,r,d,C,G,M,Y):
    m = -C*sc.gamma(-Y)*((M-1)**Y-M**Y+(G+1)**Y-G**Y)
    tmp = C*T*sc.gamma(-Y)*((M-1j*u)**Y-M**Y+(G+1j*u)**Y-G**Y)
    y = 1j*u*(lnS + (r-d+m)*T) + tmp
    return y

def psi(vj,optAlpha,lnS,T,r,d,C,G,M,Y):
    ret = np.exp(cf_cgmy_noexp(vj-(optAlpha+1)*1j,lnS,T,r,d,C,G,M,Y)) / (optAlpha**2 + optAlpha - vj**2 + 1j * (2 * optAlpha + 1)* vj)
    return ret


def CallPricingFFT(n,S0,K,T,r,d,C,G,M,Y):
    lnS = np.log(S0)
    lnK = np.log(K)

    # optAlpha = optimalAlpha(model,lnS,lnK,T,r,d,varargin{:});
    optAlpha = 0.75

    DiscountFactor = np.exp(-r*T)
    #-------------------------
    #--- FFT Option Pricing --
    #-------------------------
    # from: Option Valuation Using the Fast Fourier Transform, 
    #       Peter Carr, March 1999, pp 10-11
    #-------------------------
    
    # predefined parameters
    FFT_N = 2**n                               # must be a power of two (2^14)
    FFT_eta = 0.05                             # spacing of psi integrand
    
    # effective upper limit for integration (18)
    # uplim = FFT_N * FFT_eta;
    
    FFT_lambda = (2 * np.pi) / (FFT_N * FFT_eta);  #spacing for log strike output (23)
    FFT_b = (FFT_N * FFT_lambda) / 2;           # (20)
    
    uvec = np.arange(1,FFT_N+1)
    #log strike levels ranging from lnS-b to lnS+b
    ku = - FFT_b + FFT_lambda * (uvec - 1)     #(19)
    
    jvec = np.arange(1,FFT_N+1)
    vj = (jvec-1) * FFT_eta
    
    #applying FFT
    tmp = DiscountFactor * psi(vj,optAlpha,lnS,T,r,d,C,G,M,Y) * np.exp(1j * vj * (FFT_b)) * FFT_eta
    tmp = (tmp / 3) * (3 + (-1)**jvec - ((jvec - 1) == 0) );  #applying simpson's rule
    cpvec = np.real(np.exp(-optAlpha * ku) * fft(tmp) / np.pi)      #call price vector resulting in equation 24
    
    indexOfStrike = np.floor((lnK + FFT_b)/FFT_lambda + 1)
    iset = np.arange(np.amax(indexOfStrike)+1,np.amin(indexOfStrike)-2,-1).astype(int)
    xp = ku[iset]
    yp = cpvec[iset]
    call_price_fft = np.real(np.interp(lnK, xp, yp))
    return call_price_fft

###########################################

print("Price according to the FFT")
priceFFT = CallPricingFFT(8,x_init,strike,total_time,r,d,C,G,M,Y);
print(priceFFT)

print("Price according to Monte Carlo")
priceMC = np.mean(np.maximum(stock[:,:,-1]- strike,0))*np.exp(-r*total_time);
print(priceMC)