# %reset -f
import numpy as np
import scipy.special as sc
from scipy.stats import multivariate_normal as normal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
# from scipy.stats import norm
# import pandas as pd

C=0.1; G=1.4; M=1.3; Y=0.5; 

h = 0.01
n = 100
eps = 0.0001
Npaths = int(1e3)

r = 0; # tasso risk free
d = 0; # dividend yield
S0 = 100;
T = 1;

def CGMYpathSimulation(Npaths,h,n,epsilon, C,G,M,Y):
    a = 1-Y
    sigmaEpsSq = C/M**(2-Y)*sc.gamma(a+1)*sc.gammainc(a+1,M*eps)+C/G**(2-Y)*sc.gamma(a+1)*sc.gammainc(a+1,G*eps)    
    # print(sigmaEpsSq)    
    dW = normal.rvs(size=[Npaths,n])*np.sqrt(h)    
    diffu = np.cumsum(np.sqrt(sigmaEpsSq)*dW,axis=1)
    
    def CGMYPositiveOneSided(Npaths,h,n,eps, C,M,Y):
        a=1-Y
        bep=C/(M**a)*sc.gamma(a)*(sc.gammainc(a,M*eps) -1)
        # print(bep)
        lambdaep=C*np.exp(-M*eps)*eps**(-Y)/Y +bep*M/Y
        # print(lambdaep)
        DXep = np.zeros([Npaths,n])
        jumpSizes = np.zeros([Npaths,n])
        
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

            jumpSizes[k,:] = DJep
            DXep[k,:]=bep*h + DJep
        return DXep, jumpSizes

    [DXpep, jumpSizesUp] = CGMYPositiveOneSided(Npaths,h,n,eps, C,M,Y)
    [DXnep, jumpSizesDown] = CGMYPositiveOneSided(Npaths,h,n,eps, C,G,Y)
    DXep=DXpep-DXnep+diffu
    jumpSizes = jumpSizesUp-jumpSizesDown 
    jumpIndicator = np.zeros([Npaths,n])
    jumpIndicator[jumpSizes!=0] = 1
    return DXep,DXpep, DXnep, jumpIndicator, jumpSizes, jumpSizesUp, jumpSizesDown, dW, diffu

[DXep,DXpep, DXnep, jumpIndicator, jumpSizes, jumpSizesUp, jumpSizesDown, dW, diffu] = CGMYpathSimulation(Npaths,h,n,eps, C,G,M,Y)
Xep = np.cumsum(DXep,axis=1)


# m è il compensatore esponenziale che rende l'asset scontato martingala.
# E' l'analogo di -0.5 sigma^2 per il modello Black-Scholes.
m  = -C*sc.gamma(-Y)*((M-1)**Y-M**Y+(G+1)**Y-G**Y)

# Discretizzazione completa nel tempo
AssetSimulation = np.zeros([Npaths,n+1])
AssetSimulation[:,0]=S0;
deltaT = 1/n;

for i in range(1,n+1):
    AssetSimulation[:,i] = AssetSimulation[:,i-1]*np.exp( (r-d+m)*deltaT + DXep[:,i-1] )

