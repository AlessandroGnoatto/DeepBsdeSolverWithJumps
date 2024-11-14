# %reset -f
import numpy as np
import scipy.special as sc
from scipy.stats import multivariate_normal as normal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import time
# from scipy.stats import norm
# import pandas as pd

plt.rcParams['figure.dpi'] = 300

C=0.1; G=3.5; M=10; Y=0.5; 
C=0.15
G=13.
M=14.
Y=0.6
eps = 0.00001

a = 1 - Y
sigmaEpsSq0 = C/M**(2-Y)*sc.gamma(a+1)*sc.gammainc(a+1,M*eps)+C/G**(2-Y)*sc.gamma(a+1)*sc.gammainc(a+1,G*eps)    
bep0=C*sc.gamma(a)*(sc.gammainc(a,M*eps) -1)
lambdaep0=C*np.exp(-M*eps)*eps**(-Y)/Y +bep0*M/Y
ben0=C*sc.gamma(a)*(sc.gammainc(a,G*eps) -1)
lambdaen0=C*np.exp(-G*eps)*eps**(-Y)/Y +ben0*G/Y

h = 0.01
n = 100


def_outside = False



Npaths = int(1000)

paths = np.zeros([Npaths,n])
# jumpSizes = np.zeros([Npaths,n])
# jumpIndicator = np.zeros([Npaths,n])

def CGMYpathSimulation(Npaths,h,n,epsilon, C,G,M,Y):
    if not def_outside:
        a = 1-Y
        sigmaEpsSq = C/M**(2-Y)*sc.gamma(a+1)*sc.gammainc(a+1,M*eps)+C/G**(2-Y)*sc.gamma(a+1)*sc.gammainc(a+1,G*eps)    
    else:
        sigmaEpsSq = sigmaEpsSq0
    # print(sigmaEpsSq)    
    dW = normal.rvs(size=[Npaths,n])*np.sqrt(h)    
    diffu = np.cumsum(np.sqrt(sigmaEpsSq)*dW,axis=1)
    
    def CGMYPositiveOneSided(Npaths,h,n,eps, C,M,Y):
        a=1-Y
        #bep=C/(M**a)*sc.gamma(a)*(sc.gammainc(a,M*eps) -1)
        if not def_outside:
            bep=C*sc.gamma(a)*(sc.gammainc(a,M*eps) -1)

            # print(bep)
            lambdaep=C*np.exp(-M*eps)*eps**(-Y)/Y +bep*M/Y
        else:
            bep = bep0
            lambdaep = lambdaep0
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
    
    def CGMYNegativeOneSided(Npaths,h,n,eps, C,M,Y):
        a=1-Y
        #bep=C/(M**a)*sc.gamma(a)*(sc.gammainc(a,M*eps) -1)
        
        if not def_outside:
            ben=C*sc.gamma(a)*(sc.gammainc(a,M*eps) -1)
            lambdaen=C*np.exp(-M*eps)*eps**(-Y)/Y +ben*M/Y
        else:
            ben = ben0
            lambdaen = lambdaen0

        # print(bep)
        # print(lambdaep)
        DXep = np.zeros([Npaths,n])
        jumpSizes = np.zeros([Npaths,n])
        
        for k in range(0,Npaths): 
            N=np.random.poisson(lambdaen)
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
            DXep[k,:]=ben*h + DJep
        return DXep, jumpSizes

    [DXpep, jumpSizesUp] = CGMYPositiveOneSided(Npaths,h,n,eps, C,M,Y)
    [DXnep, jumpSizesDown] = CGMYNegativeOneSided(Npaths,h,n,eps, C,G,Y)
    DXep=DXpep-DXnep+diffu
    jumpSizes = jumpSizesUp-jumpSizesDown 
    jumpIndicator = np.zeros([Npaths,n])
    jumpIndicator[jumpSizes!=0] = 1
    return DXep,DXpep, DXnep, jumpIndicator, jumpSizes, jumpSizesUp, jumpSizesDown, dW, diffu

tic =time.perf_counter()
[DXep,DXpep, DXnep, jumpIndicator, jumpSizes, jumpSizesUp, jumpSizesDown, dW, diffu] = CGMYpathSimulation(Npaths,h,n,eps, C,G,M,Y)
Xep = np.cumsum(DXep,axis=1)
toc = time.perf_counter()

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
[density, kv] = getProbabilityDensity(12,1,0,0,C,G,M,Y)
I = np.where((kv >= -Bound) & (kv <= Bound))


fig = plt.figure()
plt.plot(kv[I], density[I])
myhist = plt.hist(Xep[:,-1]*(np.abs(Xep[:,-1])<Bound), bins = 100, density=True)
plt.show()

########################################################

r = 0.04; # tasso risk free
d = 0; # dividend yield
S0 = 1;
T = 1;

# m è il compensatore esponenziale che rende l'asset scontato martingala.
# E' l'analogo di -0.5 sigma^2 per il modello Black-Scholes.
m  = -C*sc.gamma(-Y)*((M-1)**Y-M**Y+(G+1)**Y-G**Y)

a = 1-Y
sigmaEpsSq = C/M**(2-Y)*sc.gamma(a+1)*sc.gammainc(a+1,M*eps)+C/G**(2-Y)*sc.gamma(a+1)*sc.gammainc(a+1,G*eps)    


# Discretizzazione completa nel tempo
AssetSimulation = np.zeros([Npaths,n+1])
AssetSimulation[:,0]=S0;
deltaT = 1/n;

for i in range(1,n+1):
    AssetSimulation[:,i] = AssetSimulation[:,i-1]*np.exp( ( r - d + m  )*deltaT + DXep[:,i-1] )


# MartingaleCheck
MartingaleCheck = np.mean(AssetSimulation,0)

fig2 = plt.figure()
plt.plot(range(0,101), MartingaleCheck)
plt.plot(range(0,101), np.exp(r*np.linspace(0,1,101)))
plt.show()
K = 0.9

# fin qui è corretto (uguale al codice matlab)
#############################################
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
priceFFT = CallPricingFFT(16,S0,K,T,r,d,C,G,M,Y);
print(priceFFT)

print("Price according to Monte Carlo")
priceMC = np.mean(np.maximum(AssetSimulation[:,-1] - K,0))*np.exp(-r*T);
print(priceMC)


print(f"Time passed {toc - tic:0.4f} seconds")
