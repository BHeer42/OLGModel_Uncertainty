# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:36:08 2021

@author: heerburk
"""



# Part 1: import libraries
import pandas as pd
import numpy as np
#import scipy.optimize 
import quantecon as qe
from scipy.stats import norm
from scipy import interpolate
import time
import math
import matplotlib.pyplot as plt

# part 2: definition of the functions
#
# wage function
def wage(k,l):
    return (1-alpha) * k**alpha * l**(-alpha)

# interest rate function
def interest(k,l):
    return alpha * k**(alpha - 1) * l**(1-alpha)

# production function
def production(k,l):
    return k**(alpha) * l**(1-alpha)

# utility function 
def u(c,l): 
    if eta==1:
        y =  gamma*log(c) +(1-gamma)*log(1-l) 
    else:
        y = (c**(gamma*(1-eta)) *(1-l)**((1-gamma)*(1-eta))) / (1-eta)
    return y


# marginal utility from consumption
def uc(c,l):
	y = gamma*c**(gamma*(1-eta)-1) * (1-l)**((1-gamma)*(1-eta))
	return y

# marginal disutility from labor
def ulabor(c,l):
	y = (1-gamma)*c**(gamma*(1-eta)) * (1-l)**((1-gamma)*(1-eta)-1) 
	return y


# marginal utility from consumption next period given a',iperm,iy1,iage1
# in next period
def uc1(a1,iperm,iy1,iage1):
    if iage1<=nw-1:
        if a1<=kmin: 
            c1 = cwopt[0,iy1,iperm,iage1]
            labor = lopt[0,iy1,iperm,iage1]
        elif a1>=kmax:
            c1 = cwopt[na-1,iy1,iperm,iage1]
            labor =  lopt[na-1,iy1,iperm,iage1]
        else:    
            c_polate = interpolate.interp1d(a,cwopt[:,iy1,iperm,iage1], kind=kind_tol)
            c1 = c_polate(a1)
            l_polate = interpolate.interp1d(a,lopt[:,iy1,iperm,iage1], kind=kind_tol)
            labor = l_polate(a1)
    else:
        labor = 0
        if a1<=kmin: 
            c1 = cropt[0,iage1-nw]
        elif a1>=kmax:
            c1 = cropt[na-1,iage1-nw];
        else:
            c_polate = interpolate.interp1d(a,cropt[:,iage1-nw], kind=kind_tol)
            c1 = c_polate(a1)
            
    y = uc(c1,labor)
    return y


def optimal_labor(a0,a1):
    w0=(1-taun-taup)*w*ef[iage]*perm[iperm]*ye1[iy]
    labor = gamma - ((1+(1-tauk)*(r-delta))*a0+trbar-ygrowth*a1)*(1-gamma)/w0    
    return labor

# computes the gini for distribution where x has measure g(x) 
def ginid(x,gy,ng):
    x =  np.where(x<=0, 0, x)
    xmean = np.dot(x,gy)
#    y = np.c_[x,gy]
    # sorting according to first column still has to be implemented
#    y0 = np.sort(y,axis=0)
#    x = y0[:,0]
#    g = y0[:,1]
    g = gy
    f = np.zeros(ng)  # accumulated frequency */
    f[0] = g[0]*x[0]/xmean
    gini = 1- f[0] * g[0]
    for i in range(1,ng):
        f[i] = f[i-1] + g[i]*x[i]/xmean
        gini = gini - (f[i]+f[i-1])*g[i]
	
    return gini


# value function of retried with wealth a=a[ia] and a'=x
def value1(x):
    c = (1+(1-tauk)*(r-delta))*a[ia]+pen+trbar-ygrowth*x
    c = c/(1+tauc)
    if c<=0: 
        return neg
    
    vr1 = vr_polate(x) # next-period value function at x
    y = u(c,0)+ygrowth**(gamma*(1-eta))*sp1[nw+iage-1]*beta1*vr1
    return y

# value function of worker with wealth a=a[ia]
def value2(x):
    k0 = a[ia]
    k1 = x
    
    # solving directly for optimal labor supply
    labor = optimal_labor(k0,k1)
    # alternative solution: non-linear eq. system
    # labor = scipy.optimize.fsolve(findl, laborinitial, args=(k0,k1))
    # if abs(findl(labor,k0,k1))>tol:
        # print("optimal_labor not equal zero")
        
    if labor<0:
        labor=0
    
    if labor>labormax:
        labor=labormax
        
    w0 = w*ef[iage]*ye1[iy]*perm[iperm]
    c=(1+(1-tauk)*(r-delta))*k0+(1-taun-taup)*w0*labor+trbar-ygrowth*x
    c=c/(1+tauc)
    if c<=0: 
        return neg
          
    if iage==nw-1:
        vr_polate = interpolate.interp1d(a,vr[:,0], kind=kind_tol)
        y = u(c,labor)+ygrowth**(gamma*(1-eta))*sp1[iage]*beta1*vr_polate(k1)
    else:
        y = u(c,labor)
        for iy1 in range(ny): 
            # interpolation of vw at age iage in the next period of life
            vw_polate = interpolate.interp1d(a,vw[:,iy1,iperm,iage+1], kind=kind_tol)
            y = y + py[iy,iy1]*ygrowth**(gamma*(1-eta))*sp1[iage]*beta1*vw_polate(x)
    return y


# first-order condition with respect to labor
# non-linear equation in labor
# global variable yglo: next-period wealth
def findl(labor,k0,k1):
    w0 = w*ef[iage]*perm[iperm]*ye1[iy] # hourly wage
    c=(1+(1-tauk)*(r-delta))*k0+(1-taun-taup)*w0*labor+trbar-ygrowth*k1
    c=c/(1+tauc)
    y = w0/(1+tauc)*uc(c,labor) - ulabor(c,labor)
    return y


# searches the MAXIMUM using golden section search
# see also Chapter 11.6.1 in Heer/Maussner, 2009,
# Dynamic General Equilibrium Modeling: Computational
# Methods and Applications, 2nd ed. (or later)
def GoldenSectionMax(f,ay,by,cy,tol):
    r1 = 0.61803399 
    r2 = 1-r1
    x0 = ay
    x3 = cy  
    if abs(cy-by) <= abs(by-ay):
        x1 = by 
        x2 = by + r2 * (cy-by)
    else:
        x2 = by 
        x1 = by - r2 * (by-ay)
    
    f1 = - f(x1)
    f2 = - f(x2)

    while abs(x3-x0) > tol*(abs(x1)+abs(x2)):
        if f2<f1:
            x0 = x1
            x1 = x2
            x2 = r1*x1+r2*x3
            f1 = f2
            f2 = -f(x2)
        else:
            x3 = x2
            x2 = x1
            x1 = r1*x2+r2*x0
            f2 = f1
            f1 = -f(x1)
            
    if f1 <= f2:
        xmin = x1
    else:
        xmin = x2
    
    return xmin

def testfunc(x):
    return -x**2 + 4*x + 6


# test goldensectionsearch
xmax = GoldenSectionMax(testfunc,-2.2,0.0,10.0,0.001)
print(xmax)    

start_time = time.time()

# abbreviations
exp = np.e
log = math.log

# Step 1.1: Import data
data = pd.read_excel (r'C:\Users\heerburk\Documents\papers\optimization\python\survival_probs.xlsx') 
df = pd.DataFrame(data, columns= ['sp1','ef'])
print(df)
arr = np.array(df)
print(arr)
sp1 = arr[:,0]
ef = arr[0:45,1]
print(ef)


# Step 1.2: Parameterization of the model
# demographics
nage=70                # maximum age                  
nw=45                  # number of working years        
Rage=46                # first period of retirement 
nr=nage-Rage+1         # number of retirement years
popgrowth0 = 0.0075400000 # population growth rate


# preferences
beta1=1.011             # discount factor 
gamma=0.33		        # weight of consumption in utility
lbar=0.25		        # steady-state labor supply
eta=2.0		           # 1/IES

# production
ygrowth=1.02		    # annual growth factor
alpha=0.35             # production elasticity of capital
delta=0.083            # depreciation rate
rbbar=1.04		        # initial annual real interest rate on bonds

# fiscal policy and social security
taulbar=0.28	        # both taun+taup=0.28!!, see Mendoza, Razin (1994), p. 311
tauk=0.36              # capital income tax rate
tauc=0.05              # consumption tax rate
taup=0.124             # initial guess for social security contribution rate
replacement_ratio=0.352	# gross replacement ratio US
bybar=0.63					# debt-output ratio
gybar=0.18					# government consumption-output ratio


kind_tol = 'linear'     # interpolation of policy functions:
                        # 'linear' or 'Cubic'
eps=0.05       # small parameter to check if we have boundary solution for a'(a) at a=0 or a=kmax
phi=0.80       # update aggregate variables in outer iteration over K, L, tr, taup, taun
tol=0.0001     # percentage deviation of final solution K and L
tol1=1e-5      # tolerance for golden section search 
neg=-1e10      # initial value for value function 
nq = 30        # number of outer iterations


# productivity of workers

nperm=2;        # number of permanent productivities
perm = np.zeros(2)
perm[0]=0.57
perm[1]=1.43

lamb0=.96           # autoregressive parameter 
sigmay1=0.38        # variance for 20-year old, log earnings */
sigmae=0.045        # earnings disturbance term variance */
ny=5                # number of productivity types
m=1                 # width of the productivity grid
                    # -> calibrated to replicate Gini wages=0.37


# compute productivity grid
sy = np.sqrt(sigmae/(1-lamb0**2))    # earnings variance 
ye =  np.linspace(-m*sy, m*sy, ny)
print(ye)



# transition matrix using Tauchen's approximation
# return is a class 'Markov Chain'
mc = qe.markov.approximation.tauchen(lamb0, sigmae, 0.0,  m, ny)
# transition matrix is stored in object 'P
py = mc.P

# mass of the workers
muy = np.zeros(ny)
w = ye[1]-ye[0]

# first year mass distribution
muy[0] = norm.cdf( (ye[0]+w/2)/np.sqrt(sigmay1) ) 
muy[ny-1] = 1-norm.cdf( (ye[ny-1]-w/2)/np.sqrt(sigmay1))


for i in range(1,ny-1):
    muy[i] = ( norm.cdf( (ye[i]+w/2)/np.sqrt(sigmay1) ) - 
        norm.cdf( (ye[i]-w/2)/np.sqrt(sigmay1) ) )
    


#    muy[i,:] = muy[i-1,:] @ py * sp1[i] / (1+popgrowth0)

# transform ye so that mean exp(ye) = 1
# (mean efficiency equal to one)

ye1=np.exp(ye)
meane=np.dot(muy,ye1)
#ye = ye-log(meane)
print(ye)
ye1=np.exp(ye)

# asset grids
kmin=0              # inidividual wealth
kmax=20             # upper limit of capital grid 
na=501              # number of grid points over assets a in [kmin,kmax]
a =  np.linspace(kmin, kmax, na)   # asset grid policy function
nag = 2*na
ag =  np.linspace(kmin, kmax, nag)   # asset grid distribution function

nearn=500           # individual earnings grid point number
emax = 0.01*(nearn-1);
earn =  np.linspace(0, emax, nearn)   # grid over cumulated earnings
income = np.linspace(0,emax,nearn)      #  grid over income
labormax=0.6        # maximum labor supply

# measure of living persons
mass = np.ones(nage)

for i in range(nage-1):
    mass[i+1]=mass[i]*sp1[i]/(1+popgrowth0)

mass = mass / mass.sum()



aggregate_mat = np.zeros((8,nq))     # stores initial values at each iteration q


# initial guesses
#
rbar=0.03   # interest rate
nbar=0.3    # aggregate efficienct labor L
nold=100    # initialization 
mean_labor=0.3   # average working hours
kbar=(alpha/(rbar+delta))**(1/(1-alpha))*nbar # capital stock 
kold=100
omega = kbar*1.2    # aggregate wealth
trbar=0.01          # transfers, initial guess 
w=wage(kbar,nbar) 
r=interest(kbar,nbar)
pen=replacement_ratio*(1-taulbar)*w*mean_labor*sum(mass)/sum(mass[0:nw])
taup=pen*sum(mass[nw:nage])/sum(mass)/(w*nbar)  # balanced budet social security
taun = taulbar-taup
bequests=0

#
# computation of the Gini coefficients of hourly wages    
# wage inequality
#
gwages = np.zeros((nw,nperm,ny,2))  # distribution of wages

# initialization of wage distribution at age 1
for iperm in range(nperm):
    for iy in range(ny):
        gwages[0,iperm,iy,0] = ye1[iy]*perm[iperm]*ef[0] # hourly wage at age 1
        gwages[0,iperm,iy,1] = 1/2*muy[iy]*mass[0]/sum(mass[0:nw]) # measure of households

for i in range(1,nw,1):
    print(i)
    for iperm in range(nperm):
        for iy in range(ny):
            gwages[i,iperm,iy,0] = ye1[iy]*perm[iperm]*ef[i]
            for iy1 in range(ny):
                gwages[i,iperm,iy1,1] = gwages[i,iperm,iy1,1]+sp1[i-1]/(1+popgrowth0)*py[iy,iy1]*gwages[i-1,iperm,iy,1]

# wages at age nw
for iperm in range(nperm):
    for iy in range(ny):
        gwages[nw-1,iperm,iy,0] = ye1[iy]*perm[iperm]*ef[nw-1] # hourly wage at age 1                

i0=-1
fwage=np.zeros((nw*nperm*ny,2))
for i in range(nw):
    for iperm in range(nperm):
        for iy in range(ny):
            i0=i0+1
            fwage[i0,0]=gwages[i,iperm,iy,0]
            fwage[i0,1]=gwages[i,iperm,iy,1]
            
#
# sorting according to first column 
# --> preparing for function ginid()
#
y0 = fwage[fwage[:,0].argsort()]
x = y0[:,0]
g = y0[:,1]
print("Gini_w= " + str(ginid(x,g,nw*nperm*ny)))


#
# outer iteraton over aggregates K, L, tr, taup: q = 0,..,nq
#
#


# saves aggregates during iteration: K, L, Omega, mean labor, taun, taup, tr, bequests
aggregateq = np.zeros((nq,8))

q=-1
crit = 1+tol
while q<nq-1 and crit>tol:
    q=q+1
    print("q: " + str(q))
    crit=max(abs((kbar-kold)/kbar),abs((nbar-nold)/nbar)) # percentage deviation of K and L below tol
    aggregateq[q,0] = kbar
    aggregateq[q,1] = nbar
    aggregateq[q,2] = omega
    aggregateq[q,3] = mean_labor
    aggregateq[q,4] = taun
    aggregateq[q,5] = taup
    aggregateq[q,6] = bequests
    aggregateq[q,7] = trbar
    
    w=wage(kbar,nbar)
    r=interest(kbar,nbar)   # marginal product of capital
    rb = (1-tauk)*(r-delta) # interest rate on bonds
    kold=kbar
    nold=nbar


    # 
    # computation of the policy function: value function iteration
    #
    
    
    vr=np.zeros((na,nr))    # value function with lump-sum pensions: only depends on assets a
    aropt=np.zeros((na,nr))  # optimal asset 
    cropt=np.zeros((na,nr)) # optimal consumption 

    # workers' value function 
    vw = np.zeros((na,ny,nperm,nw))
    awopt = np.zeros((na,ny,nperm,nw))
    lopt = np.zeros((na,ny,nperm,nw))
    cwopt = np.zeros((na,ny,nperm,nw))

    for ia in range(na):
        c = a[ia]*(1+(1-tauk)*(r-delta))+pen+trbar
        c = c/(1+tauc)
        vr[ia,nr-1] = u(c,0)
        cropt[ia,nr-1] = c
        aropt[ia,nr-1] = 0

    for iage in range(nr-1,0,-1):
        print(iage)
        vr_polate = interpolate.interp1d(a,vr[:,iage], kind=kind_tol) # interpolation of vr at age iage+1
        for ia in range(na):
            ax=0 
            bx=-1 
            cx=-2
            v0=neg
            m=-1
            while ax>bx or bx>cx:
                m=m+1
                v1=value1(a[m])
                if v1>v0:
                    if m==0: 
                        ax=a[m] 
                        bx=a[m]
                    else:
                        bx=a[m] 
                        ax=a[m-1];
                    
                    v0=v1
                else:
                    cx=a[m]
                
                if m==na-1: 
                    ax=a[m-2] 
                    bx=a[m-1] 
                    cx=a[m-1] 
                
            

            if ax==bx:  # check: a[1] is maximum point on grid?
                bx=ax+(a[1]-a[0])*eps
                if value1(ax)>value1(bx): # boundary solution
                    aropt[ia,iage-1]=kmin # CHECK INDICES
                else:
                    cx=a[1]
                    aropt[ia,iage-1] = GoldenSectionMax(value1,ax,bx,cx,tol1)
                
            elif bx==cx: # check: a[na] is maximum point on grid?
                ax = a[na-2]
                cx = a[na-1]
                bx = a[na-1]-eps*(cx-ax)
                if value1(cx)>value1(bx):   
                    aropt[ia,iage-1] = a[na-1]
                else:
                    aropt[ia,iage-1]=GoldenSectionMax(value1,ax,bx,cx,tol1)
                
            else:   # interior solution ax<bx<cx
                aropt[ia,iage-1]=GoldenSectionMax(value1,ax,bx,cx,tol1)
            

            vr[ia,iage-1]=value1(aropt[ia,iage-1])
            c = (1+(1-tauk)*(r-delta))*a[ia]+pen+trbar-ygrowth*aropt[ia,iage-1] 
            cropt[ia,iage-1] = c/(1+tauc) 
        print("q: " +str(q))
        print("n: " +str(nbar))
            
    
    for iage in range(nw-1,-1,-1):
        print(iage)
        print("q: " +str(q))
        print("kbar: " +str(kbar))
        for iperm in range(nperm):
            print(iperm)
            
            for iy in range(ny):
                m0=-1
                for ia in range(na):
                    k0 = a[ia]
                     
                    #
                    # critical step: find a good initial value for the optimal labor supply
                    #  in the non-linear equation optimal_labor(x)
                    #
                    # simply putting the initial value labor=0.3 does NOT work!
                    
                    #if q==0 and iage==nw-1 and ia==0 and iy==0:     
                    #    laborinitial=0.3
                    #else:
                    #    if q>0:
                    #        laborinitial = lopt[ia,iy,iperm,iage]
                    #    else:
                    #        if ia>0:
                    #            laborinitial = lopt[ia-1,iy,iperm,iage]
                    #        else:
                    #            if iage<nw-1:
                    #                laborinitial = lopt[ia,iy,iperm,iage+1]
                    #            elif iy>0:
                    #                laborinitial = lopt[ia,iy-1,iperm,iage]
                                             
                    # triple ax,bx,cx
                    ax=0 
                    bx=-1 
                    cx=-2
                    v0=neg
                    m=m0
                    while ax>bx or bx>cx:
                        m=m+1
                        v1=value2(a[m])
                    
                        if v1>v0:
                            m0=max(-1,m-2) # monotonocity in a'(a)
                            if m==0: 
                                ax=a[m] 
                                bx=a[m]
                            else:
                                bx=a[m] 
                                ax=a[m-1]
                        
                            v0=v1
                        else:
                            cx=a[m]
                        
                        if m==na-1:
                            ax=a[m-1] 
                            bx=a[m] 
                            cx=a[m] 
                        
    

                    if ax==bx:  # boundary optimum, ax=bx=a[0]  
                        bx=ax+eps*(a[1]-a[0])
                        if value2(bx)<value2(ax):
                            k1=a[0]
                        else:
                            k1 = GoldenSectionMax(value2,ax,bx,cx,tol1)
                        
                    elif bx==cx: # boundary solution at bx=cx=a[na-1]
                        bx=cx-eps*(a[na-1]-a[na-2])
                        if value2(bx)<value2(cx):
                            k1 = a[na-1]
                        else:
                            k1 = GoldenSectionMax(value2,ax,bx,cx,tol1)
                        
                    else:
                        k1 = GoldenSectionMax(value2,ax,bx,cx,tol1)
                    
        
            
                    awopt[ia,iy,iperm,iage]=k1
                    labor = optimal_labor(k0,k1)
                    # labor = scipy.optimize.fsolve(findl, laborinitial, args=(k0,k1))
                    #if abs(findl(labor,k0,k1))>tol:
                    #    print("optimal_labor not equal zero")
                    
                    if labor<0:
                        labor=0
                    
                    if labor>labormax:
                        labor=labormax
                    
                    
                    lopt[ia,iy,iperm,iage]=labor
                    w0 = w*ef[iage]*ye1[iy]*perm[iperm]
                    c =(1+(1-tauk)*(r-delta))*k0
                    c =c+(1-taun-taup)*w0*labor+trbar-ygrowth*k1
                    c =c/(1+tauc)

                    vw[ia,iy,iperm,iage] = value2(k1)
                    cwopt[ia,iy,iperm,iage] = c
  
    # save results
    np.save('vw',vw)
    np.save('vr',vr)
    np.save('cwopt',cwopt)
    np.save('cropt',cropt)
    np.save('awopt',awopt)
    np.save('aropt',aropt)
    np.save('lopt',lopt)

    
    # --------------------------------------------------------------
    #
    #
    # computation of the distribution of capital 
    #
    #    
    # ---------------------------------------------------------------

    
    gkw = np.zeros((nag,ny,nperm,nw)) # distribution of wealth among workers
    gkr = np.zeros((nag,nr))          # distribution of wealth among retirees
    kgen = np.zeros(nage)             # distribution of wealth over age
    gwealth = np.zeros(nag)       # distribution of wealth 
    gearn = np.zeros(nearn)       # distribution of earnings (workers)
    gincome = np.zeros(nearn)   # distribution of income (all households)
    
    gk = np.zeros((nag,nw))
    gk[0,0] = mass[0]       # all 1-year old hold zero wealth
    test=0

    # mass at age 1
    # all agents have zero wealth
    for iy in range(ny):
        gkw[0,iy,0,0] = 1/2*muy[iy]*mass[0]     # perm[1]
        gkw[0,iy,1,0] = 1/2*muy[iy]*mass[0]     # perm[2]
    

    #
    # distribution at age 2,...,nw
    #
    for iage in range(nw-1):
        print("distribution")
        print("iage: " +str(iage))
       
        for iperm in range(nperm):
            print("iperm: " +str(iperm))
            for iy in range(ny):    #  present-period idiosyncratic productivity
                for iag in range(nag):
                    if ag[iag]<=kmin: 
                        k1 = awopt[0,iy,iperm,iage]
                        labor = lopt[0,iy,iperm,iage]
                    elif ag[iag]>=kmax:
                        k1 = awopt[na-1,iy,iperm,iage]
                        labor = lopt[na-1,iy,iperm,iage]
                    else:   # linear interpolation between grid points
                        aw_polate = interpolate.interp1d(a,awopt[:,iy,iperm,iage], kind=kind_tol)
                        labor_polate = interpolate.interp1d(a,lopt[:,iy,iperm,iage], kind=kind_tol)
                        k1 = aw_polate(ag[iag])
                        labor = labor_polate(ag[iag])
        
                    #
                    # distribution of earnings and income at age i
                    #
                    x = perm[iperm]*ye1[iy]*ef[iage]*w*labor   # earnings 
                    y = x + (r-delta)*ag[iag]                  #  income    
        
                    if x<=0:
                        gearn[0] = gearn[0] +  gkw[iag,iy,iperm,iage]/sum(mass[0:nw-1])
          
                    elif x>=earn[nearn-1]:
                        gearn[nearn-1] = gearn[nearn-1] + gkw[iag,iy,iperm,iage]/sum(mass[0:nw-1])
             
                    else:   # linear interpolation between grid points
                        j0=sum(earn<x)
                        j0=min(j0,nearn-1)
                        n0=(x-earn[j0-1])/(earn[j0]-earn[j0-1])
                        gearn[j0-1] = gearn[j0-1]+ (1-n0)*gkw[iag,iy,iperm,iage]/sum(mass[0:nw])
                        gearn[j0] = gearn[j0]+ n0*gkw[iag,iy,iperm,iage]/sum(mass[0:nw])
                    
        
                    if y<0:
                        gincome[0] = gincome[0] + gkw[iag,iy,iperm,iage]
                    elif y>=income[nearn-1]:
                        gincome[nearn-1] = gincome[nearn-1] + gkw[iag,iy,iperm,iage]
                    else:
                        j0=sum(income<y) # CHECK income.<y or income<y
                        j0=min(j0,nearn-1)
                        n0 = (y-income[j0-1])/(income[j0]-income[j0-1])
                        gincome[j0-1] = gincome[j0-1]+ (1-n0)*gkw[iag,iy,iperm,iage]
                        gincome[j0] = gincome[j0]+ n0*gkw[iag,iy,iperm,iage]
                    
                    
                    #
                    # dynamics of the distribution function gkw
                    #
                    if k1<=kmin:
                        for iy1 in range(ny): # next-period idiosyncratic productivity
                            
                            gkw[0,iy1,iperm,iage+1] = (gkw[0,iy1,iperm,iage+1]
                                +py[iy,iy1]*sp1[iage]/(1+popgrowth0)*gkw[iag,iy,iperm,iage])
          
                            test = test + py[iy,iy1]*sp1[iage]/(1+popgrowth0)*gkw[iag,iy,iperm,iage]
                    
                    elif k1>=kmax:
                        for iy1 in range(ny):
                              
                            gkw[nag-1,iy1,iperm,iage+1] = (gkw[nag-1,iy1,iperm,iage+1]
                                +py[iy,iy1]*sp1[iage]/(1+popgrowth0)*gkw[iag,iy,iperm,iage])
          
                            test = test + py[iy,iy1]*sp1[iage]/(1+popgrowth0)*gkw[iag,iy,iperm,iage]
                            
                    else:
                        j0 = sum(ag<k1) # CHECK 
                        j0 = min(j0,nag-1)
                        n0 = (k1-ag[j0-1])/(ag[j0]-ag[j0-1])
                        for iy1 in range(ny):
                            gkw[j0-1,iy1,iperm,iage+1] = (gkw[j0-1,iy1,iperm,iage+1]
                                + (1-n0)*py[iy,iy1]*sp1[iage]/(1+popgrowth0)*gkw[iag,iy,iperm,iage])
                            gkw[j0,iy1,iperm,iage+1] = (gkw[j0,iy1,iperm,iage+1]
                                + n0*py[iy,iy1]*sp1[iage]/(1+popgrowth0)*gkw[iag,iy,iperm,iage])
                       
                            test = test + py[iy,iy1]*sp1[iage]/(1+popgrowth0)*gkw[iag,iy,iperm,iage]
                            
                           
        # summing up the entries for ag[ia], ia=1,...,nag
        for iag in range(nag):
            gk[iag,iage+1] = sum(sum(gkw[iag,:,:,iage+1]))
            
        kgen[iage+1] = np.dot(gk[:,iage+1],ag)  # check
    
        print("q: " +str(q))
        print("kbar: " +str(kbar))
        print("iage+1: " + str(iage+1))
        print("kgen in iage+1: " + str(kgen[iage+1]))
        
        

    iage=nw-1       # last year of working age -> next period retired
    print("iage: " +str(iage))
    for iperm in range(nperm):
        print("iperm: " +str(iperm))
        for iy in range(ny):    #  present-period idiosyncratic productivity
            for iag in range(nag):
                if ag[iag]<=kmin: 
                    k1 = awopt[0,iy,iperm,iage]
                    labor = lopt[0,iy,iperm,iage]
                elif ag[iag]>=kmax:
                    k1 = awopt[na-1,iy,iperm,iage]
                    labor = lopt[na-1,iy,iperm,iage]
                else:   # linear interpolation between grid points
                    aw_polate = interpolate.interp1d(a,awopt[:,iy,iperm,iage], kind=kind_tol)
                    labor_polate = interpolate.interp1d(a,lopt[:,iy,iperm,iage], kind=kind_tol)
                    k1 = aw_polate(ag[iag])
                    labor = labor_polate(ag[iag])
        
                #
                # distribution of earnings and income at age i
                #
                x = perm[iperm]*ye1[iy]*ef[iage]*w*labor   # earnings 
                y = x + (r-delta)*ag[iag]                  #  income    
        
                if x<=0:
                    gearn[0] = gearn[0] +  gkw[iag,iy,iperm,iage]/sum(mass[0:nw-1])
          
                elif x>=earn[nearn-1]:
                    gearn[nearn-1] = gearn[nearn-1] + gkw[iag,iy,iperm,iage]/sum(mass[0:nw-1])
             
                else:   # linear interpolation between grid points
                    j0=sum(earn<x)
                    j0=min(j0,nearn-1)
                    n0=(x-earn[j0-1])/(earn[j0]-earn[j0-1])
                    gearn[j0-1] = gearn[j0-1]+ (1-n0)*gkw[iag,iy,iperm,iage]/sum(mass[0:nw])
                    gearn[j0] = gearn[j0]+ n0*gkw[iag,iy,iperm,iage]/sum(mass[0:nw])
                    
        
                if y<0:
                    gincome[0] = gincome[0] + gkw[iag,iy,iperm,iage]
                elif y>=income[nearn-1]:
                    gincome[nearn-1] = gincome[nearn-1] + gkw[iag,iy,iperm,iage]
                else:
                    j0=sum(income<y) # CHECK income.<y or income<y
                    j0=min(j0,nearn-1)
                    n0 = (y-income[j0-1])/(income[j0]-income[j0-1])
                    gincome[j0-1] = gincome[j0-1]+ (1-n0)*gkw[iag,iy,iperm,iage]
                    gincome[j0] = gincome[j0]+ n0*gkw[iag,iy,iperm,iage]
                    
                    
                #
                # dynamics of the distribution function gkw
                #
                if k1<=kmin:
                    gkr[0,0] = gkr[0,0]+sp1[iage]/(1+popgrowth0)*gkw[iag,iy,iperm,iage]
          
                elif k1>=kmax:
                    gkr[nag-1,0] = gkr[nag-1,0]+sp1[iage]/(1+popgrowth0)*gkw[iag,iy,iperm,iage]
          
                else:
                    j0 = sum(ag<k1) # CHECK 
                    j0 = min(j0,nag-1)
                    n0 = (k1-ag[j0-1])/(ag[j0]-ag[j0-1])
                    gkr[j0-1,0] = (gkr[j0-1,0]
                                + (1-n0)*sp1[iage]/(1+popgrowth0)*gkw[iag,iy,iperm,iage])
                            
                    gkr[j0,0] = (gkr[j0,0]
                                + n0*sp1[iage]/(1+popgrowth0)*gkw[iag,iy,iperm,iage])
    
    kgen[iage+1] = np.dot(gkr[:,0],ag)                  
    
    print("iage+1: " + str(iage+1))
    print("kgen in iage+1: " + str(kgen[iage+1]))
    
    
    for iage in range(nr-1):
        print(iage+nw)
        for iag in range(nag):
            if ag[iag]<=kmin: 
                k1=aropt[0,iage]
            elif ag[iag]>=kmax:
                k1=aropt[na-1,iage]
            else:
                j0 = sum(a<ag[iag])
                j0 = min(j0,na-1)
                n0 = (ag[iag]-a[j0-1])/(a[j0]-a[j0-1])
                k1 = (1-n0)*aropt[j0-1,iage]+n0*aropt[j0,iage]
            
            
            #
            # distribution of income
            #
            y = pen + (r-delta)*ag[iag]               # income 
            
            if y<0:
                gincome[0] = gincome[0] + gkr[iag,iage]
            elif y>=income[nearn-1]:
                gincome[nearn-1] = gincome[nearn-1] + gkr[iag,iage]
            else:
                j0 = sum(income<y)
                j0 = min(j0,nearn-1)
                n0 = (y-earn[j0-1])/(earn[j0]-earn[j0-1])
                gincome[j0-1] = gincome[j0-1]+ (1-n0)*gkr[iag,iage]
                gincome[j0] = gincome[j0]+ n0*gkr[iag,iage]
            
            
            #
            # dynamics of the distribution during retirement
            #
            if k1<=kmin:
                gkr[0,iage+1] = gkr[0,iage+1]+sp1[iage+nw]/(1+popgrowth0)*gkr[iag,iage]
            elif k1>=kmax:
                gkr[nag-1,iage+1]=gkr[nag-1,iage+1]+sp1[iage+nw]/(1+popgrowth0)*gkr[iag,iage]
            else:
                j0 = sum(ag<k1)
                j0 = min(j0,nag-1)
                n0 = (k1-ag[j0-1])/(ag[j0]-ag[j0-1])
                gkr[j0-1,iage+1] = gkr[j0-1,iage+1] + (1-n0)*sp1[iage+nw]/(1+popgrowth0)*gkr[iag,iage]
                gkr[j0,iage+1] = gkr[j0,iage+1] + n0*sp1[iage+nw]/(1+popgrowth0)*gkr[iag,iage]
            
        
        kgen[iage+nw+1] = np.dot(gkr[:,iage+1],ag)
        print("iage+nw+1: " +str(iage+nw+1))
        print("kgen[iage+nw+1]: " +str(kgen[iage+nw+1]))
        
    # append matrix gkr as a new colum to gk
    gk = np.c_[gk,gkr]
    # normalization to one
    gk=gk/sum(sum(gk))
    # computation of the gini coefficient of capital distribution 
    gk1 = np.transpose(gk)
    gk1 = sum(gk1)
    gini_wealth = ginid(ag,gk1,nag)
    print("gini wealth: " +str(gini_wealth))
    gini_earnings = ginid(earn,gearn,nearn)
    print("gini earnings: " +str(gini_earnings))
    gini_income = ginid(income,gincome,nearn)
    print("gini income: " +str(gini_income))
        

    # total savings
    omeganew = sum(kgen)/sum(mass)
    ybar=production(kbar,nbar)
    debt = bybar*ybar
    gbar = gybar*ybar
    
    # update of aggregate wealth and K
    omega = phi*omega + (1-phi)*omeganew
    knew = omeganew - debt
    kbar = phi*kold+(1-phi)*knew
    
    

    # aggregate variablec L, Beq, C, mean working hours mean_labor
    nnew = 0
    bequests = 0
    bigc = 0
    mean_labor=0
    cgen = np.zeros(nage)
    lgen = np.zeros(nw)
    test = 0
    
    
    # Residual: Euler equation
    # compute the residuals of the first-order and Euler eqs
    # to check for accuracy of value function iteration and interpolation method        
    Euler_res = np.zeros((nag,ny,nperm,nw))
    Foc_res = np.zeros((nag,ny,nperm,nw))
    Euler_res_old = np.zeros((nag,nr-1))
    
    for iage in range(nw):
        # print(iage)
        for iperm in range(nperm):
            # print(iperm)
            for iy in range(ny):
                # print(iy)
                for iag in range(nag):
                    if ag[iag]<=kmin: 
                        k1=awopt[0,iy,iperm,iage]
                        labor=lopt[0,iy,iperm,iage]
                        c = cwopt[0,iy,iperm,iage]
                    elif ag[iag]>=kmax:
                        k1=awopt[na-1,iy,iperm,iage]
                        labor=lopt[na-1,iy,iperm,iage]
                        c = cwopt[na-1,iy,iperm,iage]
                    else:   # linear interpolation between grid points 
                        aw_polate = interpolate.interp1d(a,awopt[:,iy,iperm,iage], kind=kind_tol)
                        k1 = aw_polate(ag[iag]) 
                        c_polate = interpolate.interp1d(a,cwopt[:,iy,iperm,iage], kind=kind_tol)
                        c = c_polate(ag[iag])
                        l_polate = interpolate.interp1d(a,lopt[:,iy,iperm,iage], kind=kind_tol)
                        labor = l_polate(ag[iag])
                    
                    # print("gkw: " + str(gkw[iag,iy,iperm,iage]))
                    nnew = nnew + ef[iage]*ye1[iy]*perm[iperm]*labor*gkw[iag,iy,iperm,iage]
                    mean_labor = mean_labor + labor/(sum(mass[0:nw]))*gkw[iag,iy,iperm,iage]
                    cgen[iage] = cgen[iage] + c*gkw[iag,iy,iperm,iage]/mass[iage]
                    lgen[iage]= lgen[iage] + labor*gkw[iag,iy,iperm,iage]/mass[iage]
                    bequests = bequests + k1*(1+(1-tauk)*(r-delta))*(1-sp1[iage])/(1+popgrowth0)*gkw[iag,iy,iperm,iage]
                    bigc = bigc + c*gkw[iag,iy,iperm,iage]
                    gwealth[iag] = gwealth[iag]+gkw[iag,iy,iperm,iage]
                    
                    # computation of the Euler residual
                    # 
                    # (1+g_A)^eta u_{c,t} = beta phi^i E_t{ u_{c,t+1} [1+(1-tauk) (r-delta)] }
                    #
                    
                    x=0
                    if iage<nw-1:
                        for iy1 in range(ny):
                            x = x+beta1*sp1[iage]*py[iy,iy1]*(1+rb)*uc1(k1,iperm,iy1,iage+1)
                    else:
                        x = beta1*sp1[iage]*uc1(k1,iperm,iy1,iage+1)
                    
                    Euler_res[iag,iy,iperm,iage] = 1-x / ( ygrowth**(1-gamma*(1-eta)) * uc(c,labor))
                    x = uc(c,labor)*(1-taun-taup)/(1+tauc)*w*ef[iage]*perm[iperm]*ye1[iy]
                    Foc_res[iag,iy,iperm,iage] = 1 - ulabor(c,labor)/x
    
    print("test: " +str(test))
    print("mass workers: " +str(sum(mass[0:nw])))

    
                
    for iage in range(nr):
        for iag in range(nag):
            if ag[iag]<=kmin: 
                k1 = aropt[0,iage]
                c = cropt[0,iage]
            elif ag[iag]>=kmax:
                k1 = aropt[na-1,iage]
                c = cropt[na-1,iage]
            else:
                ar_polate = interpolate.interp1d(a,aropt[:,iage], kind=kind_tol)
                k1 = ar_polate(ag[iag]) 
                c_polate = interpolate.interp1d(a,cropt[:,iage], kind=kind_tol)
                c = c_polate(ag[iag])
                
            
            
            bequests = bequests + k1*(1+(1-tauk)*(r-delta))*(1-sp1[iage+nw])/(1+popgrowth0)*gkr[iag,iage]
            bigc = bigc + c*gkr[iag,iage]
            gwealth[iag] = gwealth[iag]+gkr[iag,iage]
            cgen[nw+iage] = cgen[nw+iage]+ c*gkr[iag,iage]/mass[nw+iage]
        
            labor=0
            if iage<nr-1:
                Euler_res_old[iag,iage] = 1- beta1*sp1[nw+iage]*(1+rb)*uc1(k1,0,0,iage+nw+1)/ (ygrowth**(1-gamma*(1-eta))*uc(c,labor))
            
    
    
    gini_wealth1 = ginid(ag,gwealth,nag)
    print("gini wealth1: " +str(gini_wealth1))     
    
    print("Mean Residual Errors")
    print("Euler eq young: " +str(np.mean(np.mean(abs(Euler_res)))))
    print("Euler eq old: " +str(np.mean(np.mean(abs(Euler_res_old)))))
    print("Euler eq FOC labor: " +str(np.mean(np.mean(abs(Foc_res)))))
    
    #
    # tatonnement process to update L: 
    #    simple dampening iterative scheme as described in Judd (1998), Section 3.9
    
    nbar = phi*nbar+(1-phi)*nnew
    taxes = taun*w*nbar + tauk*(r-delta)*kbar + tauc*bigc    
    # update of L, K, transfer, kshare, and pen
    transfernew = taxes+bequests+debt*((1+popgrowth0)*ygrowth-(1+(1-tauk)*(r-delta))) - gbar	
    trbar = phi*trbar + (1-phi)*transfernew
    pennew = replacement_ratio*w*mean_labor
    # social  security contributions are calculated 
    # so that the social security budget balances
    taupnew = pennew*sum(mass[nw:nage])/(w*nbar)
    taup = phi*taup+(1-phi)*taupnew
    taunnew = taulbar-taup
    taun = phi*taun + (1-phi)*taunnew
    
    print("mean working hours: " +str(mean_labor))    
    
print("runtime: --- %s seconds ---" % (time.time() - start_time))    
sec = (time.time() - start_time)
ty_res = time.gmtime(sec)
res = time.strftime("%H : %M : %S", ty_res)
print(res)

print("I+C+G= " +str((delta+popgrowth0+(ygrowth-1)+popgrowth0*(ygrowth-1))*kbar+bigc+gbar))
print("Y = " +str(ybar))

 # save results
np.save('cgen',cgen)
np.save('lgen',lgen)
np.save('kgen',kgen)
np.save('earn',earn)
np.save('gearn',gearn)
np.save('gwealth',gwealth)
np.save('ag',ag)



# plotting the averages at different ages 

plt.xlabel('age')
plt.ylabel('mean wealth')
plt.plot(range(21,91),kgen/mass)
plt.show()


plt.xlabel('age')
plt.ylabel('mean consumption')
plt.plot(range(21,91),cgen)
plt.show()


plt.xlabel('age')
plt.ylabel('mean working hours')
plt.plot(range(21,66),lgen)
plt.show()          
  

# plotting the Lorenz curve EARNINGS
fig, ax = plt.subplots()
label1 = 'equal distribution'
label2 = 'earnings'
earn1 = gearn*earn
# normalization
totalearn=np.dot(gearn,earn)
earn1 = earn1/totalearn
earn1 = np.cumsum(earn1)
earn0 = np.cumsum(gearn)
ax.plot(earn0,earn0, linewidth=2, label=label1)
ax.plot(earn0,earn1, linewidth=2, label=label2)
ax.legend()
plt.show()


# plotting the Lorenz curve WEALTH
fig, ax = plt.subplots()
label1 = 'equal distribution'
label2 = 'wealth'
label3 = 'earnings'
ag1 = gwealth*ag
totalwealth = np.dot(gwealth,ag)
ag1 = ag1/totalwealth
ag1 = np.cumsum(ag1)
ag1 = np.r_[0,ag1]
# add the point (0,0) to the figure
ag0 = np.cumsum(gwealth)
ag0 = np.r_[0,ag0]
ax.plot(ag0,ag0, linewidth=2, label=label1)
ax.plot(ag0,ag1, linewidth=2, label=label2)
ax.plot(earn0,earn1, linewidth=2, label=label3)
ax.legend()
plt.show()



