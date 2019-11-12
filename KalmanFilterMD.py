#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:16:36 2019
@author: Andrés Echeverri 
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import  dot
from numpy.linalg import inv
plt.style.use('ggplot' )


"""
 generate a sensor signal that can be used as an example, this signal
will have noise and some spikes that can be found in real applications.
"""
t =  np.arange(0, 100, 0.2)
y = [((3*math.exp(-0.02*t)*math.sin(0.2*t + 2) + 5) + 
       np.random.normal(0,0.2))for t in range(len(t))]

y[200] = 4.5
y[250] = 4.0

"""
Shocks are one of the many types of sensor noise that can be found, shocks 
are seing as a biased signal.
"""
shock1 = np.random.normal(0,0.2,4) + 6.5
shock2 = np.random.normal(0,0.2,7) + 3.5
shock3 = np.random.normal(0,0.2,12) + 3.5
y[290:290 + len(shock1)] = shock1
y[230:230 + len(shock2)] = shock2
y[353:353 + len(shock3)] = shock3

"""
Finding some statistics of signal is always a good practice, the most used ones
for our purpose are the mean nd the standard deviation. Howerver, it is necessary 
to find an interval where the signal does not have oscilations.
"""
signal_noise = y[400:500]
mean = np.mean(signal_noise)
std = np.std(signal_noise)
mean_line = np.ones(len(signal_noise), dtype = int)*mean
std_line_1 = np.ones(len(signal_noise), dtype = int)*(mean+std)
std_line_2 = np.ones(len(signal_noise), dtype = int)*(mean-std)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(signal_noise, label="Sensor Signal", color = 'green')
ax1.plot(mean_line,  label="mean", color = 'blue')
ax1.plot(std_line_1, label="std", color = 'blue',  linestyle = '--')
ax1.plot(std_line_2,  color = 'blue',  linestyle = '--')
ax1.set_xlabel("time")
ax1.set_ylabel("Sensor Signal")
ax1.legend()

#Transition Matrix
A = np.array([[1.0]])
#Observation Matrix
C = np.array([[1.0]])
#Process Noise Covariance
Rww = np.array([[0.7]])
#Measurement Noise Covariance
Rvv = np.array([[0.2*0.2]])
# Control action Matrix
B = np.array([0])
#Control input 
U = np.array([0])
#state vector
x = np.zeros((len(y), 1))
#Covariance Matrix
P = np.zeros((len(y), 1))
#Weighted MD vector
MDw = np.zeros((len(y), 1))
#Initial Covariance Value
P[0] = 0.72
I = np.identity(1)



"""
Kalman filter implementation: Most implementatios can be divided in 2 or 3
steos. Some consider innovation as part of the prediction. But, for this 
implementation it is nice to see where the Mahalanobis distance (MD) is taking 
place within the measurement noise covariance calculation. The approximated MD
is used due to its simplicity. 
"""
for i in range(1,len(y)):    
    #Initialization of the vector state
    x[0] = y[0]
    """
    Prediction
    """
    x[i] = dot(A, x[i-1] ) + dot(B, U)
    P[i] = dot(A, dot(P[i], A.T)) + Rww 
    
    """
    Innovation
    """
    e = y[i] - C*x[i]
    Ree = dot(C, dot(P[i], C.T)) + Rvv
    #Mahalanobbis distance approximation
    MD = math.sqrt(e*e)/Ree
    #Weighted MD     
    MDw[i] = 1/(1+(math.exp(-MD) + 0.1))
    #New Measurement Noise Covariance
    Rvv= np.array([[4*MDw[i]]])
    #Kalman gain
    K = dot(P[i], dot(C.T, inv(Ree)))
    """
    Update
    """
    x[i] = x[i] + dot(K, dot(e,K))
    P[i] = dot(I - dot(K,C), P[i])
        
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(y, label = "Sensor Signal", color ='green')
#ax2.plot(x, label = "Filtered Signal",  linewidth=4, color = 'blue')
ax2.set_xlabel("time")
ax2.set_ylabel("signal")
ax2.legend()

fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
ax3.plot(np.subtract(x, np.expand_dims(y, axis=1)),  label="Error", color = 'green')
ax3.plot(np.sqrt(P),  label = "Covariance", color = 'blue',  linestyle = '--')
ax3.plot(-np.sqrt(P), color = 'blue', linestyle = '--')
ax3.set_xlabel("time")
ax3.set_ylabel("error")
ax3.legend()


fig4 = plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
ax4.plot(MDw, label="MDw", color = 'green')
ax4.legend()
