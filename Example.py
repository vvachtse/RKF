#Runge–Kutta–Fehlberg method (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method)

import numpy as np

# Runge Kutta Fehlberg Formula 1

def RKF45I(f, init, t, args=()):

    n = len(t)
    y = np.zeros((n, len(init)))
    y[0] = init
    #We assume that 
    eps = 0.000005


    #B(K,L)
    B1= np.array([2/9., 1/12., 69/128., -17/12., 65/432.])
    B2= np.array([1/4., -243/128., 27/4., -5/16.])
    B3= np.array([135/64., -27/5., 13/16.])
    B4= np.array([16/15., 4/27.])
    B5= 5/144. 

    A1= 0.0
    A2= 2/9.
    A3= 1/3.
    A4= 3/4.
    A5= 1.0
    A6= 5/6.

    CH= np.array([47/450., 0.0, 12/25., 32/225., 1/30., 6/25.])

    CT= np.array([-1/150., 0.0, 3/100., -16/75., -1/20., 6/25.])

    #Initial Step
    h = t[1] - t[0]
    
    for i in range(n - 1):
        k1= h * f( y[i], t[i] + A1 * h, *args)
        k2= h * f( y[i] + B1[0] * k1, t[i] + A2 * h, *args)
        k3= h * f( y[i] + B1[1] * k1 + B2[0] * k2, t[i] + A3 * h, *args)
        k4= h * f( y[i] + B1[2] * k1 + B2[1] * k2 + B3[0] *k3, t[i] + A4 * h, *args)
        k5= h * f( y[i] + B1[3] * k1 + B2[2] * k2 + B3[1] *k3 + B4[0] * k4, t[i] + A5 * h, *args)
        k6= h* f( y[i] + B1[4] * k1 + B2[3] * k2 + B3[2] *k3 + B4[1] * k4 + B5 * k5, t[i] + A6 * h, *args)

        TRE= np.absolute(CT[0] * k1 + CT[1] * k2 + CT[2] * k6 + CT[3] * k4 + CT[4] * k5 + CT[5] * k6)
        
        if TRE[1]>eps :
        	i=i-1
            
        	h=h* 0.9 * (eps/TRE[1]) ** 0.2

        else:
            
        	h=h* 0.9 * (eps/TRE[1]) ** 0.2
        
        y[i+1] = y[i] + CH[0] * k1 + CH[1] * k2 + CH[2] * k3 + CH[3] * k4 + CH[4] * k5 + CH[5] * k6

    return y


# Runge Kutta Fehlberg Formula 2

def RKF45II(f, init, t, args=()):

    n = len(t)
    y = np.zeros((n, len(init)))
    y[0] = init
    #We assume that 
    eps = 0.000005

    #B(K,L)
    B1= np.array([1/4., 3/32., 1932/2197., 439/216., -8/27.])
    B2= np.array([9/32., -7200/2197., -8.0, 2.0])
    B3= np.array([7296/2197., 3680/513., -3544/2565.])
    B4= np.array([-845/4104., 1859/4104.])
    B5= -11/40. 

    A1= 0.0
    A2= 1/4.
    A3= 3/8.
    A4= 12/13.
    A5= 1.0
    A6= 1/2.

    CH= np.array([16/135., 0.0, 6656/12825., 28561/56430., -9/50., 2/55.])

    CT= np.array([1/360., 0.0, -128/4275., -2197/75240., 1/50., 2/55.])

    #Initial Step
    h = t[1] - t[0]
    
    for i in range(n - 1):
        k1= h * f( y[i], t[i] + A1 * h, *args)
        k2= h * f( y[i] + B1[0] * k1, t[i] + A2 * h, *args)
        k3= h * f( y[i] + B1[1] * k1 + B2[0] * k2, t[i] + A3 * h, *args)
        k4= h * f( y[i] + B1[2] * k1 + B2[1] * k2 + B3[0] *k3, t[i] + A4 * h, *args)
        k5= h * f( y[i] + B1[3] * k1 + B2[2] * k2 + B3[1] *k3 + B4[0] * k4, t[i] + A5 * h, *args)
        k6= h* f( y[i] + B1[4] * k1 + B2[3] * k2 + B3[2] *k3 + B4[1] * k4 + B5 * k5, t[i] + A6 * h, *args)

        TRE= np.absolute(CT[0] * k1 + CT[1] * k2 + CT[2] * k6 + CT[3] * k4 + CT[4] * k5 + CT[5] * k6)
        
        if TRE[1]>eps :
        	i=i-1
            
        	h=h* 0.9 * (eps/TRE[1]) ** 0.2

        else:
            
        	h=h* 0.9 * (eps/TRE[1]) ** 0.2
        
        y[i+1] = y[i] + CH[0] * k1 + CH[1] * k2 + CH[2] * k3 + CH[3] * k4 + CH[4] * k5 + CH[5] * k6

    return

# Runge Kutta Fehlberg Formula Sarafyan

def RKF45S(f, init, t, args=()):

    n = len(t)
    y = np.zeros((n, len(init)))
    y[0] = init
    #We assume that 
    eps = 0.000005

    #B(K,L)
    B1= np.array([1/2., 1/4., 0.0, 7/27., 28/625. ])
    B2= np.array([1/4., -1.0, 10/27., -1/5.])
    B3= np.array([2.0, 0.0, -546/625.])
    B4= np.array([1/27., 54/625.])
    B5= -378/625. 

    A1= 0.0
    A2= 1/2.
    A3= 1/2.
    A4= 1.0
    A5= 2/3.
    A6= 1/5.

    CH= np.array([1/24., 0.0, 0.0, 5/48., -27/56., 125/336.])

    CT= np.array([-1/8., 0.0, -2/3., -1/16.,-27/56., 125/336.])

    #Initial Step
    h = t[1] - t[0]
    
    for i in range(n - 1):
       k1= h * f( y[i], t[i] + A1 * h, *args)
       k2= h * f( y[i] + B1[0] * k1, t[i] + A2 * h, *args)
       k3= h * f( y[i] + B1[1] * k1 + B2[0] * k2, t[i] + A3 * h, *args)
       k4= h * f( y[i] + B1[2] * k1 + B2[1] * k2 + B3[0] *k3, t[i] + A4 * h, *args)
       k5= h * f( y[i] + B1[3] * k1 + B2[2] * k2 + B3[1] *k3 + B4[0] * k4, t[i] + A5 * h, *args)
       k6= h* f( y[i] + B1[4] * k1 + B2[3] * k2 + B3[2] *k3 + B4[1] * k4 + B5 * k5, t[i] + A6 * h, *args)

       TRE= np.absolute(CT[0] * k1 + CT[1] * k2 + CT[2] * k6 + CT[3] * k4 + CT[4] * k5 + CT[5] * k6)
       
       if TRE[1]>eps :
       	i=i-1
           
       	h=h* 0.9 * (eps/TRE[1]) ** 0.2

       else:
           
       	h=h* 0.9 * (eps/TRE[1]) ** 0.2
       
       y[i+1] = y[i] + CH[0] * k1 + CH[1] * k2 + CH[2] * k3 + CH[3] * k4 + CH[4] * k5 + CH[5] * k6

    return y
