import numpy as np
from rvv_fields import *
from rvv_functions import *

def analytical_gyro(time,x0,v0):
    xref = np.zeros(x0.shape,dtype=np.float)
    vref = np.zeros(v0.shape,dtype=np.float)

    # Relativistic gyro frequency (cgs): omega_B = qB/(gamma*m*c)
    # Relativistic gyro frequency (runko): omega_B*dt = qh*Bh*/(gamma*mh*ch) * m-/m+
    # lfreq = (conf.qe*conf.binit)/(conf.gamma*abs(conf.qe)*conf.cfl**2)

    t = time
    q = np.sign(1)

    # Phase-lag, particles start at a=0 radians
    a = 0

    gamma = gu(v0)
    lfreq = 1*np.linalg.norm(B(x0),axis=1)/(1*1*gamma)
    larmor = v0[:,1]/gamma/lfreq
    cx = x0[:,0] - q*larmor
    cy = x0[:,1]

    vref[:,0] = -v0[:,1]*np.sin(lfreq*t+a)
    vref[:,1] = v0[:,1]*np.cos(lfreq*t+a)
    vref[:,2] = -E(x0)[:,2]*t + v0[:,2]

    xref[:,0] = cx+v0[:,1]/gamma/lfreq*np.cos(lfreq*t+a)
    xref[:,1] = cy+v0[:,1]/gamma/lfreq*np.sin(lfreq*t+a)
    xref[:,2] = -0.5*E(x0)[:,2]*t**2 + v0[:,2]*t + x0[:,2]

    return xref,vref


def analytical_gyro_array(times,x0,v0):
    xref = np.zeros((times.shape[0],x0.shape[0],x0.shape[1]),dtype=np.float)
    vref = np.zeros((times.shape[0],v0.shape[0],v0.shape[1]),dtype=np.float)

    for ti in range(0,times.shape[0]):
        xref[ti],vref[ti] = analytical_gyro(times[ti],x0,v0)

    return xref, vref
