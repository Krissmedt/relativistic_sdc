import numpy as np
from rvv_functions import *

def B(x,q=1):
    B = np.zeros((x.shape[0],3),dtype=np.float)
    B[:,2] = 1.

    B = B * 25.0/q
    return B

def E(x,q=1):
    E = np.zeros((x.shape[0],3),dtype=np.float)
    nq = x.shape[0]
    E_mat = np.array([[1,0,0],[0,1,0],[0,0,-2]])
    E_mag = 1*(4.9**2 / q)
    # E_mag = -0.1
    for pii in range(0,nq):
        E[pii,:] = np.dot(E_mat,x[pii,:]) * E_mag

    #
    #E[:,2] = -0.1
    # E[:,2] = -0.1*x[:,2]
    #E[:,2] = 0.
    return E

def F(vel,E,B,q=1,c=1):
    # F = q/1 * (E + np.cross(vel/(gu(vel,c=c)[:,np.newaxis]/c),B))
    F = q/1 * (E + np.cross(vel/(gu(vel,c=c)[:,np.newaxis]),B))
    return F
