import numpy as np
from rvv_functions import *
from rvv_fields import *

def boris(u,E,B,dt,q=-1,ck=0,c=1):
    um = u + dt/2 * (q*E/1 + ck)
    g = gu(um,c)[:,np.newaxis]
    Bmag = np.linalg.norm(B,axis=1)[:,np.newaxis]
    Bdir = np.nan_to_num(B/Bmag)
    t = Bdir * np.tan(q*dt*Bmag/(2*g*1))

    ud = um + np.cross(um,t)

    up = um + np.cross(ud,2*t/(1+t*t))

    vel = up + dt/2 * (q*E/1 + ck)

    return vel

def boris_daniel(v,E,B,dt,ck,gamma,q=-1):
    t = 0.5*dt*q*B/gamma[:,np.newaxis]
    s = 2.0*t/(1.0 + np.linalg.norm(t**2,axis=1)[:,np.newaxis])
    v_min = v + 0.5*dt*q*E + 0.5*ck
    v_star  = v_min + np.cross(v_min, t)
    v_plus  = v_min + np.cross(v_star, s)
    return v_plus + 0.5*dt*q*E + 0.5*ck


def hicary(x,u,E,B,dt,q=-1,ck=0):
    um = u + dt/2 * (q*E/1 + ck)

    t = q*B/(2*1) * dt
    g2 = g_biquad(um,t)
    t = t/g2
    ud = um + np.cross(um,t)

    up = um + np.cross(ud,2*t/(1+np.linalg.norm(t**2,axis=1)[:,np.newaxis]))

    vel = up + dt/2 * (q*E/1 + ck)

    return vel


def g_biquad(um,beta):
    b = np.linalg.norm(beta,axis=1)[:,np.newaxis]
    gm = gu(um)[:,np.newaxis]

    dot = np.linalg.norm(b*um**2,axis=1)
    inner_sq = np.sqrt((gm**2+b**2)**2 + 4*(b**2+dot[:,np.newaxis]))
    g2 = np.sqrt(0.5*(gm**2-b**2+inner_sq))

    return g2



def boris_nr(vel, E, B, dt, alpha=1, ck=0):
    """
    Applies Boris' trick for given velocity, electric and magnetic
    field for vector data in the shape (N x 3), i.e. particles as rows
    and x,y,z components for the vector as the columns.
    k = delta_t * alpha / 2
    """

    k = dt*alpha/2

    tau = k*B

    vMinus = vel + dt/2 * (alpha*E + ck)

    tauMag = np.linalg.norm(tau,axis=1)
    vDash = vMinus + np.cross(vMinus,tau)

    tm = 2/(1+tauMag**2)

    for col in range(0,3):
        vDash[:,col] = tm[:] * vDash[:,col]

    vPlus = vMinus + np.cross(vDash,tau)

    vel_new = vPlus + dt/2 * (alpha*E + ck)

    return vel_new
