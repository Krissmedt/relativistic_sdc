import numpy as np
from rvv_io import *
from rvv_functions import *
from rvv_solution import *
from rvv_pushers import *
from rvv_fields import *
from rel_col18 import *
from gauss_legendre import CollGaussLegendre
from gauss_lobatto import CollGaussLobatto

sims = [10,20,40,80,160,320]
tend = 1
M = 5

c = 29979
q = 1
gamma_max = 5.
# gamma_max = 1.00000005
beta_max = np.sqrt(1-1./gamma_max**2.)
uy_max = beta_max*c

new = True

for Nt in sims:
    dt = tend/Nt

    nq = 1

    pos = np.zeros((nq,3),dtype=np.float)
    vel = np.zeros((nq,3),dtype=np.float)

    vel[:,1] = np.linspace(uy_max/8,uy_max,nq)
    vel[:,2] = np.sqrt(1-1./gamma_max**2.)

    pos = np.array([[10.,0.,0.]])
    vel = np.array([[100.,0.,100.]])

    # gamma = gu(vel)
    # lfreq = -q*np.linalg.norm(B(pos),axis=1)/(1*c*gamma)
    # larmor = vel[:,1]/gamma/lfreq
    # larmor = 1*vel[:,1]/(-q*np.linalg.norm(B(pos),axis=1))
    # pos[:,0] = larmor

    t = 0

    rx_array = [0]
    rv_array = [0]
    x_array = [pos]
    v_array = [vel]
    t_array = [t]

    col = coll(CollGaussLobatto,dt,nq,M=M,K=1,c=c,q=q)
    for ti in range(1,Nt+1):
        t = ti*dt

        pos, vel, col = implicit_coll(pos,vel,col)

        rx_array.append(np.linalg.norm(col.Rx,axis=1))
        rv_array.append(np.linalg.norm(col.Rv,axis=1))
        x_array.append(np.copy(pos))
        v_array.append(np.copy(vel))
        t_array.append(t)

    rx_array = np.array(rx_array)
    rv_array = np.array(rv_array)
    x_array = np.array(x_array)
    v_array = np.array(v_array)
    t_array = np.array(t_array)

    wp_dump(t_array,x_array,v_array,dt,"coll_wp_vvrel.h5",new=new)
    new = False

plot_isotraj(x_array,"coll_"+str(Nt),plim=1,label="sim")
plot_traj(x_array,"coll_"+str(Nt),label="sim")
plot_vel(t_array,v_array,"coll_"+str(Nt),label="sim")
# plot_xres(t_array,rx_array,"coll"+str(Nt))
# plot_vres(t_array,rv_array,"coll"+str(Nt))
