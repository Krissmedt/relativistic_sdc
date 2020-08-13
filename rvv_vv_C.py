import numpy as np
from rvv_io import *
from rvv_functions import *
from rvv_solution import *
from rvv_pushers import *
from rvv_fields import *


sims = [10,20,40,80,160,320,640,1280]
tend = 1
c = 29979
q = 1

gamma_max = 5.
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

    t = 0

    x_array = [pos]
    v_array = [vel]
    t_array = [t]

    for ti in range(1,Nt+1):
        t = ti*dt

        vhalf = vel+F(vel,E(pos,q=q),B(pos,q=q),q=q,c=c)*dt/2
        pos = pos + G(vhalf,c=c)*dt

        Bn = B(pos,q=q)
        gamma = gu(vhalf,c=c)
        ck = -(0.5*dt/gamma[:,np.newaxis])*np.cross(vhalf,Bn)
        vel = boris_daniel(vhalf,0.5*E(pos,q=q),Bn,dt,ck,gamma,q=q)

        x_array.append(pos)
        v_array.append(vel)
        t_array.append(t)

    x_array = np.array(x_array)
    v_array = np.array(v_array)
    t_array = np.array(t_array)

    wp_dump(t_array,x_array,v_array,dt,"vvC_wp_vvrel.h5",new=new)
    new = False

plot_traj(x_array,"vvC_"+str(Nt),label="sim")
plot_isotraj(x_array,"vvC_"+str(Nt),plim=1,label="sim")
plot_vel(t_array,v_array,"vvC_"+str(Nt),label="sim")
