import numpy as np
from rvv_io import *
from rvv_functions import *
from rvv_solution import *
from rvv_pushers import *
from rvv_fields import *

sims = [100,200,400,800,1600,3200]
#sims = [1]
tend = 10
c = 1
Bfield = 1

q = -1
gamma_max = 5.
# gamma_max = 1.00000005
beta_max = np.sqrt(1-1./gamma_max**2.)
uy_max = beta_max*c

new = True

for Nt in sims:
    dt = tend/Nt

    nq = 4

    pos = np.zeros((nq,3),dtype=np.float)
    vel = np.zeros((nq,3),dtype=np.float)

    vel[:,1] = np.linspace(uy_max/8,uy_max,nq)
    vel[:,2] = np.sqrt(1-1./gamma_max**2.)

    gamma = gu(vel)
    lfreq = -q*Bfield/(1*c*gamma)
    larmor = vel[:,1]/gamma/lfreq

    pos[:,0] = larmor
    t = 0

    x_array = [pos]
    v_array = [vel]
    t_array = [t]

    for ti in range(1,Nt+1):
        t = ti*dt

        En = E(pos)
        Bn = B(pos)

        Fn = q/1 * (En + np.cross(G(vel)/c,Bn))
        pos += G(vel)*dt + Gdot3(vel,Fn)*dt**2/2

        Eh = (E(pos) + En)/2
        Bh = (B(pos) + Bn)/2

        vel = boris(pos,vel,Eh,Bh,dt)

        x_array.append(pos)
        v_array.append(vel)
        t_array.append(t)

    x_array = np.array(x_array)
    v_array = np.array(v_array)
    t_array = np.array(t_array)

    wp_dump(t_array,x_array,v_array,dt,"vvB_wp_vvrel.h5",new=new)
    new = False

xref,vref = analytical_gyro_array(t_array,x_array[0,:,:],v_array[0,:,:])
plot_traj(x_array,"vvB_"+str(Nt),label="sim")
plot_vel(t_array,v_array,"vvB_"+str(Nt),label="sim")
