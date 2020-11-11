import numpy as np
from rvv_io import *
from rvv_functions import *
from rvv_solution import *
from rvv_pushers import *
from rvv_fields import *
from rel_col18 import implicit_coll
from rel_sdc2 import *
from gauss_legendre import CollGaussLegendre
from gauss_lobatto import CollGaussLobatto

sims = [10,20,40,80,160,320]
# sims = [100]
tend = 1

M = 3
K = 2

c = 29979
q = 1

# gamma_max = 1.0000000000005
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

    # gamma = gu(vel,c=c)
    # lfreq = -q*Bfield/(1*c*gamma)
    # larmor = vel[:,1]/gamma/lfreq
    # #larmor = 1*vel[:,1]/(-q*B)
    # pos[:,0] = larmor

    t = 0

    x_array = [pos]
    x2_array = [pos]
    v_array = [vel]
    t_array = [t]

    col = coll(CollGaussLobatto,dt,nq,K=K,M=M,c=c,q=q,predictor=True)
    rx_array = [np.linalg.norm(col.Rx,axis=1)]
    rv_array = [np.linalg.norm(col.Rv,axis=1)]

    # Collocation solution stuff
    posc = np.copy(pos)
    velc = np.copy(vel)
    colc = coll(CollGaussLobatto,dt,nq,M=5,K=1,c=c,q=q)

    for ti in range(1,Nt+1):
        t = ti*dt

        pos, vel, col = boris_SDC(pos,vel,col)
        # posc, velc, colc = implicit_coll(posc,velc,colc)
        rx_array.append(np.linalg.norm(col.Rx,axis=1))
        rv_array.append(np.linalg.norm(col.Rv,axis=1))
        x2_array.append(posc)
        x_array.append(pos)
        v_array.append(vel)
        t_array.append(t)

    # colc.calc_residual_2018(1)
    # col.calc_residual_2018(4)
    # errorx = np.abs(col.x[2:,0,:]-np.around(colc.x[2:,0,:],14))/np.abs(np.around(colc.x[2:,0,:],14))
    # errorf = np.abs(col.F[2:,0,:]-np.around(colc.F[2:,0,:],14))/np.abs(np.around(colc.F[2:,0,:],14))
    # erroru = np.abs(col.u[2:,0,:]-np.around(colc.u[2:,0,:],14))/np.abs(np.around(colc.u[2:,0,:],14))
    # print("Diff in x: {0}".format(errorx))
    # print("Diff in F: {0}".format(errorf))
    # print("Diff in u: {0}".format(erroru))
    # print("SDC solution: {0}".format(col.Rv))
    # print("Collocation solution: {0}".format(colc.Rv))
    rx_array = np.array(rx_array)
    rv_array = np.array(rv_array)
    x_array = np.array(x_array)
    x2_array = np.array(x2_array)
    v_array = np.array(v_array)
    t_array = np.array(t_array)

    if col.predictor == True:
        rhs = (M-1)*(K+1)*Nt
    else:
        rhs = (M-1)*K*Nt

    wp_dump(t_array,x_array,v_array,dt,"sdc_M{0}K{1}_wp_vvrel.h5".format(M,K),rhs=rhs,new=new)
    new = False

plot_xres(t_array,rx_array,"sdc_M{0}K{1}_".format(M,K)+str(Nt))
plot_vres(t_array,rv_array,"sdc_M{0}K{1}_".format(M,K)+str(Nt))
plot_isotraj(x_array,"sdc_"+str(Nt),label="sim")
# plot_isotraj(x2_array,"col2_"+str(Nt),label="sim")
plot_vel(t_array,v_array,"sdc_"+str(Nt),label="sim")
