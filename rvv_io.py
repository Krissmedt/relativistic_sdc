import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

def plot_traj(x,name,label=""):

    fig_traj = plt.figure(1)
    ax_traj = fig_traj.add_subplot(111)
    ax_traj.plot(x[:,:,0],x[:,:,1],label=label)
    ax_traj.set_aspect('equal')
    ax_traj.legend()

    fig_xztraj = plt.figure(2)
    ax_xztraj = fig_xztraj.add_subplot(111)
    ax_xztraj.plot(x[:,:,0],x[:,:,2],label=label)
    ax_xztraj.legend()

    fig_traj.savefig(name+'_trajectory.png')
    fig_xztraj.savefig(name+'_xzSlice.png')


def plot_vel(t,vel,name,label=""):
    fig_xvel = plt.figure(3)
    ax_xvel = fig_xvel.add_subplot(111)
    ax_xvel.plot(t,vel[:,:,0],label=label)
    ax_xvel.set_xlim([0,t[-1]])
    ax_xvel.legend()

    fig_yvel = plt.figure(4)
    ax_yvel = fig_yvel.add_subplot(111)
    ax_yvel.plot(t,vel[:,:,1],label=label)
    ax_yvel.set_xlim([0,t[-1]])
    ax_yvel.legend()

    fig_zvel = plt.figure(5)
    ax_zvel = fig_zvel.add_subplot(111)
    ax_zvel.plot(t,vel[:,:,2],label=label)
    ax_zvel.set_xlim([0,t[-1]])
    ax_zvel.legend()

    fig_xvel.savefig(name+'_xvelocity.png')
    fig_yvel.savefig(name+'_yvelocity.png')
    fig_zvel.savefig(name+'_zvelocity.png')


def plot_isotraj(x,name,plim=1,label=""):

    fig_isotraj = plt.figure(6)
    ax = fig_isotraj.gca(projection='3d')
    for pii in range(0,plim):
        ax.plot3D(x[:,pii,0],
                  x[:,pii,1],
                  zs=x[:,pii,2])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    fig_isotraj.savefig(name+'_isotraj.png')


def plot_xres(t,xres,name,label=""):
    fig_xres = plt.figure(8)
    ax_xres = fig_xres.add_subplot(111)
    for k in range(0,xres.shape[1]):
        ax_xres.plot(t[1:],xres[1:,k],label=label+" K={0}".format(k))
    ax_xres.set_xlim([0,t[-1]])
    ax_xres.set_yscale('log')
    ax_xres.legend()

    fig_xres.savefig(name+'_xres.png')


def plot_vres(t,vres,name,label=""):
    fig_vres = plt.figure(9)
    ax_vres = fig_vres.add_subplot(111)
    for k in range(0,vres.shape[1]):
        ax_vres.plot(t[1:],vres[1:,k],label=label+" K={0}".format(k))
    ax_vres.set_xlim([0,t[-1]])
    ax_vres.set_yscale('log')
    ax_vres.legend()

    fig_vres.savefig(name+'_vres.png')


def wp_dump(t,pos,vel,dt,filename,rhs=0,new=False):
    if new == True:
        try:
            file = h5.File(filename,'w')
        except OSError:
            file.close()
            file = h5.File(filename,'w')

        grp = file.create_group('fields')
        grp.create_dataset('Nt',data=np.array(pos.shape[0]-1)[np.newaxis],maxshape=(None,))
        grp.create_dataset('rhs',data=np.array(rhs)[np.newaxis],maxshape=(None,))
        grp.create_dataset('dt',data=np.array(dt)[np.newaxis],maxshape=(None,))
        grp.create_dataset('pos',data=pos[np.newaxis,-1,:,:],maxshape=(None,None,None))
        grp.create_dataset('vel',data=vel[np.newaxis,-1,:,:],maxshape=(None,None,None))
        grp.create_dataset('t',data=t[np.newaxis,-1],maxshape=(None,))
        grp.create_dataset('x0',data=pos[0,:,:])
        grp.create_dataset('v0',data=vel[0,:,:])
        file.close()

    if new == False:
        try:
            file = h5.File(filename,'r+')
        except OSError:
            file.close()
            file = h5.File(filename,'r+')

        Nt = pos.shape[0]-1
        file["fields/Nt"].resize((file["fields/Nt"].shape[0]+1),axis=0)
        file["fields/Nt"][-1] = Nt

        file["fields/dt"].resize((file["fields/dt"].shape[0]+1),axis=0)
        file["fields/dt"][-1] = dt

        file["fields/rhs"].resize((file["fields/rhs"].shape[0]+1),axis=0)
        file["fields/rhs"][-1] = rhs

        file["fields/pos"].resize((file["fields/pos"].shape[0]+1),axis=0)
        file["fields/pos"][-1,:,:] = pos[-1,:,:]

        file["fields/vel"].resize((file["fields/vel"].shape[0]+1),axis=0)
        file["fields/vel"][-1,:,:] = vel[-1,:,:]

        file["fields/t"].resize((file["fields/t"].shape[0]+1),axis=0)
        file["fields/t"][-1] = t[-1]

        file.close()



def orderLines(order,xRange,yRange):
    if order < 0:
        a = yRange[1]/xRange[0]**order
    else:
        a = yRange[0]/xRange[0]**order

    oLine = [a*xRange[0]**order,a*xRange[1]**order]
    return oLine
