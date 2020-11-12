import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

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
