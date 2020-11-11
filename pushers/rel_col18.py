import numpy as np
import scipy.optimize as scop
from rvv_functions import *
from rvv_fields import *
from rvv_pushers import *
from gauss_legendre import CollGaussLegendre
from gauss_lobatto import CollGaussLobatto
class coll:
    def __init__(self,collclass,dt,nq,M=3,K=3,q=-1,c=1,**kwargs):
        self.collclass = collclass
        coll = self.collclass(M,0,1)

        self.K = K
        self.M = M

        self.nodes = coll._getNodes
        self.weights = coll._getWeights(coll.tleft,coll.tright) #Get M  nodes and weights

        self.Qmat = coll._gen_Qmatrix           #Generate q_(m,j), i.e. the large weights matrix
        self.Smat = coll._gen_Smatrix           #Generate s_(m,j), i.e. the large node-to-node weights matrix

        self.delta_m = coll._gen_deltas         #Generate vector of node spacings

        self.Qmat *= dt
        self.Smat *= dt
        self.delta_m *= dt

        self.ssi = 1

        self.nq = nq

        self.c = c
        self.qe = q

        #Collocation solution stuff
        Ix = np.array([1,0])
        Iv = np.array([0,1])
        Ixv = np.array([[0,1],[0,0]])
        Id = np.identity(nq*3)
        I2d = np.identity(nq*3*2)

        self.Ix = Ix
        self.Iv = Iv
        self.Ixv = Ixv
        self.Id = Id

        Qtil = self.Qmat[1:,1:]
        I3M = np.identity(3*M)
        self.Q = np.kron(np.identity(2),np.kron(Qtil,Id))

        #Define required calculation matrices
        QE = np.zeros((M+1,M+1),dtype=np.float)
        QI = np.zeros((M+1,M+1),dtype=np.float)
        QT = np.zeros((M+1,M+1),dtype=np.float)

        SX = np.zeros((M+1,M+1),dtype=np.float)

        for i in range(0,M):
            QE[(i+1):,i] = self.delta_m[i]
            QI[(i+1):,i+1] = self.delta_m[i]

        QT = 1/2 * (QE + QI)
        QX = QE @ QT + (QE*QE)/2
        SX[:,:] = QX[:,:]
        SX[1:,:] = QX[1:,:] - QX[0:-1,:]

        self.SX = SX
        self.SQ = self.Smat @ self.Qmat

        d = 3*nq

        self.x0 = np.zeros((M+1,nq,3),dtype=np.float)
        self.x = np.zeros((M+1,nq,3),dtype=np.float)
        self.xn = np.zeros((M+1,nq,3),dtype=np.float)

        self.u0 = np.zeros((M+1,nq,3),dtype=np.float)
        self.u = np.zeros((M+1,nq,3),dtype=np.float)
        self.un = np.zeros((M+1,nq,3),dtype=np.float)

        self.F = np.zeros((M+1,nq,3),dtype=np.float)
        self.Fn = np.zeros((M+1,nq,3),dtype=np.float)

        self.x_con = np.zeros((K,M))
        self.x_res = np.zeros((K,M))
        self.u_con = np.zeros((K,M))
        self.u_res = np.zeros((K,M))

        self.Rx = np.zeros((K,M),dtype=np.float)
        self.Rv = np.zeros((K,M),dtype=np.float)


    def calc_residual_2018(self,k):
        s = self
        q =  self.Qmat
        M = s.M

        for m in range(1,M+1):
            qvsum = 0
            qfsum = 0
            for j in range(1,M+1):
                qvsum += q[m,j] * G(s.u[j,:,:],c=self.c)
                qfsum += q[m,j] * s.F[j,:,:]
            s.Rx[k-1,m-1] = np.linalg.norm(s.x[0,:,:] + qvsum - s.x[m,:,:])
            s.Rv[k-1,m-1] = np.linalg.norm(s.u[0,:,:] + qfsum - s.u[m,:,:])


def implicit_coll(pos,vel,coll):
    M = coll.M
    nq = coll.nq
    #Remap collocation weights from [0,1] to [tn,tn+1]
    weights =  coll.weights

    for m in range(0,M+1):
        coll.x[m,:,:] = pos
        coll.u[m,:,:] = vel

    Id = coll.Id
    Ix = coll.Ix
    Iv = coll.Iv

    U0 = np.append(coll.x[1:,:,:].ravel(),coll.u[1:,:,:].ravel())
    FU = FXV(U0,coll)
    sol = scop.root(rootF,U0,args=(coll,U0),tol=10**-14,jac=False)
    U = sol.x
    md = M*coll.nq*3
    coll.x[1:,:,:] = U[0:md].reshape((M,coll.nq,3))
    coll.u[1:,:,:] = U[md:].reshape((M,coll.nq,3))
    coll.calc_residual_2018(1)

    pos = coll.x[-1,:,:]
    vel = coll.u[-1,:,:]

    return pos, vel, coll


def rootF(U,*args):
    coll = args[0]
    U0 = args[1]

    f = U - coll.Q @ FXV(U,coll) - U0

    return f


def FXV(U,coll):
    M = coll.M
    Id = coll.Id
    Ix = coll.Ix
    Iv = coll.Iv
    md = M*coll.nq*3
    x = U[0:md].reshape((M,coll.nq,3))
    v = U[md:].reshape((M,coll.nq,3))
    Fx = np.zeros((M,coll.nq,3),dtype=np.float)
    Fv = np.zeros((M,coll.nq,3),dtype=np.float)
    for m in range(0,M):
        Fx[m,:,:] = G(v[m,:,:],c=coll.c)
        Fv[m,:,:] = F(v[m,:,:],E(x[m,:,:],q=coll.qe),B(x[m,:,:],q=coll.qe),q=coll.qe,c=coll.c)

    coll.F[1:,:,:] = Fv
    FXV = np.append(Fx.ravel(),Fv.ravel())
    return FXV
