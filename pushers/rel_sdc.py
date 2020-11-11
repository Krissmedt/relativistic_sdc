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
        self.qe = q

        self.c = c

        self.predictor = False
        if "predictor" in kwargs:
            if kwargs["predictor"] == True:
                self.predictor = True

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

        self.Rx = np.zeros((K+1,M),dtype=np.float)
        self.Rv = np.zeros((K+1,M),dtype=np.float)

        self.IV = np.zeros((nq,3),dtype=np.float)
        self.IF = np.zeros((nq,3),dtype=np.float)


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

            s.Rx[k,m-1] = np.linalg.norm(s.x[0,:,:] + qvsum - s.x[m,:,:])
            s.Rv[k,m-1] = np.linalg.norm(s.u[0,:,:] + qfsum - s.u[m,:,:])



def boris_SDC(pos,vel,coll):
    M = coll.M
    K = coll.K
    nq = coll.nq
    c = coll.c
    qe = coll.qe

    #Remap collocation weights from [0,1] to [tn,tn+1]
    weights =  coll.weights

    q =  coll.Qmat

    dm =  coll.delta_m

    ## Populate node solutions with x0, v0, F0 ##
    coll.x[0,:,:] = pos
    coll.u[0,:,:] = vel
    coll.F[0,:,:] = F(vel,E(pos,q=qe),B(pos,q=qe),c=c)

    coll.xn[0,:,:] = coll.x[0,:,:]
    coll.un[0,:,:] = coll.u[0,:,:]
    coll.Fn[0,:,:] = coll.F[0,:,:]

    coll.IV = 0
    coll.IF = 0

    ###### Initial Step #########################
    v_half = vel + 0.5*dm[0]*F(vel,E(pos,q=qe),B(pos,q=qe),c=c)
    coll.x[1,:,:] = pos + dm[0]*G(v_half,c=c)

    En         = 0.5*(E(pos) + E(coll.x[1,:,:]))*qe
    Bn         = B(coll.x[1,:,:])*qe
    gamma      = gu(coll.u[0,:,:],c=c)
    c_1        = 0.5*dm[0]*np.cross(G(coll.u[0,:,:],c=c), B(coll.x[0,:,:]))*qe
    c_2        = -(0.5*dm[0]/gamma)*np.cross(coll.u[0,:,:], Bn) + c_1
    coll.u[1,:,:] = boris_daniel(coll.u[0,:,:],En,Bn,dm[0],c_2,gamma,q=1)
    coll.F[1,:,:] = F(coll.u[0,:,:],E(coll.x[0,:,:]),B(coll.x[0,:,:]),c=c)

    if coll.predictor == False:
        coll.x[1,:,:] = coll.x[0,:,:]
        coll.u[1,:,:] = coll.u[0,:,:]
        coll.F[1,:,:] = coll.F[0,:,:]

    ############################################
    ######## Predictor Step ####################
    for m in range(1,M):
        v_half = coll.u[m,:,:] + 0.5*dm[m]*coll.F[m,:,:]
        coll.x[m+1,:,:] = coll.x[m,:,:] + dm[m]*G(v_half,c=c)

        En         = 0.5*(E(coll.x[m,:,:]) + E(coll.x[m+1,:,:]))*qe
        Bn         = B(coll.x[m,:,:])*qe
        gamma      = gu(coll.u[m,:,:],c=c)
        c_1        = 0.5*dm[m]*np.cross(G(coll.u[m,:,:],c=c), B(coll.x[m,:,:]))*qe
        c_2        = -(0.5*dm[m]/gamma)*np.cross(coll.u[m,:,:], Bn) + c_1
        coll.u[m+1,:,:] = boris_daniel(coll.u[m,:,:],En,Bn,dm[m],c_2,gamma,q=1)
        coll.F[m+1,:,:] = F(coll.u[m+1,:,:],E(coll.x[m+1,:,:]),B(coll.x[m+1,:,:]),c=c)

        if coll.predictor == False:
            coll.x[m+1,:,:] = coll.x[m,:,:]
            coll.u[m+1,:,:] = coll.u[m,:,:]
            coll.F[m+1,:,:] = coll.F[m,:,:]

    coll.calc_residual_2018(0)

    for k in range(1,K+1):
        coll.IV = 0
        coll.IF = 0

        coll.xn[1,:,:] = coll.x[0,:,:] + coll.IV

        En         = 0.5*(E(coll.x[0,:,:]) + E(coll.xn[1,:,:]))*qe
        Bn         = B(coll.xn[1,:,:])*qe
        gamma      = gu(coll.u[1,:,:],c=c)
        c_1        = 0.5*dm[0]*np.cross(G(coll.u[0,:,:],c=c), B(coll.x[0,:,:]))*qe
        c_1       += -0.5*dm[0]* (coll.F[0,:,:] + coll.F[1,:,:])
        c_1       += coll.IF
        c_2        = -(0.5*dm[0]/gamma)*np.cross(coll.u[0,:,:], Bn)
        coll.un[1,:,:] = boris_daniel(coll.u[0,:,:],En,Bn,dm[0],c_2,gamma,q=1)
        coll.Fn[1,:,:] = F(coll.un[1,:,:],E(coll.xn[1,:,:]),B(coll.xn[1,:,:]),c=c)

        if coll.predictor == False:
            coll.xn[1,:,:] = coll.x[1,:,:]
            coll.un[1,:,:] = coll.u[1,:,:]
            coll.Fn[1,:,:] = coll.F[1,:,:]

        for m in range(coll.ssi,M):
            # Calculate collocation terms required for pos update
            coll.IV = 0
            for j in range(1,M+1):
                coll.IV += (q[m+1,j]-q[m,j])*G(coll.u[j,:,:],c=c)

            v_half = coll.u[m,:,:] + 0.5*dm[m]*coll.F[m,:,:]
            vn_half = coll.un[m,:,:] + 0.5*dm[m]*coll.Fn[m,:,:]

            ### POSITION UPDATE FOR NODE m/SWEEP k ###
            coll.xn[m+1,:,:] = coll.xn[m,:,:]
            coll.xn[m+1,:,:] += dm[m]* (G(vn_half,c=c)-G(v_half,c=c))
            coll.xn[m+1,:,:] += coll.IV

            # Calculate collocation terms required for pos update
            coll.IF = 0
            for j in range(1,M+1):
                coll.IF += (q[m+1,j]-q[m,j])*coll.F[j,:,:]

            En         = 0.5*(E(coll.xn[m+1,:,:]) + E(coll.xn[m,:,:]))*qe
            Bn         = B(coll.xn[m+1,:,:])*qe
            gamma      = gu(coll.u[m+1,:,:],c=c)

            c_1        = 0.5*dm[m]*np.cross(G(coll.un[m,:,:],c=c), B(coll.xn[m,:,:]))*qe
            c_1       += -0.5*dm[m]* (coll.F[m+1,:,:] + coll.F[m,:,:])
            c_1       += coll.IF
            c_2        = -(0.5*dm[m]/gamma)*np.cross(coll.un[m,:,:], Bn) + c_1
            coll.un[m+1,:,:] = boris_daniel(coll.un[m,:,:],En,Bn,dm[m],c_2,gamma,q=qe)
            coll.Fn[m+1,:,:] = F(coll.un[m+1,:,:],E(coll.xn[m+1,:,:]),B(coll.xn[m+1,:,:]),c=c)
            # sol = scop.root(rootF,v_old.ravel(),args=(v_old,half_E,Bn,dm[m]),tol=10**-14,jac=False)
            # v_new = sol.x.reshape(v_old.shape)



        coll.F = np.copy(coll.Fn[:,:,:])
        coll.x = np.copy(coll.xn[:,:,:])
        coll.u = np.copy(coll.un[:,:,:])
        coll.calc_residual_2018(k)

    pos = np.copy(coll.xn[m+1,:,:])
    vel = np.copy(coll.un[m+1,:,:])

    return pos, vel, coll


def Gdot(um,Fm):
    betasq = np.sum((um/1)**2,axis=1)[:,np.newaxis]
    gdot = Fm/gu(um,c=coll.c)[:,np.newaxis] - betasq * Fm/np.power(gu(um,c=coll.c)[:,np.newaxis],3)

    return gdot

def rootF(U,*args):
    u0 = args[0]
    E = args[1]
    B = args[2]
    dt = args[3]
    coll = args[4]

    f = U - FXV(U,u0,E,B,dt,coll) - u0.ravel()

    return f

def FXV(U,u0,E,B,dt,coll):
    u = U.reshape(u0.shape)
    # f = (E+np.cross(G(u0,c=coll.c),B) + (E+np.cross(G(u,c=coll.c),B))) *dt/2
    f = coll.qe*(E+np.cross((G(u0,c=coll.c)+G(u,c=coll.c))/2,B)) *dt
    F = f.ravel()

    return F
