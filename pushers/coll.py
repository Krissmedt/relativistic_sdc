import numpy as np
from gammafac import G
from gauss_legendre import CollGaussLegendre
from gauss_lobatto import CollGaussLobatto

class coll:
    def __init__(self,collclass,dt,nq,M=3,K=3,**kwargs):
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


    def calc_residual_2018(self,k,c=1):
        s = self
        q =  self.Qmat
        M = s.M

        for m in range(1,M+1):
            qvsum = 0
            qfsum = 0
            for j in range(1,M+1):
                qvsum += q[m,j] * G(s.u[j,:,:],c=c)
                qfsum += q[m,j] * s.F[j,:,:]

            s.Rx[k,m-1] = np.linalg.norm(s.x[0,:,:] + qvsum - s.x[m,:,:])
            s.Rv[k,m-1] = np.linalg.norm(s.u[0,:,:] + qfsum - s.u[m,:,:])
