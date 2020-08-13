import numpy as np

def gu(u,c=1):
    # Lorentz factor calculated from proper / 4-velocity (u)
    gamma = np.sqrt(1+np.sum((u/c)**2,axis=1))
    # print(gamma)
    return gamma

def gv(v,c=1):
    # Lorentz factor calculated from coordinate velocity (v)
    gamma = np.sqrt(1/(1-np.sum((v/c)**2,axis=1)))

    return gamma

def G(um,c=1):
    gamma = gu(um,c=c)
    vm = um/gamma[:,np.newaxis]
    return vm

def Gdot(um,fm):
    c = 1
    gdot = (c**3 * fm) / np.power(c**2+np.power(um,2),3/2)
    return gdot

def Gdot2(um,fm):
    c = 1
    gdot = (c**3 * fm) / np.power(c**2+np.sum(um**2,axis=1)[:,np.newaxis],3/2)
    return gdot

def Gdot3(um,fm):
    c = 1
    gdot = c**4 * gu(um)[:,np.newaxis] * fm / np.power(c**2+um**2,2)
    return gdot

def Gdot4(um,fm):
    c = 1
    gdot = c**4 * gu(um)[:,np.newaxis] * fm / np.power(c**2+np.sum(um**2,axis=1)[:,np.newaxis],2)
    return gdot

def Gdot5(um,fm):
    c = 1
    gdot = c**4 * gu(um)[:,np.newaxis] * fm / np.sum(np.power(c**2+um**2,2),axis=1)[:,np.newaxis]
    return gdot
