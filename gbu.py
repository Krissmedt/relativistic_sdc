import numpy as np

c = 1
gamma = np.linspace(1.005,10,100,dtype=np.float)
gamma = np.array([1.0005,1.00005,1.000005,1.0000005,1.00000005])

for i in range(0,gamma.shape[0]):
    print("gamma = {0}".format(gamma[i]))
    beta = np.sqrt(1-1./gamma[i]**2.)
    u = beta*c

    print("beta = {0}".format(beta))
