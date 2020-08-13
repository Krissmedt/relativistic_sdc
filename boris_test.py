import numpy as np
import random as rand
import scipy.optimize as scop
from rvv_pushers import *
from rvv_functions import *

def rootF(U,*args):
    u0 = args[0]
    E = args[1]
    B = args[2]
    dt = args[3]

    f = U - FXV(U,u0,E,B,dt) - u0.ravel()

    return f

def FXV(U,u0,E,B,dt):
    u = U.reshape(u0.shape[0],u0.shape[1])
    # f = (E+np.cross(G(u0),B) + (E+np.cross(G(u),B))) *dt/2
    f = (E+np.cross((G(u0)+G(u))/2,B)) *dt
    F = f.ravel()

    return F

def G(u):
    gamma = gu(u)[:,np.newaxis]
    v = u/gamma

    return v

nq = 4
c = 1
dt = 0.1
gamma = 5.
beta = np.sqrt(1-1./gamma**2.)
v_max = beta
u_max = gamma*v_max

B = []
E = []
x = []
u = []

for i in range(0,nq):
    Bi = np.array([rand.random(),rand.random(),rand.random()])
    Ei = np.array([rand.random(),rand.random(),rand.random()])
    xi  = np.array([rand.random(),rand.random(),rand.random()])
    ui  = np.array([rand.random(),rand.random(),rand.random()])
    ui = ui/np.linalg.norm(ui,axis=0) * u_max

    B.append(Bi)
    E.append(Ei)
    x.append(xi)
    u.append(ui)

B = np.array(B)
E = np.array(E)
x = np.array(x)
u = np.array(u)


sol = scop.root(rootF,u.ravel(),args=(u,E,B,dt),tol=10**-14,jac=False)
u_sol = sol.x.reshape(u.shape)
sol = scop.root(rootF,u.ravel(),args=(u,E,B,dt*0.5),tol=10**-14,jac=False)
u_sol_half = sol.x.reshape(u.shape)
yt_sol = gu(u_sol_half)[:,np.newaxis]

print(yt_sol)
u_boris = boris(x,u,E,B,dt,q=1)
u_hicary = hicary(x,u,E,B,dt,q=1)

# print(yt_sol)
# print("Defect of scipy solution for form A (F_old+F_new)/2")
# defect_solA = u_sol - u - (E+np.cross(G(u),B) + (E+np.cross(G(u_sol),B))) *dt/2
# print(defect_solA)
#
# print("Defect of scipy solution for form B (u_old+u_new)/(2*gamma_half)")
# yt_sol = gu(u_sol_half)[:,np.newaxis]
# defect_solB = u_sol - u - (E+np.cross((u+u_sol)/(2*yt_sol),B))*dt
# print(defect_solB)
#
# print("Defect of scipy solution for form C (u_old/gamma_old+u_new/gamma_new)/2")
# defect_solC = u_sol - u - (E+np.cross((G(u)+G(u_sol))/2,B))*dt
# print(defect_solC)
#
# print("Defect of non-relativistic Boris")
# u_new = boris_nr(u, E, B, dt)
# defect = u_new - u - (E+np.cross((u+u_new)/2,B))*dt
# print(defect)
#
#
# print("Defect of relativistic Boris for form A (F_old+F_new)/2")
# u_relA = boris(x,u,E,B,dt,q=1)
# defect_relA = u_relA - u - (E+np.cross(G(u),B) + (E+np.cross(G(u_relA),B))) *dt/2
# print(defect_relA)
#
# print("Defect of relativistic Boris for form B (u_old+u_new)/(2*gamma_half)")
# u_relB = boris(x,u,E,B,dt,q=1)
# defect_relB = u_relB - u - (E+np.cross((u+u_relB)/(2*yt_sol),B))*dt
# print(defect_relB)
#
# print("Defect of relativistic Boris for form C (u_old/gamma_old+u_new/gamma_new)/2")
# u_relC = boris(x,u,E,B,dt,q=1)
# defect_relC = u_relC - u - (E+np.cross((G(u)+G(u_relC))/2,B))*dt
# print(defect_relC)
#
# print("Defect of Hicary for form C (u_old/gamma_old+u_new/gamma_new)/2")
# u_hicary = hicary(x,u,E,B,dt,q=1)
# defect_hicary = u_hicary - u - (E+np.cross((G(u)+G(u_hicary))/2,B))*dt
# print(defect_hicary)
