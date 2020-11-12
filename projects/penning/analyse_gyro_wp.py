from math import sqrt, fsum, pi, exp, cos, sin, floor
from decimal import Decimal
import io
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import h5py as h5
import sys
import traceback
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

from tools.writing import *
from tools.plotting import *

data_root = "./"
fig_name = "beta_9999"
testcase = "A"

filenames = {}

filenames["Ref"] = "sdc_M3K2_wp_{0}.h5".format(testcase)
# filenames["Leapfrog"] = "lf_wp_vvrel.h5"
# filenames["Velocity-Verlet"] = "vvA_wp_{0}.h5".format(testcase)
# filenames["Boris-SDC M3K1"] = "sdc_M3K1_wp_{0}.h5".format(testcase)
filenames["Boris-SDC M3K2"] = "sdc_M3K2_wp_{0}.h5".format(testcase)
# filenames["Boris-SDC M5K1"] = "sdc_M5K1_wp_{0}.h5".format(testcase)
# filenames["Boris-SDC M5K2"] = "sdc_M5K2_wp_{0}.h5".format(testcase)
# filenames["Boris-SDC M5K3"] = "sdc_M5K3_wp_{0}.h5".format(testcase)
# filenames["Boris-SDC M5K4"] = "sdc_M5K4_wp_{0}.h5".format(testcase)
# filenames["Boris-SDC M5K5"] = "sdc_M5K5_wp_{0}.h5".format(testcase)
# filenames["Boris-SDC M5K6"] = "sdc_M5K6_wp_{0}.h5".format(testcase)
# filenames["Boris-SDC M5K7"] = "sdc_M5K7_wp_{0}.h5".format(testcase)
# filenames["Boris-SDC M5K5"] = "sdc_M5K5_wp_vvrel.h5"


plot_params = {}
plot_params['legend.fontsize'] = 16
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 20
plot_params['axes.titlesize'] = 20
plot_params['xtick.labelsize'] = 16
plot_params['ytick.labelsize'] = 16
plot_params['lines.linewidth'] = 4
plot_params['axes.titlepad'] = 5
plot_params['legend.loc'] = 'upper right'
plt.rcParams.update(plot_params)
r = 1
b = 1

for key,value in filenames.items():
    file = h5.File(data_root+value,'r')
    Nt = file["fields/Nt"][:]
    rhs = file["fields/rhs"][:]
    dt = file["fields/dt"]
    times = file["fields/t"]
    x = file["fields/pos"]
    v = file["fields/vel"]
    x0 = file["fields/x0"]
    v0 = file["fields/v0"]

    if key == "Ref":
        solx = x
        solv = v
        continue

    ref_errors = np.abs(solx[-1,:,:]-x[:,:,:])/np.abs(solx[-1,:,:])
    ref_errors = np.linalg.norm(ref_errors,axis=1)
    ref_errors = np.linalg.norm(ref_errors,axis=1)

    refv_errors = np.abs(solv[-1,:,:]-v[:,:,:])/np.abs(solv[-1,:,:])
    refv_errors = np.linalg.norm(refv_errors,axis=1)
    refv_errors = np.linalg.norm(refv_errors,axis=1)

    # ref_errors = np.abs(x[-1,:,:]-x[:-1,:,:])/np.abs(x[-1,:,:])
    # ref_errors = np.linalg.norm(ref_errors,axis=1)
    # ref_errors = np.linalg.norm(ref_errors,axis=1)
    #
    # refv_errors = np.abs(v[-1,:,:]-v[:-1,:,:])/np.abs(v[-1,:,:])
    # refv_errors = np.linalg.norm(refv_errors,axis=1)
    # refv_errors = np.linalg.norm(refv_errors,axis=1)

    xfactors = np.log2(ref_errors[:-1]/ref_errors[1:])
    vfactors = np.log2(refv_errors[:-1]/refv_errors[1:])
    print(key+" x order: {0}".format(xfactors))
    print(key+" v factors: {0}".format(vfactors))
    print(rhs)


    if key == "Velocity-Verlet":
        c = "black"
    if "Boris-SDC M3" in key:
        sims = 2
        c = (0,0,b)
        b -= 1/sims
    if "Boris-SDC M5" in key:
        sims = 7
        c = (r,0,b)
        r -= np.round(0.7/sims,2)

    label = key
    ##Order Plot w/ rhs
    fig_rhs = plt.figure(1)
    ax_rhs = fig_rhs.add_subplot(1, 1, 1)
    ax_rhs.plot(rhs,ref_errors,marker="o",color=c,label=label)
    ax_rhs.set_ylabel(r'$\Delta x^{\mathrm{rel}}$')

    ##Order Plot w/ Nt
    fig_nt = plt.figure(2)
    ax_nt = fig_nt.add_subplot(1, 1, 1)
    ax_nt.plot(Nt,ref_errors,marker="o",color=c,label=label)
    ax_nt.set_ylabel(r'$\Delta x^{\mathrm{rel}}$')

    ##Velocity Order Plot w/ Nt
    fig_nt_v = plt.figure(3)
    ax_nt_v = fig_nt_v.add_subplot(1, 1, 1)
    ax_nt_v.plot(Nt,refv_errors,marker="o",color=c,label=label)
    ax_nt_v.set_ylabel(r'$\Delta v^{\mathrm{rel}}$')

handles, labels = fig_rhs.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax_rhs.legend(by_label.values(), by_label.keys(),loc='lower left')

handles, labels = fig_nt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax_nt.legend(by_label.values(), by_label.keys(),loc='lower left')

handles, labels = fig_nt_v.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax_nt_v.legend(by_label.values(), by_label.keys(),loc='lower left')


axnl_list = []
axnl_list.append(ax_rhs)
axnl_list.append(ax_nt)
axnl_list.append(ax_nt_v)

i = 0
for ax in axnl_list:
    i +=1
    if i == 1:
        orderSlope = -1
        ax.set_xlabel('RHS evaluations')
    else:
        ax.set_xlabel(r'$N t$')
        orderSlope = -1

    ax.set_xscale('log')
    #ax_rhs.set_xlim(10**3,10**5)
    ax.set_yscale('log')
    ax.set_ylim(10**(-15),10**(5))

    xRange = ax.get_xlim()
    yRange = ax.get_ylim()

    ax.plot(xRange,orderLines(1*orderSlope,xRange,yRange),
                ls='dashdot',c='0.2')
    ax.plot(xRange,orderLines(2*orderSlope,xRange,yRange),
                ls='dotted',c='0.4')
    ax.plot(xRange,orderLines(4*orderSlope,xRange,yRange),
                ls='dashed',c='0.6')
    ax.plot(xRange,orderLines(8*orderSlope,xRange,yRange),
                ls='solid',c='0.8')

fig_rhs.savefig(data_root + 'pen_x_'+ fig_name + '_rhs.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
fig_nt.savefig(data_root + 'pen_x_' + fig_name + '_nt.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
fig_nt_v.savefig(data_root + 'pen_v_' + fig_name + '_nt.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
