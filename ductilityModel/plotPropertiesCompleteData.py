import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator
import re

figureOutput='BindingEnergies'
error_on=0

SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

rcParams['axes.linewidth']   = 2
rcParams['figure.figsize']   = 5,4
rcParams['figure.dpi']       = 600

rcParams['xtick.major.size']  = 7
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size']  = 5
rcParams['xtick.minor.width'] = 1
rcParams['ytick.major.size']  = 7
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size']  = 5
rcParams['ytick.minor.width'] = 1

header=["chemical formula","Formation enthalpy, $E_f$ [eV/atom]","Lattice constant, $a_0$ [$\\AA$]","Lattice mismatch, $\\delta$ [%]",\
        "VEC [e$^{-}$]","average displacement, $\\langle u \\rangle$, [$\\AA$]",\
        " $l_{2,1} norm$ [$\\AA$]",  "$\\langle u \\rangle/l_{2,1} norm$", "LLD [e$^{-}]$",\
        "Total occupied states, $g_{occ}$ [e$^{-}$]","total bonding states, $g_{bond}$ [e$^{-}$]",\
        r"$\frac{g_{occ}}{g_{bond}}$","Hartigan's dip, $dip$","$\epsilon_p$ [%]"]
colors=['blueviolet','g','r','limegreen','pink','orangered','k','navy']
markers=['o','D','>','h','d','+','x','*']

#epsilon_p = peak true strain before fracture
#BSD = bonding state depletion = gocc/gbond
def ep_polymodel(BSD):
    return( -13706.12*(BSD)**3 + 38196.22*(BSD)**2 - 35519.70*(BSD) + 11025.99 )

fig1, axes1 = plt.subplots()

fig, axes = plt.subplots(3, 4, figsize=(16, 9))
axes=axes.flatten()

cindex=-1
for idir,dir in enumerate(['/pscratch/sd/k/kcpitike/MPEA/binaries/','/pscratch/sd/k/kcpitike/MPEA/ternaries/','/pscratch/sd/k/kcpitike/MPEA/quadrinary/3x3x3/']):
    datafile=dir+'data.csv'
    df = pd.read_csv(datafile, delimiter=",")  # Specify a semicolon delimiter
    data=df.to_numpy()

    #text only
    text_only = np.array([re.sub(r'\d+', '', element) for element in data[:,0]])
    combinations=np.unique(text_only)

    #numbers only
    numbers_only = np.array([list(map(int, re.findall(r'\d+', item))) for item in data[:,0]])
    row_sums = numbers_only.sum(axis=1, keepdims=True)  # Sum along rows
    numbers_only = numbers_only / row_sums

    unique=np.unique(data[:,0])
    #print(unique)

    #header in data.csv file
    #0  chemical formula,
    #1  "formation enthalpy, E_f"
    #2  "lattice constant, a0",
    #3  "lattice mismatch, delta",
    #4  "valence electron concentration, vec",
    #5  "average displacement, <u>",
    #6  "displacement l21 norm, l21_norm_u",
    #7  "<u>/l21_norm_u"
    #8  "local lattice displacement, lld",
    #9  "total occupied states, g_occ",
    #10 "total bonding states, g_bond",
    #11 g_occ/g_bond,
    #12 "Hartigan's dip, dip"
    #13 "epsilon_p [%]"


    subPlotList=[1,2,3,4,5,6,7,8,11,12,13,10]
    for icombination, combination in enumerate(combinations):
        cindex+=1
        # Find indices of all instances of the target
        indices   = [i for i, x in enumerate(text_only) if x == combination]
        selected_rows = data[indices]
        selected_frac = numbers_only[indices,-1]
        for isubPlot, subPlot in enumerate(subPlotList):
            axes[isubPlot].scatter(selected_frac,selected_rows[:,subPlotList[isubPlot]],facecolor='none',edgecolors=colors[cindex],marker=markers[cindex])

            # Calculate mean at each unique fraction
            unique_fracs = np.unique(selected_frac)
            mean_values = [
                np.mean(selected_rows[selected_frac == frac, subPlotList[isubPlot]]) 
                for frac in unique_fracs
            ]
            
            # Plot the mean values
            axes[isubPlot].plot(unique_fracs, mean_values, c=colors[cindex], marker=markers[cindex],label=combination)

            #axes[isubPlot].plot(selected_frac,np.mean(selected_rows[:,subPlotList[isubPlot]]),c=colors[icombination],marker=markers[icombination])
            axes[isubPlot].legend(frameon=0)
            axes[isubPlot].set_xlabel('W atomic fraction')
            axes[isubPlot].set_ylabel(header[subPlotList[isubPlot]])


            axes[isubPlot].tick_params(axis='both',which='both',direction='in',colors='k',\
                        bottom=True,top=True,left=True,right=True,\
                        #labelbottom=True, labeltop=True, labelleft=True, labelright=True,\
                        labelrotation=0)
            axes[isubPlot].xaxis.set_minor_locator(AutoMinorLocator())
            axes[isubPlot].yaxis.set_minor_locator(AutoMinorLocator())
            axes[isubPlot].grid(alpha=0.4,which='major',linewidth=0.5)
            axes[isubPlot].grid(alpha=0.2,which='minor',linewidth=0.2)
            axes[isubPlot].set_xlim([0.2,1.0])
            axes[isubPlot].set_xticks(np.arange(2,11,1)*0.1)
            #axes[isubPlot].set_xlim([0.3,0.55])

            #when reading 11th column plot epsilon_p from model fit
            if (subPlot==10):
                # Filter both lists
                lower_bound=0.8
                upper_bound=0.975
                filtered_data = [(num, val) for num, val in zip(mean_values, unique_fracs) if lower_bound <= num <= upper_bound]

                # Check if any numbers satisfy the condition
                if filtered_data:
                    filtered_y, filtered_x = zip(*filtered_data)
                    #filtered=[num for num in mean_values if 0.8 <= num <= 0.975]
                    axes1.plot(filtered_x, [ep_polymodel(y) for y in filtered_y]  ,label=combination, c=colors[icombination], marker=markers[icombination])
                    axes1.legend(frameon=0)
                    axes1.set_xlabel('W atomic fraction')
                    axes1.set_ylabel('$\epsilon_p [\%]$')
                    axes1.set_xlim([0.3,0.55])
                    axes1.set_ylim([-1,20])
                    axes1.xaxis.set_minor_locator(AutoMinorLocator())
                    axes1.yaxis.set_minor_locator(AutoMinorLocator())
                    axes1.grid(alpha=0.4,which='major',linewidth=0.5)
                    axes1.grid(alpha=0.2,which='minor',linewidth=0.2)
                    axes1.tick_params(axis='both',which='both',direction='in',colors='k',\
                            bottom=True,top=True,left=True,right=True,\
                            #labelbottom=True, labeltop=True, labelleft=True, labelright=True,\
                            labelrotation=0)
                    axes[isubPlot].set_xlabel('W atomic fraction')
                    axes[isubPlot].set_ylabel(header[subPlotList[isubPlot]])

plt.tight_layout()
fig.savefig('properties.png')

fig1.savefig('model.png')

