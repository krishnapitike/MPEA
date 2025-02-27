# %% importing packages
##################################################################################################

from sys import argv
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
import pandas as pd
from scipy.integrate import trapezoid
import diptest
import os
import csv
#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr

# %% gaussian smoothing of density of states
##################################################################################################

def gaussianSmoothing2D(x_vals,y_vals,sigma=0.1):
    y_smth = np.zeros(y_vals.shape)
    for it in range(0,len(x_vals)):
        x_position      = x_vals[it]
        gaussian_kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        y_smth[it]      = np.sum(y_vals * gaussian_kernel)
    return(y_smth)


# %% parcing doscar
##################################################################################################
def parse_doscar(filename,sigma):
    """
    Parse the DOSCAR file and extract energies and DOS data.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Skip header lines and extract the number of atoms
    num_atoms = int(lines[0].split()[0])
    emax=float(lines[5].split()[0])
    emin=float(lines[5].split()[1])
    nedos=int(lines[5].split()[2])

    # Fermi energy is in the 6th line
    e_fermi = float(lines[5].split()[3])

    # DOS data starts after 6 lines of header + 1 line for energy range
    dos_data_start = 6
    dos_data_stop = 6 + nedos
    dos_data = np.loadtxt(lines[dos_data_start:dos_data_stop])

    #partial dos of d states
    for n in range(num_atoms):
        dos_data_start = dos_data_stop+1
        dos_data_stop = dos_data_start+nedos
        ldos_data=np.loadtxt(lines[dos_data_start:dos_data_stop])
        if n==0:
            d_ldos=np.sum(ldos_data[:, -5:], axis=1)
        else:
            d_ldos+=np.sum(ldos_data[:, -5:], axis=1)

    # Extract energies and DOS
    energy = dos_data[:, 0]-e_fermi  # First column is energy
    total_dos = dos_data[:, 1]/num_atoms  # Second column is total DOS
    integrated_dos = dos_data[:, 2]/num_atoms  # Third column is integrated DOS
    d_ldos=d_ldos/num_atoms # d states of all atoms

    d_ldos_smoothed = gaussianSmoothing2D(energy,d_ldos,sigma)

    return energy, total_dos, d_ldos, d_ldos_smoothed

# %% plotting dos
##################################################################################################
def plot_dos(energy, total_dos, d_ldos, d_ldos_smoothed, sigma, output_file=None):
    """
    Plot the density of states (DOS).
    """

    #calculating where is pseudogap
    fermiIndex=np.argmin(np.abs(energy))
    dosmaxIndex1, dosmaxIndex2 = np.argmax(d_ldos[:fermiIndex]), np.argmax(d_ldos[fermiIndex:])+fermiIndex
    dosminIndex=np.argmin(d_ldos[dosmaxIndex1:dosmaxIndex2])
    pgIndex=dosminIndex+dosmaxIndex1
    #calculating g_occ/g_bond
    #g_occ=np.sum(d_ldos[:fermiIndex])
    #g_bond=np.sum(d_ldos[:pgIndex])
    g_occ=trapezoid(d_ldos[:fermiIndex], x=energy[:fermiIndex])
    g_bond=trapezoid(d_ldos[:pgIndex], x=energy[:pgIndex])

    #calculating hartigan's dip
    # Target values
    targets = [-6, 4]
    #targets = [np.min(energy),np.max(energy)]

    # Find the indices of the closest elements
    indices = [np.abs(energy - t).argmin() for t in targets]
    print( indices,energy[indices[0]],energy[indices[1]] )

    dip, pval = diptest.diptest(d_ldos[indices[0]:indices[1]])
    dip_smoothed, pval_smoothed = diptest.diptest(d_ldos_smoothed[indices[0]:indices[1]])
    print(dip)


    # Import R's "diptest" package
    #diptestr = importr('diptest')

    # Perform the dip test
    #dipr = diptestr.dip_test(d_ldos_smoothed)
    #r_vector = robjects.FloatVector(d_ldos_smoothed[indices[0]:indices[1]])
    #dipr = diptestr.dip_test(r_vector)
    #print(dipr[0][0])

    plt.figure(figsize=(8, 6))
    plt.plot(energy , total_dos, label="Total DOS", color="k",alpha=0.2,linestyle=':')
    plt.plot(energy , d_ldos, label="d LDOS", color="blue")
    plt.plot(energy , d_ldos_smoothed, label="d LDOS smoothed", color="red",linestyle='--')
    plt.fill_between(energy[:fermiIndex] , d_ldos[:fermiIndex], color="blue", alpha=0.5, label="$g_{occ}$")
    plt.fill_between(energy[fermiIndex:pgIndex] , d_ldos[fermiIndex:pgIndex], color="red", alpha=0.5, label="$g_{bond}$")
    #plt.title('{:s}: $\\frac{{g_{{occ}}}}{{g_{{bond}}}} = {:0.3f} ~;~ dip= {:0.3f}$'.format(chemFormula,g_occ/g_bond,dip))
    #plt.plot(energy)

    plt.axvline(0, color='brown', linestyle='--', label="Fermi Level")  # Fermi level at 0
    plt.axvline(energy[pgIndex],color='black', linestyle=':', label="Pseudo gap")
    plt.axvline(energy[indices[0]],color='black', linestyle=':')
    plt.axvline(energy[indices[1]],color='black', linestyle=':')
    plt.xlabel("Energy (eV)")
    plt.ylabel("Density of States (states/eV/atom)")
    plt.title("BSD={:0.3f}, dip={:0.3f}\n sigma={:0.3f}, dip={:0.3f}".format(g_occ/g_bond,dip,sigma,dip_smoothed))
    plt.legend()
    plt.grid(alpha=0.5)

    if output_file:
        plt.savefig(output_file, dpi=300)
    #plt.show()

    return(g_occ,g_bond,dip)

# %% main function
##################################################################################################
def main():

    #setting header of data
    data=[]
    sigma=0.2
    #collecting all contcars
    doscars=glob('**/DOSCAR',recursive=1)
    ldosplot='ldos.png'

    #looping through all contcars
    for idoscar,doscar in enumerate(doscars):

        #calculating if DOSCAR is present at ldos/DOSCAR
        if os.path.isfile(doscar):
            with open(doscar, 'r') as file: line_count = len(file.readlines())
            if (line_count>0):
                energy, total_dos, d_ldos, d_ldos_smoothed = parse_doscar(doscar,sigma) #parsing doscar
                ##calculating descriptors and plotting
                g_occ, g_bond, dip = plot_dos(energy, total_dos, d_ldos, d_ldos_smoothed, sigma, output_file=ldosplot)
                g_occ_over_g_bond = g_occ/g_bond

# %% calling main
if __name__ == "__main__":
    main()
