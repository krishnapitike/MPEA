import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import diptest

def dip_test(x):
    """
    Calculate Hartigan's Dip Test for unimodality.
    """
    x = np.sort(x)  # Sort the input data
    n = len(x)  # Number of data points

    if n < 2:
        raise ValueError("Dip test requires at least two data points.")

    # Empirical CDF
    cdf = np.arange(1, n + 1) / n

    # Calculate the dip statistic
    gcm, lcm = [], []
    dip = 0

    for i in range(1, n - 1):
        # Lower CDF slope
        lower_cdf = (cdf[i] - cdf[i - 1]) / (x[i] - x[i - 1])
        # Upper CDF slope
        upper_cdf = (cdf[i + 1] - cdf[i]) / (x[i + 1] - x[i])

        # Update dip statistic
        dip = max(dip, abs(lower_cdf - upper_cdf))
        gcm.append(lower_cdf)
        lcm.append(upper_cdf)

    return dip

def parse_doscar(filename):
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

    return energy, total_dos, d_ldos, integrated_dos, e_fermi

def plot_dos(energy, total_dos, d_ldos, e_fermi, output_file=None):
    """
    Plot the density of states (DOS).
    """

    #calculating where is pseudogap
    fermiIndex=np.argmin(np.abs(energy))
    dosmaxIndex1, dosmaxIndex2 = np.argmax(d_ldos[:fermiIndex]), np.argmax(d_ldos[fermiIndex:])+fermiIndex
    dosminIndex=np.argmin(d_ldos[dosmaxIndex1:dosmaxIndex2])
    pgIndex=dosminIndex+dosmaxIndex1
    #calculating g_occ/g_bond
    g_occ=np.sum(d_ldos[:fermiIndex])
    g_bond=np.sum(d_ldos[:pgIndex])

    #calculating hartigan's dip
    dip, pval = diptest.diptest(d_ldos)

    plt.figure(figsize=(8, 6))
    plt.plot(energy , total_dos, label="Total DOS", color="k",alpha=0.2,linestyle=':')
    plt.plot(energy , d_ldos, label="d LDOS", color="blue")
    plt.fill_between(energy[:fermiIndex] , d_ldos[:fermiIndex], color="blue", alpha=0.5, label="$g_{occ}$")
    plt.fill_between(energy[fermiIndex:pgIndex] , d_ldos[fermiIndex:pgIndex], color="red", alpha=0.5, label="$g_{bond}$")
    plt.title('$\\frac{{g_{{occ}}}}{{g_{{bond}}}} = {:0.3f} ~;~ dip= {:0.3f}$'.format(g_occ/g_bond,dip))
    #plt.plot(energy)

    plt.axvline(0, color='brown', linestyle='--', label="Fermi Level")  # Fermi level at 0
    plt.axvline(energy[pgIndex],color='black', linestyle=':', label="Pseudo gap")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Density of States (states/eV/atom)")
    #plt.title("Density of States (DOS)")
    plt.legend()
    plt.grid(alpha=0.5)

    if output_file:
        plt.savefig(output_file, dpi=300)
    plt.show()

# Path to the DOSCAR file
doscar_file = "DOSCAR"

# Parse the DOSCAR file
energy, total_dos, d_ldos, integrated_dos, e_fermi = parse_doscar(doscar_file)

# Plot the DOS
plot_dos(energy, total_dos, d_ldos, e_fermi, output_file="dos_plot.png")
