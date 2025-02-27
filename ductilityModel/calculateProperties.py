# %% importing packages
##################################################################################################

from sys import argv
import numpy as np
import matplotlib.pyplot as plt

import ase
from ase import Atoms
from ase.io import read, write
from ase.build import bulk
from ase.visualize import view
from ase.neighborlist import neighbor_list
from glob import glob
import pandas as pd
from scipy.integrate import trapezoid
import diptest
import os
import csv

# %% calculate ceter of mass by calculating periodic images of atoms close to periodic boundaries
##################################################################################################
def get_com(atoms,sc):
    periodic_images=[]
    positions=atoms.get_positions()
    cell=atoms.get_cell()[:]
    sa=sc[0,0]
    sb=sc[1,1]
    sc=sc[2,2]
    for i, position in enumerate(positions):
        if (position[0]<0.25*cell[0,0]/sa):
            periodic_images.append(position+cell[0])
            #print(position,position+cell[0])
        if (position[1]<0.25*cell[1,1]/sb):
            periodic_images.append(position+cell[1])
            #print(position,position+cell[1])
        if (position[2]<0.25*cell[2,2]/sc):
            periodic_images.append(position+cell[2])
            #print(position,position+cell[2])
        if (position[0]> cell[0,0]*(1-0.25/sa) ):
            periodic_images.append(position-cell[0])
            #print(position,position-cell[0])
        if (position[1]> cell[1,1]*(1-0.25/sb) ):
            periodic_images.append(position-cell[1])
            #print(position,position-cell[1])
        if (position[2]> cell[2,2]*(1-0.25/sc) ):
            periodic_images.append(position-cell[2])
            #print(position,position-cell[2])
        if (position[0]<0.25*cell[0,0]/sa and position[1]<0.25*cell[1,1]/sb):
            periodic_images.append(position+cell[0]+cell[1])
        if (position[1]<0.25*cell[1,1]/sb and position[2]<0.25*cell[2,2]/sc):
            periodic_images.append(position+cell[1]+cell[2])
        if (position[0]<0.25*cell[0,0]/sa and position[2]<0.25*cell[2,2]/sc):
            periodic_images.append(position+cell[0]+cell[2])
        if (position[0]>cell[0,0]*(1-0.25/sa) and position[1]>cell[1,1]*(1-0.25/sb)):
            periodic_images.append(position-cell[0]-cell[1])
        if (position[1]>cell[1,1]*(1-0.25/sb) and position[2]>cell[2,2]*(1-0.25/sc)):
            periodic_images.append(position-cell[1]-cell[2])
        if (position[0]>cell[0,0]*(1-0.25/sa) and position[2]>cell[2,2]*(1-0.25/sc) ):
            periodic_images.append(position-cell[0]-cell[2])
        if (position[0]<0.25*cell[0,0]/sa and position[1]<0.25*cell[1,1]/sb and position[2]<0.25*cell[2,2]/sc):
            periodic_images.append(position+cell[0]+cell[1]+cell[2])
        if (position[0]>cell[0,0]*(1-0.25/sa) and position[1]>cell[1,1]*(1-0.25/sb) and position[2]>cell[2,2]*(1-0.25/sc) ):
            periodic_images.append(position-cell[0]-cell[1]-cell[2])
    periodic_images=np.array(periodic_images)
    com=np.vstack((positions,periodic_images)).mean(axis=0)
    return(com)


# %% parcing doscar
##################################################################################################
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

    return energy, total_dos, d_ldos

# %% l21 norm definition1
##################################################################################################
def l2_1_norm(matrix):
    # Calculate the L2 norm of each column
    column_norms = np.linalg.norm(matrix, axis=0)
    # Sum the row norms to get the L2,1 norm
    return np.sum(column_norms)

# %% l21 norm definition2
##################################################################################################
def l2_1_norm_alt(matrix):
    # Calculate the L2 norm of each row
    #row_norms = np.linalg.norm(matrix, axis=1)
    column_norms = (np.linalg.norm(matrix, axis=0))**2
    # Sum the row norms to get the L2,1 norm
    return np.sqrt(np.sum(column_norms))

# %% calculate valence electron concentration
##################################################################################################
def calculateVEC(chemSpecies):
    valenceElectrons={\
        ('Cr',6), ('Ti',4), ('V',5), ('W',6), \
        ('Nb',5), ('Zr',4), ('Ta',5), ('Mo',6),\
        ('Al',3), ('Hf',4), ('Re',7) \
        } #valence electrons
    sum=0
    for key,value in valenceElectrons: 
        sum += chemSpecies.count(key) * value
    return(sum/len(chemSpecies))

#Cr	Hf	Mo	Nb	Re	Ta	Ti	V	W	Zr

# %% calculating lattice mismatch
##################################################################################################
def calculateLatticeMismatch(chemSpecies):
    #https://pubs.acs.org/doi/abs/10.1021/j100785a001
    #https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)#Note_b
    metallicRadii={\
        ('Cr',1.28), ('Ti',1.47), ('V',1.34), ('W',1.39), \
        ('Nb',1.46), ('Zr',1.60), ('Ta',1.46), ('Mo',1.39),\
        ('Al',1.43), ('Hf',1.59), ('Re',1.37) \
        } #angstroms
    natm=len(chemSpecies)
    averageRadii=0
    for key,metallicRadius in metallicRadii: 
        averageRadii += metallicRadius*chemSpecies.count(key)/natm
    sum=0
    for key,metallicRadius in metallicRadii: 
        atomicFraction=chemSpecies.count(key)/natm
        sum += atomicFraction*(1-metallicRadius/averageRadii)**2
    latticeMismatch=100*np.sqrt(sum)
    return(latticeMismatch)

# %% calculating formation enthalpy
##################################################################################################
def calculateFormationEnthalpy(chemSpecies,contcar):
    #parcing outcar and getting dft energy
    outcar=contcar.replace('CONTCAR','OUTCAR') #creaing path to outcar
    # Open the file and saving the last occurrence of "free energy"
    last_line = None
    with open(outcar, 'r') as file:
        for line in file:
            if "free  energy" in line:
                E = float(line.split()[4])  # Update whenever "free  energy" is found
    #chemical potential = energy/atom in its ground state
    #Cr, V, W = BCC and Ti=?
    chemicalPotentials={\
        ('Cr',-19.01176346/2), ('Ti',-15.52404561/2), ('V',-17.88015161/2), ('W',-26.02666999/2), \
        } #eV/atom
    natm=len(chemSpecies)
    totalChemicalPotential=0
    for key,value in chemicalPotentials: 
        totalChemicalPotential += chemSpecies.count(key) * value
    formationEnthalpy=(E-totalChemicalPotential)/natm
    return(formationEnthalpy)

# %% calculate supercell size based on lattice constants and number of atoms
##################################################################################################
def getSize(atoms):
    natm=len(atoms)
    cell=atoms.get_cell()[:]
    V=atoms.get_volume()
    a0=(V/(natm/2))**(1/3)
    size=np.rint(cell/a0)
    size=np.array(size,dtype=int)
    return(size)

# %% calculating lld
def getAtomicDisplacements(atoms,contcar):

    #calculating supercell size
    sc=getSize(atoms)
    chemSpecies=atoms.get_chemical_symbols()

    #creating ideal bcc supercell with relaxed dimensions
    relaxedCell=atoms.get_cell()[:]
    natm=len(atoms)
    bcc_uc = bulk('Cr', 'bcc', a=1,cubic=1)
    bcc_sc = bcc_uc.repeat((sc[0,0],sc[1,1],sc[2,2]))
    symmetric_positions=bcc_sc.get_scaled_positions()
    bcc_sc.set_cell(relaxedCell)
    bcc_sc.set_scaled_positions(symmetric_positions)

    #shifting the center of mass of ideal cell to match with relaxed structure
    ideal_com=get_com(bcc_sc,sc)
    #print("  ideal_com before resetting =",ideal_com)
    relaxed_com=get_com(atoms,sc)
    #bcc_sc.set_positions(bcc_sc.get_positions()+relaxed_com-ideal_com)
    ideal_com=get_com(bcc_sc,sc)

    #print("  ideal_com after  resetting =",ideal_com)
    #print("                 relaxed_com =",relaxed_com)

    #checking if the center of mass of both cells match
    #disp=relaxed_com-ideal_com
    #if (np.sqrt(disp[0]**2+disp[1]**2+disp[2]**2)>1E-8):
    #    print(np.sqrt(disp[0]**2+disp[1]**2+disp[2]**2))
    #    exit("center of mass is not matched\nsomething is wrong\nexiting")
    
    #calculating rcut
    relaxed_cell=atoms.get_cell()[:]
    rcut=(np.sqrt(3)/4)*(relaxed_cell[0,0]/sc[0,0]+relaxed_cell[1,1]/sc[1,1]+relaxed_cell[2,2]/sc[2,2])/3
    #print("rcut=",rcut)

    #combining Atoms objects to identify the ideals coordinates as first neirest neighbors
    combined=atoms+bcc_sc

    #calculating atomic displacements
    ai,aj,d,D = neighbor_list('ijdD', combined, rcut)
    #collecting only first half of the displacements
    #because we combined atoms+bcc_sc
    ai=ai[:natm]
    aj=aj[:natm]
    d=d[:natm]
    D=D[:natm]

    #block to plot histogram of displacements and
    #to identify which species has highest atomic displacements based on j
    plotDisplacements(ai,chemSpecies,d,D,contcar)

    #checking if only first nearest neighbor atoms are identified
    coordination = np.bincount(ai)
    if (np.sum(coordination)==len(atoms)):
        #print("Identified the centrosymmetric positions")
        dummy=1
    else:
        print(coordination)
        exit("Exiting. Decrease rcut value in the code")

    return(ai,d,D)

# %% plotting dos
##################################################################################################
def plot_dos(energy, total_dos, d_ldos, chemFormula, output_file=None):
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
    dip, pval = diptest.diptest(d_ldos)

    plt.figure(figsize=(8, 6))
    plt.plot(energy , total_dos, label="Total DOS", color="k",alpha=0.2,linestyle=':')
    plt.plot(energy , d_ldos, label="d LDOS", color="blue")
    plt.fill_between(energy[:fermiIndex] , d_ldos[:fermiIndex], color="blue", alpha=0.5, label="$g_{occ}$")
    plt.fill_between(energy[fermiIndex:pgIndex] , d_ldos[fermiIndex:pgIndex], color="red", alpha=0.5, label="$g_{bond}$")
    plt.title('{:s}: $\\frac{{g_{{occ}}}}{{g_{{bond}}}} = {:0.3f} ~;~ dip= {:0.3f}$'.format(chemFormula,g_occ/g_bond,dip))
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

    return(g_occ,g_bond,dip)

# %% #plotting displacements
##################################################################################################
def plotDisplacements(ai,chemSpecies,d,D,contcar):
    #contcar has complete path to contcar
    dir=contcar.rpartition('/')[0]
    #print(dir)
    if dir=='':
        outputFigure='displacements.png'
        outputCSV='displacements.csv'
        outputIndex='u_vs_index.png'
    else:
        outputFigure=dir+'/displacements.png'
        outputCSV=dir+'/displacements.csv'
        outputIndex=dir+'u_vs_index.png'
    #plotting histogram of atomic displacements
    plt.figure()
    plt.hist(d)
    #print("mu={:0.2f} sigma={:0.2f}".format(np.mean(d),np.std(d)))
    plt.ylabel('number of atoms')
    plt.xlabel('$d$ [$\mathrm{\AA}$]')
    plt.title("Histogram of atomic displacements\n $\mu=${:0.2f} $\sigma=${:0.2f}".format(d.mean(),d.std()))
    plt.savefig(outputFigure)
    plt.show()

    #plotting individual displacements
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1,len(D)+1),D[:,0],label='x',c='k',marker='o')
    plt.plot(np.arange(1,len(D)+1),D[:,1],label='y',c='r',marker='s')
    plt.plot(np.arange(1,len(D)+1),D[:,2],label='z',c='b',marker='d')
    #plt.xlim(0,73)
    plt.ylim(-0.4,0.4)
    plt.ylabel('$\\Delta u$')
    plt.xlabel('atom index')
    plt.savefig(outputIndex)
    plt.legend(frameon=0)
    plt.show()

    combinedData=np.transpose([ai,chemSpecies,d,D[:,0],D[:,1],D[:,2]]) #(np.shape(ai),np.shape(aj),np.shape(d),np.shape(D))
    df = pd.DataFrame(combinedData)
    #combinedData=np.concatenate((ai,aj,d,D), axis=1)
    # Save the array as a CSV file
    #np.savetxt(outputCSV, combinedData, delimiter=",",fmt='%.1f,%s,%.2f,%.2f,%.2f,%.2f')#, fmt=["%2d","%2s","%0.8e","%0.8e","%0.8e","%0.8e"])
    df.to_csv(outputCSV, index=False,header=['atom','type','|d|','dx','dy','dz'])
    return()

# %% # Write the data to a CSV file
##################################################################################################
def write_csv(file_path,data):
    header=['chemical formula','formation enthalpy, E_f','lattice constant, a0',\
          'lattice mismatch, delta','valence electron concentration, vec',\
          'average displacement, <u>','displacement l21 norm, l21_norm_u', '<u>/l21_norm_u',\
          'local lattice displacement, lld','total occupied states, g_occ',\
          'total bonding states, g_bond','g_occ/g_bond',"Hartigan's dip, dip", 'epsilon_p']
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(header)

        # Write the data rows
        writer.writerows(data)
    return()

# %% # epsilon_p model
##################################################################################################
#epsilon_p = peak true strain before fracture
#BSD = bonding state depletion = gocc/gbond
def ep_polymodel(BSD):
    epsilon_p=-13706.12*(BSD)**3 + 38196.22*(BSD)**2 - 35519.70*(BSD) + 11025.99
    if (epsilon_p>1):
        return( epsilon_p )
    else:
        return( None )


# %% main function
##################################################################################################
def main():

    #setting header of data
    data=[]

    #collecting all contcars
    contcars=glob('**/CONTCAR',recursive=1)

    #looping through all contcars
    for icontcar,contcar in enumerate(contcars):

        #not considering backup contcars
        if ('backup' in contcar):
            dummy=1
        
        #calculating doscar and plotting
        elif ('ldos' in contcar):
            dummy=2
        
        #calculating lld and plotting
        else:
            relaxed = read(contcar) #reading contcar

            #getting chemical symbols
            chemSpecies=relaxed.get_chemical_symbols()
            chemFormula=relaxed.get_chemical_formula()

            #calculating formation enthalpy
            formationEnthalpy=calculateFormationEnthalpy(chemSpecies,contcar)

            #calculating a0
            natm=len(relaxed)
            V0=relaxed.get_volume()
            a0=(V0/(natm/2))**(1/3)

            #calculating ldos, g_occ/g_bond, dip and plotting ldos
            doscar=contcar.replace('CONTCAR','/ldos/DOSCAR') #creaing path to doscar
            ldosplot=contcar.replace('CONTCAR','ldos/ldos.png') #creating path to plot
            #calculating if DOSCAR is present at ldos/DOSCAR 
            if os.path.isfile(doscar):
                with open(doscar, 'r') as file: line_count = len(file.readlines())
                if (line_count>0):
                    energy, total_dos, d_ldos = parse_doscar(doscar) #parsing doscar
                    ##calculating descriptors and plotting
                    g_occ, g_bond, dip = plot_dos(energy, total_dos, d_ldos, chemFormula, output_file=ldosplot)
                    g_occ_over_g_bond = g_occ/g_bond
            #setting electronic descriptors to None if DOSCAR is absent at ldos/DOSCAR
            else:
                g_occ, g_bond, dip = None, None, None
                g_occ_over_g_bond= None

            #block to calculate w_VEC and valence electron concentration (VEC)
            vec=calculateVEC(chemSpecies)
            w_vec=vec-2
            
            #block to calculate lattice mismatch
            latticeMismatch=calculateLatticeMismatch(chemSpecies)

            #calculate atomic displacements
            ai,d,D=getAtomicDisplacements(relaxed,contcar)

            #lld
            average_d=np.mean(d)
            l21_norm=l2_1_norm(D)
            lld=w_vec*average_d/l21_norm

            data.append([chemFormula,\
                        formationEnthalpy,\
                        a0,\
                        latticeMismatch,\
                        vec,\
                        average_d,\
                        l21_norm,\
                        average_d/l21_norm,\
                        lld,\
                        g_occ,\
                        g_bond,\
                        g_occ_over_g_bond,\
                        dip,\
                        ep_polymodel(g_occ_over_g_bond)])
    write_csv('./data.csv',data)

# %% calling main
if __name__ == "__main__":
    main()