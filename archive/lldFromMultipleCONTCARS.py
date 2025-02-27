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

#calculating l21 norm
def l2_1_norm(matrix):
    # Calculate the L2 norm of each row
    #row_norms = np.linalg.norm(matrix, axis=1)
    column_norms = np.linalg.norm(matrix, axis=0)
    # Sum the row norms to get the L2,1 norm
    return np.sum(column_norms)

#calculating l21 norm (alternate definition) probably wrong
def l2_1_norm_alt(matrix):
    # Calculate the L2 norm of each row
    #row_norms = np.linalg.norm(matrix, axis=1)
    column_norms = (np.linalg.norm(matrix, axis=0))**2
    # Sum the row norms to get the L2,1 norm
    return np.sqrt(np.sum(column_norms))

#calculate ceter of mass by calculating periodic images of atoms close to periodic boundaries
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

#calculate valence electron concentration
def calculateVEC(chemSpecies):
    valenceElectrons={\
        ('Cr',6), ('Ti',4), ('V',5), ('W',6), \
        ('Nb',5), ('Zr',4), ('Ta',5), ('Mo',6),\
        ('Al',3)\
        } #valence electrons
    sum=0
    for key,value in valenceElectrons: 
        sum += chemSpecies.count(key) * value
    return(sum/len(chemSpecies)+4-6)

#calculating lattice mismatch
def calculateLatticeMismatch(chemSpecies):
    metallicRadii={\
        ('Cr',1.28), ('Ti',1.47), ('V',1.34), ('W',1.39), \
        ('Nb',1.46), ('Zr',1.60), ('Ta',1.46), ('Mo',1.39),\
        ('Al',1.43)\
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

#calculate supercell size based on lattice constants and number of atoms
def getSize(atoms):
    natm=len(atoms)
    cell=atoms.get_cell()[:]
    V=atoms.get_volume()
    a0=(V/(natm/2))**(1/3)
    size=np.rint(cell/a0)
    size=np.array(size,dtype=int)
    return(size)

#this is yet to be developed
#plotting displacements
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

#calculating lld
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

def main():

    #collecting all contcars
    #contcars=glob('**/CONTCAR',recursive=1)
    contcars=glob('**/POSCAR',recursive=1)
    #looping through all contcars
    for icontcar,contcar in enumerate(contcars):

        if ('backup' not in contcar):

            #reading the contcar
            relaxed=read(contcar)

            #since VEC and lattice mismatch only depends on chemical composition
            #calculating them only once for each composition
            if(icontcar==0):
                
                #getting chemical symbols
                chemSpecies=relaxed.get_chemical_symbols()
                

                #block to calculate valence electron concentration (VEC)
                w_vec=calculateVEC(chemSpecies)
                
                #block to calculate lattice mismatch
                latticeMismatch=calculateLatticeMismatch(chemSpecies)

                #calculate atomic displacements
                ai,d,D=getAtomicDisplacements(relaxed,contcar)

            else:
                
                #getting chemical symbols
                chemSpecies_temp=relaxed.get_chemical_symbols()
                chemSpecies = np.concatenate((chemSpecies,chemSpecies_temp), axis=0)

                #calculate atomic displacements
                ai_temp,d_temp,D_temp=getAtomicDisplacements(relaxed,contcar)

                #vertically stacking atomic displacements
                d  = np.concatenate((d,  d_temp ), axis=0)
                D  = np.concatenate((D,  D_temp ), axis=0) 
                ai = np.concatenate((ai, ai_temp), axis=0)

    plotDisplacements(ai,chemSpecies,d,D,'./CONTCAR')
    #mean_d=np.mean(d)
    #l21norm_D=l2_1_norm(D)
    #lld = w_vec * mean_d/l21norm_D

    mean_d=np.mean(d)
    l21norm_D=l2_1_norm(D)
    lld = w_vec * mean_d/l21norm_D

    #print("               lattice mismatch, $\\delta=$ {:0.3f}".format(latticeMismatch))
    #print("average of displacement norms, $\\Delta u=$ {:0.3f}".format(mean_d))
    print("          $l_{{2,1}} norm of disaplcements=$ {:0.3f}".format(l21norm_D))
    #print("                    $\\omega_\\mathrm{{VEC}}=$ {:0.3f}".format(w_vec))
    #print("                                    $lld=$ {:0.3f}".format(lld))

if __name__ == "__main__":
    main()