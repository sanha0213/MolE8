#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:55:11 2021

MolE8_v1.py

Reads in .xyz files and predicts energy of the molecules using trained neural 
networks or kernel ridge regression or multivariable linear regression machine 
learning models.

@author: sanha
University of Cambridge

"""

##############################################################################
# Import Modules
##############################################################################


from keras.models import load_model
import sys
import numpy as np
import pickle
import openbabel as ob
import argparse
import logging
import time
start_time = time.time()
from datetime import datetime
datetime_now = datetime.now()
formatted_datetime = datetime_now.strftime("%Y %b %d %H:%M:%S")


##############################################################################
# Settings
##############################################################################

'''

Code setup definitions:

dictionary_loc        directory where traind ML models are saved
dictionary_name       directory where the dictionaries required for feature generation algorithms are stored

'''

# -- ML model locations and names
database_loc = 'ML_Models'
dictionary_loc = 'Feat_Dicts'
NN_EModelName = 'NN_Energy_Model.h5'
KRR_EModelName = 'KRR_Energy_Model.plk'
MLR_EModelName = 'MLR_Energy_Model.plk'
NN_GModelName = 'NN_FreeEnergy_Model.h5'
KRR_GModelName = 'KRR_FreeEnergy_Model.plk'


##############################################################################
# Do not change the below settings
##############################################################################

scaling_factor = 627.509608030593 # kcal/mol
atomweight = 100
bondweight = 1

PTable = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', \
          'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', \
          'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', \
          'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', \
          'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', \
          'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', \
          'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']


##############################################################################
# Load information
##############################################################################


parser = argparse.ArgumentParser(description='Generate features for machine learning MolE8.py code') # create parser object
parser.add_argument('filename', help=".xyz file to analyse the energy or free energy") 
parser.add_argument("model_option", help='choose between NN, KRR or MLR machine learning models', action='store')
parser.add_argument("energy_option", help='choose between energy (E) or free energy (G)', action='store')
args = parser.parse_args()

# -- check the calculation options are entered correctly

model_option_correct = ['NN', 'KRR', 'MLR']
energy_option_correct = ['E', 'G']

Filename = args.filename
logging.basicConfig(filename='./MolE8_v1_'+Filename[:-4]+'.log', level=logging.DEBUG, format='%(message)s', filemode='w')

if args.model_option not in model_option_correct:
    print('model_option not entered correctly. Use -n for neural networks (NN), -k for kernel ridge regression (KRR) and -m for multivariable linear regression (MLR) machine learning models')
    logging.info('model_option not entered correctly. Use -n for neural networks (NN), -k for kernel ridge regression (KRR) and -m for multivariable linear regression (MLR) machine learning models')
    sys.exit()

if args.energy_option not in energy_option_correct:
    print('energy_option not entered correclty, use -e for energy prediction and -f for free energy prediction')
    logging.info('energy_option not entered correclty, use -e for energy prediction and -f for free energy prediction')
    sys.exit()


Model_Option = args.model_option
Energy_Option = args.energy_option


if Model_Option == 'MLR' and Energy_Option == 'G':
    print('Multivariable Linear Regression is only available for energy predicitons not for free energy predictions')
    logging.info('Multivariable Linear Regression is only available for energy predicitons not for free energy predictions')
    sys.exit()

if Energy_Option == 'E':
    dictionary_name = 'E'
elif Energy_Option == 'G':
    dictionary_name = 'G'
else:
    print('energy_option not entered correclty, use -e for energy prediction and -f for free energy prediction')
    logging.info('energy_option not entered correclty, use -e for energy prediction and -f for free energy prediction')
    sys.exit()


##############################################################################
# Load Data
##############################################################################


print('')      
print('**********************************************************************')
print('')
print('MolE8_v1.py')
print('')
print('Reads in .xyz files and predicts energy of the molecules using trained neural ')
print('networks or kernel ridge regression or multivariable linear regression machine')
print('learning models.')
print('')
print('Author: Sanha Lee')
print('University of Cambridge')
print('')
print('**********************************************************************')
print('')
print('')
print('ML model to be used:')
print(Model_Option)
print('Calculating following property:')
print(Energy_Option)
print('Using following xyz file:')
print(Filename)
print('')
print('Loading features definition dictionaries')


logging.info('')      
logging.info('**********************************************************************')
logging.info('')
logging.info('MolE8_v1.py')
logging.info('')
logging.info('Reads in .xyz files and predicts energy of the molecules using trained neural ')
logging.info('networks or kernel ridge regression or multivariable linear regression machine')
logging.info('learning models.')
logging.info('')
logging.info('Author: Sanha Lee')
logging.info('University of Cambridge')
logging.info('')
logging.info('**********************************************************************')
logging.info('')
logging.info('')
logging.info('ML model to be used:')
logging.info(Model_Option)
logging.info('Calculating following property:')
logging.info(Energy_Option)
logging.info('Using following xyz file:')
logging.info(Filename)
logging.info('')
logging.info('Loading features definition dictionaries')


with open(dictionary_loc+'/'+dictionary_name+'_BondFeaturesDef.plk', "rb") as read_file1:
    BondFeaturesDef = pickle.load(read_file1)
    BondFeaturesKeys = list(BondFeaturesDef.keys())

with open(dictionary_loc+'/'+dictionary_name+'_AngleFeaturesDef.plk', "rb") as read_file2:
    AngleFeaturesDef = pickle.load(read_file2)
    AngleFeaturesKeys = list(AngleFeaturesDef.keys())

with open(dictionary_loc+'/'+dictionary_name+'_DihedralFeaturesDef.plk', "rb") as read_file3:
    DihedralFeaturesDef = pickle.load(read_file3)
    DihedralFeaturesKeys = list(DihedralFeaturesDef.keys())

with open(dictionary_loc+'/'+dictionary_name+'_HbondFeaturesDef.plk', "rb") as read_file4:
    HbondFeaturesDef = pickle.load(read_file4)
    HbondFeaturesKeys = list(HbondFeaturesDef.keys())

with open(dictionary_loc+'/'+dictionary_name+'_MaxAtomTypesDef.plk', "rb") as read_file5:
    Max_Atomtypes = pickle.load(read_file5)

print('')
print('Loaded'+dictionary_name+'_BondFeaturesDef:')
print(BondFeaturesDef.keys())
print('')
print('Loaded'+dictionary_name+'_AngleFeaturesDef:')
print(AngleFeaturesDef.keys())
print('')
print('Loaded'+dictionary_name+'_DihedralFeaturesDef:')
print(DihedralFeaturesDef.keys())
print('')
print('Loaded'+dictionary_name+'_HbondFeaturesDef:')
print(HbondFeaturesDef.keys())
print('')
print('Loaded Max_Atomtypes:')
print(Max_Atomtypes.keys())

logging.info('')
logging.info('Loaded'+dictionary_name+'_BondFeaturesDef:')
logging.info(BondFeaturesDef.keys())
logging.info('')
logging.info('Loaded'+dictionary_name+'_AngleFeaturesDef:')
logging.info(AngleFeaturesDef.keys())
logging.info('')
logging.info('Loaded'+dictionary_name+'_DihedralFeaturesDef:')
logging.info(DihedralFeaturesDef.keys())
logging.info('')
logging.info('Loaded'+dictionary_name+'_HbondFeaturesDef:')
logging.info(HbondFeaturesDef.keys())
logging.info('')
logging.info('Loaded Max_Atomtypes:')
logging.info(Max_Atomtypes.keys())

##############################################################################
# Functions
##############################################################################

def LoadDBFromPickle(pklfile):
     with open(database_loc+'/'+pklfile, "rb") as read_file:
         return pickle.load(read_file)


def BuildOBMol(atoms, coords):
    '''
    Generate OBmol object
    
    '''

    mol = ob.OBMol()
    for anum, acoords in zip(atoms, coords):
        atom = ob.OBAtom()
        atom.thisown = False
        atom.SetAtomicNum(GetAtomNum(anum))
        atom.SetVector(acoords[0], acoords[1], acoords[2])
        mol.AddAtom(atom)

    # Restore the bonds
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()

    # mol.Kekulize()

    return mol


def ExtractEnergy(GDBmolecule):
    '''
    Returns energy of GDB molecule
    Change the scaling factor to the desired units
    
    '''

    energy = float(GDBmolecule.G298) * scaling_factor

    return energy



def GetAtomSymbol(AtomNum):

    if AtomNum > 0 and AtomNum < len(PTable):
        return PTable[AtomNum-1]
    else:
        print("No such element with atomic number " + str(AtomNum))
        return 0


def GetAtomNum(AtomSymbol):

    if AtomSymbol in PTable:
        return PTable.index(AtomSymbol)+1
    else:
        print("No such element with symbol " + str(AtomSymbol))
        return 0


def GenerateFeatures(OBmol_data, bonddict, angledict, dihedraldict, maxatomtypes, hbonddict):
    '''
    
    Creates fecture matrix X for all OBmol database from the bond, angle, 
    dihedral dictionaries
    
    bondict is the list of maximas and minimas found in the bond distribution 
    for all bonds found in the entire dataset. Likewise for angledict and
    dihedraldict.
    
    Note: the bonddict, angledict, dihedraldict dictionary values are in the 
    format [[maximas], [minimas]]
    
    OBmol_data += [[MolBondData, MolAngleData, MolDihedralData, MolAtomData, MolHbondData, MolNHatomData]]
    
    '''
    
    X = []
    readable_X = []
    
    
    # Generate readable X vector for the output:
    
    for bonddict_key, bonddict_values in bonddict.items():
        feat_vec = [bonddict_key]*len(bonddict_values[0])
        readable_X += feat_vec
    
    for angledict_key, angledict_values in angledict.items():
        feat_vec = [angledict_key]*len(angledict_values[0])
        readable_X += feat_vec
        
    for dihedraldict_key, dihedraldict_values in dihedraldict.items():
        feat_vec = [dihedraldict_key]*len(dihedraldict_values[0])
        readable_X += feat_vec
        
    for atom_key, atom_values in maxatomtypes.items():
        feat_vec = [atom_key]
        readable_X += feat_vec
    
    for Hbond_key, Hbond_values in hbonddict.items():
        feat_vec = [Hbond_key]*len(Hbond_values[0])
        readable_X += feat_vec
        
    feat_vec = ['NH1', 'NH2', 'NH3']
    readable_X += feat_vec
    
    
    # -- Now generate actual feature vector X
    for OBmol in OBmol_data:
        
        # -- Extract all data for OBmol
        MolBondDict = OBmol[0]
        MolBondKeys = MolBondDict.keys()
        MolAngleDict = OBmol[1]
        MolAngleKeys = MolAngleDict.keys()
        MolDihedralDict = OBmol[2]
        MolDihedralKeys = MolDihedralDict.keys()
        MolAtomDict = OBmol[3]
        MolAtomKeys = list(MolAtomDict.keys())
        MolHbondDict = OBmol[4]
        MolHbondKeys = list(MolHbondDict.keys())
        MolNHatomDict = OBmol[5]
        MolNHatomKeys = list(MolNHatomDict.keys())
        MolNHatomValues = list(MolNHatomDict.values())
        Mol_X = []
        
        # -- Bond Features Vector Generation
        # -- (use 'print' statements to test the code is working properly)
             
        for bonddict_key, bonddict_values in bonddict.items():
            
            feat_vec = [0] * len(bonddict_values[0]) # create list of zeros, same size as list of maximas
        
            if bonddict_key in MolBondKeys: # check if particular bond type from total dictionary exists in the molecule dictionary
                
                #print('molbonddict: ', MolBondDict[bonddict_key])
                
                for mol_bondvalue in MolBondDict[bonddict_key]: # for each bondlength for particular bond type...
                    if all(item <= mol_bondvalue for item in bonddict_values[1]): # check whether the bond length is greater than all values in the list of minimas
                        feat_vec[-1] += 1 * bondweight # append one to the last element of the features vec (corresponds to the maxima of greatest bond length)
                    else:
                        # MOST IMPORTANT PART
                        # returns the first index of the minima list when the total dict minima value gets larger than the molecule dict bond value
                        feat_idx = next(index for index, bond_dictvalue in enumerate(bonddict_values[1]) if bond_dictvalue > mol_bondvalue) 
                        # else find the index of the minima where the bond length under consideration becomes larger 
                        feat_vec[feat_idx] += 1 * bondweight
                        
                Mol_X += feat_vec
                
            else:
                #print('no such bond exist')
                Mol_X += feat_vec
            
            #print(bonddict_key, bonddict_values)
            #print(feat_vec)
            
            #print(feat_vec)
        #if FileName in fileswanted:
        #    print('Mol_X: ', Mol_X)
        

        #print(MolAngleDict)
        # -- Angle Features Vector Generation
        for angledict_key, angledict_values in angledict.items():
            
            #print('angledict_key: ', angledict_key)
            #print('maximas: ', angledict_values[0])
            
            feat_vec = [0] * len(angledict_values[0])
            
            if angledict_key in MolAngleKeys:    
                
                for mol_anglevalue in MolAngleDict[angledict_key]:
                    if all(item <= mol_anglevalue for item in angledict_values[1]):
                        feat_vec[-1] += 1 # append one to the last element of the feature vec
                    else:
                        feat_idx = next(index for index, angle_dictvalue in enumerate(angledict_values[1]) if angle_dictvalue  > mol_anglevalue)
                        feat_vec[feat_idx] += 1
                
                Mol_X += feat_vec
                
            else:
                Mol_X += feat_vec
        
            #print(angledict_key, angledict_values)
            #print(feat_vec)
        
        #print('Mol_X: ', Mol_X)
        
        # -- Dihedral Features Vector Generation

        #print(MolDihedralDict)
        for dihedraldict_key, dihedraldict_values in dihedraldict.items():
            
            #print('dihedraldict_key: ', dihedraldict_key)
            #print('maximas: ', dihedraldict_values[0])
            
            feat_vec = [0] * len(dihedraldict_values[0])
            
            if dihedraldict_key in MolDihedralKeys:
                
                for mol_dihedralvalue in MolDihedralDict[dihedraldict_key]:
                    if all(item <= mol_dihedralvalue for item in dihedraldict_values[1]):
                        feat_vec[-1] += 1
                    else:
                        feat_idx = next(index for index, dihedral_dictvalue in enumerate(dihedraldict_values[1]) if dihedral_dictvalue > mol_dihedralvalue)
                        feat_vec[feat_idx] += 1
    
                Mol_X += feat_vec
            
            else:
                Mol_X += feat_vec
            
            #print(dihedraldict_key, dihedraldict_values)
            #print(feat_vec)
    
        #print('Mol_X: ', Mol_X)
        
        # -- Atom Feature Vector Generation
        feat_vec = [0] * len(maxatomtypes)
        
        for atom_key, atom_values in maxatomtypes.items():
            
            #print('atom_key: ', atom_key)
            
            maxatomkeys = list(maxatomtypes.keys())    
            
            if atom_key in MolAtomKeys:
                feat_vec[ maxatomkeys.index(atom_key) ] = MolAtomDict[atom_key] * atomweight

        Mol_X += feat_vec
        
        
        # -- Hbond Feature Vector Generation
        for hbonddict_key, hbonddict_values in hbonddict.items():
            
            #print('hbonddict_key: ',hbonddict_key)
            #print('maximas: ', hbonddict_values[0])
            
            feat_vec = [0] * len(hbonddict_values[0])
            
            if hbonddict_key in MolHbondKeys:
                
                for mol_hbondvalue in MolHbondDict[hbonddict_key]:
                    if all(item <= mol_hbondvalue for item in hbonddict_values[1]):
                        feat_vec[-1] += 1
                    else:
                        feat_idx = next(index for index, hbond_dictvalue in enumerate(hbonddict_values[1]) if hbond_dictvalue > mol_hbondvalue)
                        feat_vec[feat_idx] += 1
    
                Mol_X += feat_vec
            
            else:
                Mol_X += feat_vec
    

        # -- NHatom Feature Vector Generation
        feat_vec = [0] * len(MolNHatomKeys)

        for index in range(len(MolNHatomKeys)):
            
            feat_vec[index] = MolNHatomValues[index]
    
        Mol_X += feat_vec
        
        
        #print('MaxAtomTypes: ', maxatomtypes)
        #print('MolAtomDict: ', MolAtomDict)
        #print('MolAtomKeys: ', MolAtomKeys)
        #print('feat_vec: ', feat_vec)
        
        

        X += [Mol_X]
        
        sys.stdout.write('\r')
        sys.stdout.write(str(round(100.0*float(OBmol_data.index(OBmol))/float(len(OBmol_data)), 1))+'% : ' + str(OBmol_data.index(OBmol))+' out of '+str(len(OBmol_data))+' molecules')
        sys.stdout.flush()
    
    return np.array(X), readable_X


# Hbonders are atoms the SelectHBonds will detect as H-bonding atoms
Hbonders = ['O', 'N']

def SelectHBonds(mol):
    '''
    Function to detect H bonds in OBmol and generate features
    Edit 'Hbonders' above to change the H-bonding atoms
    
    '''
    natoms = mol.NumAtoms()
    AcidHs = []
    ENatoms = []

    # Go through all atoms
    # For all protons, check if covalently bonded to an EN atom (defined by Hbonders list)
    # Make a list of all acidic H atoms
    for a in range(1, natoms+1):
        atom = mol.GetAtom(a)
        if atom.GetAtomicNum() != 1:
            continue
        else:
            for NbrAtom in ob.OBAtomAtomIter(atom): # iterate over all neighbouring atoms
                if GetAtomSymbol(NbrAtom.GetAtomicNum()) in Hbonders:
                    AcidHs.append(a)

    # Make a list of all EN atoms (defined by Hbonders list) in molecule
    ENatoms = []
    for a in range(1, natoms+1):
        atom = mol.GetAtom(a)
        if GetAtomSymbol(atom.GetAtomicNum()) in Hbonders:
            ENatoms.append(a)
    

    # We take all the detected X-H...Y distances and classify them
    # as either covalent, H-bonded or non-bonded
    ActiveHBonds = []
    ActiveHDists = []
    for Ha in AcidHs: # for all Hs attached to e-neg atoms [[H1-E1, H1-E2, ...], [H2-E1, H2-E2, ...], etc]

        for ENa in ENatoms: # append distances to each e-neg attoms
            H_dist = GetDistance(mol, [Ha, ENa])
            
            if H_dist > 1.3 and H_dist < 2.6:
                
                ActiveHDists += [H_dist]
                
                for NbrAtom in ob.OBAtomAtomIter(mol.GetAtom(Ha)):
                    bond = GetAtomSymbol(NbrAtom.GetAtomicNum())+\
                           GetAtomSymbol(mol.GetAtom(Ha).GetAtomicNum())+'-'+\
                           GetAtomSymbol(mol.GetAtom(ENa).GetAtomicNum())
                    ActiveHBonds.append(bond)

    return ActiveHBonds, ActiveHDists


def GetDistance(mol, atoms):
    '''
    Returns distance of two specific atoms in OBmol
    
    '''

    a1 = mol.GetAtom(atoms[0])

    return a1.GetDistance(atoms[1])


def ExtractFilename(GDBmolecule):
    '''
    Returns the output filename of GDB molecule
    
    '''
    outputfilename = GDBmolecule.geomfilename
    outputfilenamelist = outputfilename.split('/')
    
    return outputfilenamelist[-1]



###############################################################################
# Main Code Part 1: Generate Features
###############################################################################

OBmol_Objects = []      # list of all molecules in the xyz file as OBmol object
OBmol_atoms = []        # list of atoms for each molecule in the database
OBmol_coords = []       # list of coordinates for each molecule in the database
OBmol_data = []         # list of list containing bond, angle, dihedral etc data for each molecules

OBMol = ob.OBMol()
conv = ob.OBConversion()
conv.SetInFormat('xyz')
conv.ReadFile(OBMol, Filename)

mol_atoms = []
mol_coords = []

# Extract molecule data from the first molecule in the xyz file
for OBmolatom in ob.OBMolAtomIter(OBMol):
    atomtype = str(OBmolatom.GetType())
    atomtype = atomtype[0]
    coords = [OBmolatom.GetX(), OBmolatom.GetY(), OBmolatom.GetZ()]
    mol_atoms += [atomtype]
    mol_coords += [coords]
    
OBmol_Objects += [BuildOBMol(mol_atoms, mol_coords)]
OBmol_atoms += [mol_atoms]
OBmol_coords += [mol_coords]


while (conv.Read(OBMol)): # loop though rest of the molecules in the xyz file
    
    mol_atoms = []
    mol_coords = []
    
    for OBmolatom in ob.OBMolAtomIter(OBMol):
        atomtype = str(OBmolatom.GetType())
        atomtype = atomtype[0]
        coords = [OBmolatom.GetX(), OBmolatom.GetY(), OBmolatom.GetZ()]
        mol_atoms += [atomtype]
        mol_coords += [coords]
    
    OBmol_Objects += [BuildOBMol(mol_atoms, mol_coords)]
    OBmol_atoms += [mol_atoms]
    OBmol_coords += [mol_coords]


print('')
print('Reading xyz file: '+Filename)
print('Read in '+str(len(OBmol_Objects))+' molecules')

print('')
print('Reading bonds, angles and torsions for all OBmol objects')

logging.info('')
logging.info('Reading xyz file: '+Filename)
logging.info('Read in '+str(len(OBmol_Objects))+' molecules')

logging.info('')
logging.info('Reading bonds, angles and torsions for all OBmol objects')


for OBmol in OBmol_Objects:
    
    # -- Extract data for each molecules
    MolBondData = dict()
    MolAngleData = dict()
    MolDihedralData = dict()
    MolAtomData = dict()
    MolHbondData = dict()
    MolNHatomData = {'NH1':0,'NH2':0,'NH3':0}
    
    
    # -- Extract N atoms with one, two and three H atoms
    for OBmolatom in ob.OBMolAtomIter(OBmol): # iterate through all atoms in OBmol
        valencedata = str(OBmolatom.GetValence())
        atomdata = str(OBmolatom.GetType())
        
        if 'N' in atomdata:
            
            number_of_neighbourH = 0
            
            for neighbour_atom in ob.OBAtomAtomIter(OBmolatom):
                neighbour_atomtype = str(neighbour_atom.GetType())
                
                if neighbour_atomtype[0] == 'H':
                    number_of_neighbourH += 1
            
            if number_of_neighbourH > 0:
                MolNHatomData['NH'+str(number_of_neighbourH)] += 1
    
    
    # -- Extract hydrogen bonding data
    OBmol_Hbondtype, OBmol_Hbonddata = SelectHBonds(OBmol)
    
    for index in range(len(OBmol_Hbondtype)):
        
        if str(OBmol_Hbondtype[index]) in MolHbondData.keys():
            MolHbondData[str(OBmol_Hbondtype[index])].append(OBmol_Hbonddata[index])
        else:
            MolHbondData[str(OBmol_Hbondtype[index])] = [OBmol_Hbonddata[index]]


    # -- Iterate over all bonds
    for OBmolbond in ob.OBMolBondIter(OBmol):
        bondtype = str(GetAtomSymbol(OBmolbond.GetBeginAtom().GetAtomicNum())) + str(GetAtomSymbol(OBmolbond.GetEndAtom().GetAtomicNum()))

        if bondtype in MolBondData.keys():
            MolBondData[bondtype].append(OBmolbond.GetLength())
        elif bondtype[::-1] in MolBondData.keys():
            MolBondData[bondtype[::-1]].append(OBmolbond.GetLength())
        else:
            if bondtype in BondFeaturesDef.keys():
                MolBondData[bondtype] = [OBmolbond.GetLength()]
            elif bondtype[::-1] in BondFeaturesDef.keys():
                MolBondData[bondtype[::-1]] = [OBmolbond.GetLength()]
            else:
                print('Bondtype does not exist in MolBondData or BondLenths')
                logging.info('Bondtype does not exist in MolBondData or BondLenths')
                exit()
                

    # -- Iterate over all angles
    for OBmolangle in ob.OBMolAngleIter(OBmol):
        
        angletype = [GetAtomSymbol(OBmol.GetAtom(OBmolangle[0]+1).GetAtomicNum()), \
                     GetAtomSymbol(OBmol.GetAtom(OBmolangle[1]+1).GetAtomicNum()), \
                     GetAtomSymbol(OBmol.GetAtom(OBmolangle[2]+1).GetAtomicNum())]  # vertex first


        # -- Test for carbon coordination:
        # create separate group depending on central carbon coordination number
        # for example CC3C means CCC bond angle where the central carbon has coordination number of 3
        if str(angletype[0]) == 'C':
            C_coordno = OBmol.GetAtom(OBmolangle[0]+1).GetValence()  # CHANGED
            angletype[0] = angletype[0] + str(C_coordno)

        if str(angletype[1]+angletype[0]+angletype[2]) in MolAngleData.keys():
            MolAngleData[str(angletype[1]+angletype[0]+angletype[2])].append(OBmol.GetAngle(\
                        OBmol.GetAtom(OBmolangle[1]+1),\
                        OBmol.GetAtom(OBmolangle[0]+1),\
                        OBmol.GetAtom(OBmolangle[2]+1)))
        elif str(angletype[2]+angletype[0]+angletype[1]) in MolAngleData.keys():
            MolAngleData[str(angletype[2]+angletype[0]+angletype[1])].append(OBmol.GetAngle(\
                        OBmol.GetAtom(OBmolangle[2]+1),\
                        OBmol.GetAtom(OBmolangle[0]+1),\
                        OBmol.GetAtom(OBmolangle[1]+1)))            
        else:
            if str(angletype[1]+angletype[0]+angletype[2]) in AngleFeaturesDef.keys():
                MolAngleData[str(angletype[1]+angletype[0]+angletype[2])] = [OBmol.GetAngle(\
                        OBmol.GetAtom(OBmolangle[1]+1),\
                        OBmol.GetAtom(OBmolangle[0]+1),\
                        OBmol.GetAtom(OBmolangle[2]+1))]
            elif str(angletype[2]+angletype[0]+angletype[1]) in AngleFeaturesDef.keys():
                MolAngleData[str(angletype[2]+angletype[0]+angletype[1])] = [OBmol.GetAngle(\
                        OBmol.GetAtom(OBmolangle[2]+1),\
                        OBmol.GetAtom(OBmolangle[0]+1),\
                        OBmol.GetAtom(OBmolangle[1]+1))]
            else:
                print('angletype does not exist in MolAngleData or AnglesTypes')
                logging.info('angletype does not exist in MolAngleData or AnglesTypes')
                exit()



    # -- Iterate over all dihedrals
    for OBdihedral in ob.OBMolTorsionIter(OBmol):
        
        dihedraltype = [GetAtomSymbol(OBmol.GetAtom(OBdihedral[0]+1).GetAtomicNum()), \
                        GetAtomSymbol(OBmol.GetAtom(OBdihedral[1]+1).GetAtomicNum()), \
                        GetAtomSymbol(OBmol.GetAtom(OBdihedral[2]+1).GetAtomicNum()), \
                        GetAtomSymbol(OBmol.GetAtom(OBdihedral[3]+1).GetAtomicNum())]    # CHANGED
        
        if str(dihedraltype[0]+dihedraltype[1]+dihedraltype[2]+dihedraltype[3]) in MolDihedralData.keys():
            MolDihedralData[str(dihedraltype[0]+dihedraltype[1]+dihedraltype[2]+dihedraltype[3])].append(abs(OBmol.GetTorsion(\
                          OBmol.GetAtom(OBdihedral[0]+1),\
                          OBmol.GetAtom(OBdihedral[1]+1),\
                          OBmol.GetAtom(OBdihedral[2]+1),\
                          OBmol.GetAtom(OBdihedral[3]+1))))            
        elif str(dihedraltype[3]+dihedraltype[2]+dihedraltype[1]+dihedraltype[0]) in MolDihedralData.keys():
            MolDihedralData[str(dihedraltype[3]+dihedraltype[2]+dihedraltype[1]+dihedraltype[0])].append(abs(OBmol.GetTorsion(\
                          OBmol.GetAtom(OBdihedral[3]+1),\
                          OBmol.GetAtom(OBdihedral[2]+1),\
                          OBmol.GetAtom(OBdihedral[1]+1),\
                          OBmol.GetAtom(OBdihedral[0]+1))))            
        else:
            if str(dihedraltype[0]+dihedraltype[1]+dihedraltype[2]+dihedraltype[3]) in DihedralFeaturesDef.keys():
                MolDihedralData[str(dihedraltype[0]+dihedraltype[1]+dihedraltype[2]+dihedraltype[3])] = [abs(OBmol.GetTorsion(\
                          OBmol.GetAtom(OBdihedral[0]+1),\
                          OBmol.GetAtom(OBdihedral[1]+1),\
                          OBmol.GetAtom(OBdihedral[2]+1),\
                          OBmol.GetAtom(OBdihedral[3]+1)))]
            elif str(dihedraltype[3]+dihedraltype[2]+dihedraltype[1]+dihedraltype[0]) in DihedralFeaturesDef.keys():
                MolDihedralData[str(dihedraltype[3]+dihedraltype[2]+dihedraltype[1]+dihedraltype[0])] = [abs(OBmol.GetTorsion(\
                          OBmol.GetAtom(OBdihedral[3]+1),\
                          OBmol.GetAtom(OBdihedral[2]+1),\
                          OBmol.GetAtom(OBdihedral[1]+1),\
                          OBmol.GetAtom(OBdihedral[0]+1)))]
            else:
                print('dihedraltype does not exist in MolDihedralData or DihedralTypes')
                logging.info('dihedraltype does not exist in MolDihedralData or DihedralTypes')
                exit()
            

    # -- Extract number of each atom types
    atoms_data = OBmol_atoms[OBmol_Objects.index(OBmol)]
    for atom in atoms_data:
        MolAtomData[atom] = MolAtomData.get(atom, 0) + 1  # '0' is the value to return if the specified key does not exist

    # -- Save all information about the molecule to OBmol_data
    OBmol_data += [[MolBondData, MolAngleData, MolDihedralData, MolAtomData, MolHbondData, MolNHatomData]]


print('')
print('The algorithm will use the following weights:')
print('bondweight: ', atomweight)
print('angleweight: ',bondweight)

# Features vector
print('')
print('Generating features matrix X for all molecules')
print('')

logging.info('')
logging.info('The algorithm will use the following weights:')
logging.info('bondweight:')
logging.info(atomweight)
logging.info('angleweight:')
logging.info(bondweight)

# Features vector
logging.info('')
logging.info('Generating features matrix X for all molecules')
logging.info('')

X, read_X = GenerateFeatures(OBmol_data, \
                     BondFeaturesDef, \
                     AngleFeaturesDef, \
                     DihedralFeaturesDef, \
                     Max_Atomtypes, \
                     HbondFeaturesDef)

print('')
print('Matrix X preview:')
print(X)
print('')
print('Readable matrix X:')
print(read_X)

logging.info('')
logging.info('Matrix X preview:')
logging.info(X)
logging.info('')
logging.info('Readable matrix X:')
logging.info(read_X)


# -- Save X vector information --

list_X = X.tolist()

X_MatrixOutput = open('./MolE8_v1_ReadableXMatrix_'+Filename[:-4]+'.txt', 'w')

for vector in list_X:
    string_vector = ' '.join(str(component) for component in vector)
    X_MatrixOutput.write(str(list_X.index(vector)+1)+'\n')
    X_MatrixOutput.write(string_vector+'\n')

X_MatrixOutput.close()

readableXfile = open('./MolE8_v1_ReadableXFeatures_'+Filename[:-4]+'.txt', 'w')
for index in range(len(read_X)):
    readableXfile.write(read_X[index]+' ')
readableXfile.close()


# -- Save X vector --

np.save('./MolE8_v1_X'+'_'+Filename[:-4], X, allow_pickle=True)

print('')
print('There are '+str(len(X[0]))+' number of features')
print('')
print('Successfully created feature vector X and target vector y and saved')
print('')

logging.info('')
logging.info('There are '+str(len(X[0]))+' number of features')
logging.info('')
logging.info('Successfully created feature vector X and target vector y and saved')
logging.info('')


###############################################################################
# Main Code Part 2: Make Predictions
###############################################################################


if Model_Option == 'NN' and Energy_Option == 'E':
    
    batch_size = 64
    X_trainstd = 41.130651628928526
    X_scaled = X/X_trainstd
    
    model = load_model('./'+database_loc+'/'+NN_EModelName)
    model.summary()
    
    predictions = model.predict(X_scaled, batch_size=batch_size)
    
    unwrapped_predictions = []
    
    for pred_list in predictions:
        unwrapped_predictions += [pred_list[0]]


elif Model_Option == 'NN' and Energy_Option == 'G':
    
    batch_size = 64
    X_trainstd = 40.42285839734875
    X_scaled = X/X_trainstd
    
    model = load_model('./'+database_loc+'/'+NN_GModelName)
    model.summary()
    
    predictions = model.predict(X_scaled, batch_size=batch_size)
    
    unwrapped_predictions = []
    
    for pred_list in predictions:
        unwrapped_predictions += [pred_list[0]]
        

elif Model_Option == 'KRR' and Energy_Option == 'E':
    
    kernel = pickle.load(open('./'+database_loc+'/'+KRR_EModelName, 'rb'))

    unwrapped_predictions = kernel.predict(X)


elif Model_Option == 'KRR' and Energy_Option == 'G':
    
    kernel = pickle.load(open('./'+database_loc+'/'+KRR_GModelName, 'rb'))

    unwrapped_predictions = kernel.predict(X)
    

elif Model_Option == 'MLR' and Energy_Option == 'E':
    
    reg = pickle.load(open('./'+database_loc+'/'+MLR_EModelName, 'rb'))
    
    unwrapped_predictions = reg.predict(X_scaled)
    
        
print('')
print('')
print('**********************************************************************')
print('')
print('Successfully calculated molecular properties')
print('')
print(Energy_Option+' predictions using '+Model_Option+' ML model:')
for energy_prediciton in unwrapped_predictions:
    print("%.2f" % np.round(energy_prediciton,2))
print('')
print('**********************************************************************')
print('')
print("--- %s seconds ---" % (time.time() - start_time))
print('')
print('End')
print('')

logging.info('')
logging.info('**********************************************************************')
logging.info('')
logging.info('Successfully calculated molecular properties')
logging.info('')
logging.info(Energy_Option+' predictions using '+Model_Option+' ML model:')
for energy_prediciton in unwrapped_predictions:
    logging.info("%.2f" % np.round(energy_prediciton,2))
logging.info('')
logging.info("--- %s seconds ---" % (time.time() - start_time))
logging.info('')
logging.info('**********************************************************************')
logging.info('')
logging.info('End')
logging.info('')


